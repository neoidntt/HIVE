"""3D volume rendering and rotating-projection GIF export (Phase 3.1).

Two rendering models are provided:

**Maximum intensity projection (MIP)** -- each output pixel takes the brightest
voxel along the view ray. Depth-agnostic and unambiguous: a bright object is
visible regardless of what lies in front of it. This is the standard way to
present fluorescence z-stacks, and it is the default. Its weakness is that it
discards depth ordering entirely, so a MIP cannot show which of two objects is
in front.

**Alpha-blended volume rendering** -- voxels are composited front-to-back with
opacity proportional to intensity, using the standard over operator (Porter &
Duff, 1984):

.. math::

    C = \\sum_i c_i \\alpha_i \\prod_{j<i} (1 - \\alpha_j)

This conveys depth and internal structure, at the cost of dim structures being
occluded by brighter ones nearer the camera.

Interpretation warning: a rotating projection of an **anisotropic** stack is
geometrically wrong unless the voxels are resampled to isotropic spacing first.
Confocal z-steps are routinely 3-10x the xy pixel size; rotating such a volume
as if it were isotropic squashes it and makes spheres render as ellipsoids that
change shape as they turn. :func:`render_rotation` resamples by default --
pass real voxel sizes and let it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .config import RenderConfig


class RenderError(RuntimeError):
    """Raised when a volume cannot be rendered."""


def _apply_colormap(projection: np.ndarray, colormap: str) -> np.ndarray:
    """Map a scalar projection to RGB via a matplotlib colormap.

    Args:
        projection: 2D array, expected in ``[0, 1]``.
        colormap: Matplotlib colormap name.

    Returns:
        uint8 RGB array of shape ``(H, W, 3)``.

    Raises:
        RenderError: If matplotlib is missing or the colormap is unknown.
    """
    try:
        import matplotlib
    except ImportError as exc:
        raise RenderError(
            "Rendering needs matplotlib for colormaps: pip install matplotlib"
        ) from exc
    try:
        cmap = matplotlib.colormaps[colormap]
    except KeyError as exc:
        available = ", ".join(sorted(matplotlib.colormaps)[:12])
        raise RenderError(
            f"Unknown colormap {colormap!r}. Try one of: {available}, ..."
        ) from exc
    rgba = cmap(np.clip(projection, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def make_isotropic(
    volume: np.ndarray, voxel_size_xy: float = 1.0, voxel_size_z: float = 1.0, order: int = 1
) -> np.ndarray:
    """Resample a volume to isotropic voxels.

    Rotating an anisotropic volume treats one voxel as one unit along every
    axis, so a stack with 0.2 um pixels and 1.0 um z-steps renders 5x
    compressed in z. Objects then visibly change shape as the view turns. This
    upsamples z (never downsamples xy, which would throw away real resolution)
    so the rendered geometry is faithful.

    Args:
        volume: 3D array ``(Z, Y, X)``.
        voxel_size_xy: Physical pixel size in XY.
        voxel_size_z: Physical z-step.
        order: Spline interpolation order; 1 (linear) is a good speed/quality
            trade-off for rendering.

    Returns:
        Resampled volume with isotropic voxels. Returned unchanged when the
        z:xy ratio is already within 1%.

    Raises:
        ValueError: If ``volume`` is not 3D or voxel sizes are not positive.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"make_isotropic expects a 3D volume, got shape {vol.shape}")
    if voxel_size_xy <= 0 or voxel_size_z <= 0:
        raise ValueError(f"Voxel sizes must be positive, got xy={voxel_size_xy}, z={voxel_size_z}")

    ratio = float(voxel_size_z) / float(voxel_size_xy)
    if abs(ratio - 1.0) < 0.01:
        return vol
    return ndi.zoom(vol, (ratio, 1.0, 1.0), order=order).astype(np.float32)


def _pad_for_rotation(volume: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Pad the rotation plane so corners are not clipped as the volume turns.

    ``ndi.rotate(reshape=False)`` keeps the array shape, so voxels rotated
    outside the original bounds are discarded -- an object's corners get
    scissored off mid-spin. Padding each rotated axis to the plane's diagonal
    guarantees every voxel stays in frame at all angles.

    Args:
        volume: 3D array.
        axes: The two axes defining the rotation plane.

    Returns:
        Padded volume.
    """
    diagonal = int(np.ceil(np.hypot(volume.shape[axes[0]], volume.shape[axes[1]])))
    pad_width = [(0, 0)] * volume.ndim
    for axis in axes:
        total = max(0, diagonal - volume.shape[axis])
        pad_width[axis] = (total // 2, total - total // 2)
    return np.pad(volume, pad_width, mode="constant", constant_values=0)


def _project(volume: np.ndarray, mode: str, alpha_scale: float) -> np.ndarray:
    """Collapse a volume to a 2D image along the viewing axis (axis 0).

    Args:
        volume: 3D array in ``[0, 1]``.
        mode: ``"mip"`` or ``"alpha"``.
        alpha_scale: Opacity multiplier for ``"alpha"``.

    Returns:
        2D float32 projection in ``[0, 1]``.

    Raises:
        RenderError: On an unknown mode.
    """
    if mode == "mip":
        return volume.max(axis=0)

    if mode == "alpha":
        # Front-to-back "over" compositing. Accumulating transmittance as we go
        # lets us stop early once the ray is opaque.
        accumulated = np.zeros(volume.shape[1:], dtype=np.float32)
        transmittance = np.ones(volume.shape[1:], dtype=np.float32)
        for z in range(volume.shape[0]):
            alpha = np.clip(volume[z] * alpha_scale, 0.0, 1.0)
            accumulated += volume[z] * alpha * transmittance
            transmittance *= 1.0 - alpha
            if float(transmittance.max()) < 1e-3:
                break  # fully opaque: remaining slices cannot contribute
        return np.clip(accumulated, 0.0, 1.0)

    raise RenderError(f"Unknown render mode {mode!r}; use 'mip' or 'alpha'")


def render_rotation(
    volume: np.ndarray,
    cfg: RenderConfig | None = None,
    voxel_size_xy: float = 1.0,
    voxel_size_z: float = 1.0,
    resample_isotropic: bool = True,
    progress: object = None,
) -> list[np.ndarray]:
    """Render a volume as a sequence of rotated projections.

    Args:
        volume: 3D array ``(Z, Y, X)``. Any scale; normalised internally.
        cfg: Render parameters. ``None`` uses defaults.
        voxel_size_xy: Physical pixel size in XY.
        voxel_size_z: Physical z-step.
        resample_isotropic: Resample to isotropic voxels first. Disable only if
            the volume is already isotropic, or you accept distorted geometry.
        progress: Optional callable ``(index, total)`` for UI progress bars.

    Returns:
        List of uint8 RGB frames, each ``(H, W, 3)``.

    Raises:
        ValueError: If the volume is not 3D or has fewer than 2 z-planes.
        RenderError: On an unknown mode/colormap, or matplotlib missing.
    """
    cfg = cfg or RenderConfig()
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"render_rotation expects a 3D volume, got shape {vol.shape}")
    if vol.shape[0] < 2:
        raise ValueError(
            f"Volume has {vol.shape[0]} z-plane(s); a rotation needs at least 2. "
            "A single plane has no depth to rotate through."
        )

    if resample_isotropic:
        vol = make_isotropic(vol, voxel_size_xy, voxel_size_z)

    if cfg.downsample > 1:
        vol = ndi.zoom(vol, 1.0 / cfg.downsample, order=1).astype(np.float32)
        if min(vol.shape) < 2:
            raise ValueError(
                f"downsample={cfg.downsample} shrank the volume to {vol.shape}; "
                "use a smaller factor."
            )

    # Normalise to a high percentile rather than the true max. Fluorescence
    # volumes have a few very bright voxels; dividing by the max squashes all the
    # real signal into the dark end of the range, which is why an unadjusted MIP
    # looks dim. Clipping to `contrast_percentile` (default 100 = old behaviour)
    # lets the signal fill the range without one hot voxel dominating.
    lo = float(vol.min())
    hi = (float(np.percentile(vol, cfg.contrast_percentile))
          if cfg.contrast_percentile < 100 else float(vol.max()))
    if hi > lo:
        vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    else:
        vol = np.zeros_like(vol)

    # Gamma < 1 brightens midtones (dim neurites), > 1 darkens. Applied to the
    # normalised volume; since gamma is monotonic it commutes with the MIP max.
    if cfg.gamma != 1.0:
        vol = vol ** float(cfg.gamma)

    # "y" spins about the vertical screen axis (turntable) -> rotate in the Z-X
    # plane. "x" tumbles top-over-bottom -> rotate in the Z-Y plane.
    axes = (0, 2) if cfg.rotation_axis == "y" else (0, 1)
    vol = _pad_for_rotation(vol, axes)

    angles = np.linspace(0.0, cfg.total_degrees, cfg.n_frames, endpoint=False)
    frames: list[np.ndarray] = []
    for i, angle in enumerate(angles):
        rotated = (
            vol if abs(angle) < 1e-6
            else ndi.rotate(vol, float(angle), axes=axes, reshape=False, order=1, prefilter=False)
        )
        projection = _project(rotated, cfg.mode, cfg.alpha_scale)
        frames.append(_apply_colormap(projection, cfg.colormap))
        if callable(progress):
            progress(i + 1, len(angles))
    return frames


def save_gif(
    frames: list[np.ndarray], path: str | Path, fps: int = 15, loop: int = 0,
    optimize: bool = True,
) -> Path:
    """Write RGB frames to an optimised animated GIF.

    GIF supports only 256 colours per frame. Quantising each frame independently
    makes the palette drift between frames, which reads as flickering. This
    derives one adaptive palette from the brightest frame -- the one with the
    widest colour range -- and maps every frame onto it, giving a stable
    animation and a smaller file.

    Args:
        frames: uint8 RGB frames, all the same shape.
        path: Output ``.gif`` path.
        fps: Playback rate. GIF stores per-frame delay in 10 ms units, so the
            effective rate is quantised (e.g. 15 fps -> 66 ms -> ~15.2 fps).
        loop: Loop count; ``0`` loops forever.
        optimize: Let Pillow drop redundant pixels between frames.

    Returns:
        The written path.

    Raises:
        ValueError: If ``frames`` is empty or frames differ in shape.
        RenderError: If Pillow is unavailable.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RenderError("Writing GIFs needs Pillow: pip install pillow") from exc

    if not frames:
        raise ValueError("Cannot write a GIF with no frames")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"All frames must share a shape, got {shapes}")

    images = [Image.fromarray(np.asarray(f, dtype=np.uint8), mode="RGB") for f in frames]
    palette_source = max(images, key=lambda im: np.asarray(im).sum())
    master = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [im.quantize(palette=master, dither=Image.Dither.FLOYDSTEINBERG) for im in images]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        str(path),
        save_all=True,
        append_images=quantized[1:],
        duration=max(20, int(round(1000.0 / max(fps, 1)))),
        loop=int(loop),
        optimize=optimize,
        disposal=2,
    )
    return path


def render_rotation_gif(
    volume: np.ndarray,
    path: str | Path,
    cfg: RenderConfig | None = None,
    voxel_size_xy: float = 1.0,
    voxel_size_z: float = 1.0,
    progress: object = None,
) -> Path:
    """Render a rotating projection of a volume and write it as a GIF.

    Convenience wrapper over :func:`render_rotation` and :func:`save_gif`.

    Args:
        volume: 3D array ``(Z, Y, X)``.
        path: Output ``.gif`` path.
        cfg: Render parameters.
        voxel_size_xy: Physical pixel size in XY.
        voxel_size_z: Physical z-step.
        progress: Optional callable ``(index, total)``.

    Returns:
        The written path.

    Raises:
        ValueError: If the volume is unsuitable for rotation.
        RenderError: On rendering or encoding failure.
    """
    cfg = cfg or RenderConfig()
    frames = render_rotation(volume, cfg, voxel_size_xy, voxel_size_z, progress=progress)
    return save_gif(frames, path, fps=cfg.fps, loop=cfg.loop)
