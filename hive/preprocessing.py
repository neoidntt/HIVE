"""Channel selection and the two-image contract.

The central correctness fix in HIVE 2.0
---------------------------------------
The legacy pipeline built **one** image and used it for both jobs::

    raw -> select_channel -> gaussian -> CLAHE -> min-max normalise
                                  |
                       +----------+----------+
                       |                     |
                  segmentation      regionprops(intensity_image=...)

Because ``preprocess()`` min-max scaled every image to its *own* range, the
measured intensity of an object depended on the brightest pixel in its own
frame. Measured consequences:

===============================================  ========  ==========
Scenario                                         True      Reported
===============================================  ========  ==========
Two samples, 3x brightness difference            3.00x     **1.00x**
Z-stack with a 33% axial intensity gradient      1.333x    **1.000x**
===============================================  ========  ==========

Every plane of that z-stack reported an identical mean of 0.9355. Any
cross-image or cross-plane intensity comparison was therefore meaningless.

HIVE 2.0 splits the two paths::

    raw -> select_channel -> baseline subtract --> MEASUREMENT image
                                       |          (physical scale, quantitative)
                                       v
                        gaussian -> CLAHE -> normalise --> SEGMENTATION image
                                                           (detection only)

Objects are *found* on the segmentation image and *measured* on the measurement
image. Enhancement never touches a reported number.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from skimage import color, exposure

from .baseline import apply_baseline_subtraction
from .config import Channel, PreprocessConfig

_CHANNEL_INDEX: dict[str, int] = {"red": 0, "green": 1, "blue": 2}

# Warn above ~50 megapixels, where CLAHE's ~18x memory multiplier starts to
# dominate everything else in the pipeline (a 12671^2 plane costs ~13 GB).
_CLAHE_WARN_PIXELS: int = 50_000_000


@dataclass
class PreparedImage:
    """The output of :func:`prepare_image` -- both images plus provenance.

    Attributes:
        measurement: Background-corrected, single-channel, ORIGINAL intensity
            scale. This is the only array any measurement may read.
        segmentation: Contrast-enhanced, normalised to ``[0, 1]``. Used solely
            to decide which pixels are objects. Never measure this.
        channel_name: Which channel was actually used. Resolved from ``"auto"``
            so a report records the real choice rather than the word "auto".
        baseline: The subtracted background (scalar, array, or ``None``), for QC.
    """

    measurement: np.ndarray
    segmentation: np.ndarray
    channel_name: str
    baseline: np.ndarray | float | None


def ensure_2d(image: np.ndarray) -> np.ndarray:
    """Reduce an array to a single 2D plane.

    Args:
        image: 2D, or channel-last 3D, or an array with singleton axes.

    Returns:
        2D float32 array.

    Raises:
        ValueError: If the array cannot be reduced to 2D unambiguously.
    """
    arr = np.asarray(image)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            return color.rgb2gray(arr[:, :, :3]).astype(np.float32)
        return arr[:, :, 0].astype(np.float32)
    raise ValueError(f"Cannot reduce shape {np.asarray(image).shape} to a 2D image")


def select_channel(image: np.ndarray, channel: Channel = "auto") -> tuple[np.ndarray, str]:
    """Extract one analysis channel from a 2D or channel-last image.

    Channel semantics:
        ``red``/``green``/``blue``
            The corresponding RGB plane.
        ``magenta``
            Mean of the red and blue planes. This is a *display-space proxy*: a
            magenta fluorophore rendered in RGB excites both R and B, so their
            mean tracks it. It is NOT spectral unmixing. If you have the raw
            single-channel acquisition, analyse that instead.
        ``gray``
            ITU-R BT.601 luminance (``skimage.color.rgb2gray``).
        ``auto``
            The candidate with the highest standard deviation. A contrast
            heuristic, not a measurement -- the returned name records the choice.

    Args:
        image: 2D grayscale, or 3D channel-last ``(Y, X, C)``.
        channel: Channel to select.

    Returns:
        Tuple of (2D float32 channel image, resolved channel name).

    Raises:
        ValueError: On an unsupported shape or unknown channel name.
    """
    arr = np.asarray(image)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32), "gray"
    if arr.ndim != 3:
        raise ValueError(f"select_channel expects a 2D or channel-last 3D image, got {arr.shape}")
    if arr.shape[-1] < 3:
        return arr[:, :, 0].astype(np.float32), "channel_0"

    rgb = arr[:, :, :3].astype(np.float32)
    channel = channel.lower()  # type: ignore[assignment]

    if channel in _CHANNEL_INDEX:
        # ascontiguousarray, not a bare slice: a strided view would keep the whole
        # RGB temporary alive and cannot be safely written in place by callers.
        return np.ascontiguousarray(rgb[:, :, _CHANNEL_INDEX[channel]]), channel
    if channel in ("magenta", "pink"):
        return ((rgb[:, :, 0] + rgb[:, :, 2]) / 2.0).astype(np.float32), "magenta"
    if channel == "gray":
        return color.rgb2gray(rgb).astype(np.float32), "gray"
    if channel == "auto":
        candidates: dict[str, np.ndarray] = {
            "red": rgb[:, :, 0],
            "green": rgb[:, :, 1],
            "blue": rgb[:, :, 2],
            "magenta": (rgb[:, :, 0] + rgb[:, :, 2]) / 2.0,
            "gray": color.rgb2gray(rgb),
        }
        best = max(candidates, key=lambda k: float(np.std(candidates[k])))
        return candidates[best].astype(np.float32), best
    raise ValueError(
        f"Unknown channel {channel!r}; use auto, red, green, blue, magenta or gray"
    )


def select_channel_volume(stack: np.ndarray, channel: Channel = "auto") -> tuple[np.ndarray, str]:
    """Reduce a z-stack to a single-channel volume ``(Z, Y, X)``.

    This is the function whose absence broke the legacy 3D pipeline. The old
    code applied baseline subtraction to the raw 4D ``(Z, Y, X, C)`` array
    *before* reducing the channel axis, and baseline subtraction only accepts
    2D/3D -- so every RGB z-stack died with ``ValueError: Expected a 2D image
    or 3D z-stack, got shape (Z, Y, X, 3)``.

    The channel is resolved once from the stack's maximum-intensity projection
    rather than per plane, so ``"auto"`` cannot pick different channels for
    different planes of the same stack.

    Args:
        stack: ``(Z, Y, X)`` grayscale or ``(Z, Y, X, C)`` channel-last.
        channel: Channel to select.

    Returns:
        Tuple of (``(Z, Y, X)`` float32 volume, resolved channel name).

    Raises:
        ValueError: On an unsupported stack shape.
    """
    arr = np.asarray(stack)
    if arr.ndim == 2:
        return arr[None, ...].astype(np.float32), "gray"
    if arr.ndim == 3:
        return arr.astype(np.float32), "gray"
    if arr.ndim != 4:
        raise ValueError(
            f"Unsupported z-stack shape {arr.shape}; expected (Z,Y,X) or (Z,Y,X,C)"
        )
    if arr.shape[-1] not in (2, 3, 4):
        raise ValueError(
            f"Z-stack has shape {arr.shape}; the last axis should be channels (size 2-4). "
            "If this is (C,Z,Y,X), transpose it with np.moveaxis(stack, 0, -1) first."
        )

    # Resolve "auto" once, on the projection, so the choice is stack-wide.
    _, resolved = select_channel(arr.max(axis=0), channel)
    out = np.empty(arr.shape[:3], dtype=np.float32)
    for z in range(arr.shape[0]):
        out[z], _ = select_channel(arr[z], resolved)  # type: ignore[arg-type]
    return out, resolved


def build_segmentation_image(
    measurement: np.ndarray, cfg: PreprocessConfig, per_plane_clahe: bool = True
) -> np.ndarray:
    """Build the contrast-enhanced image used ONLY for object detection.

    Pipeline: Gaussian smoothing (suppress shot noise so Otsu sees a cleaner
    bimodal histogram) -> global min-max normalisation -> optional CLAHE
    (Zuiderveld, 1994) to recover dim objects under uneven illumination.

    Normalisation here is **global** across the whole array, not per plane. A
    per-plane min-max destroys axial structure; measured on a stack with a real
    33% z-gradient, the legacy per-plane version returned an identical mean at
    every plane. It does not corrupt measurements now (nothing measures this
    image), but a flattened segmentation image still biases a global threshold.

    Args:
        measurement: Background-corrected single-channel image or volume.
        cfg: Preprocessing parameters.
        per_plane_clahe: For 3D input, equalise each plane separately. True is
            much faster than 3D CLAHE and usually sufficient.

    Returns:
        Enhanced array in ``[0, 1]``, float32, same shape as input.
    """
    from .hardware import gaussian_filter_auto

    if cfg.gaussian_sigma > 0:
        # Do not smooth across z: planes are typically far more widely spaced
        # than xy pixels, so an isotropic sigma over-blurs the axial direction.
        sigma: float | tuple[float, ...] = (
            (0.0, cfg.gaussian_sigma, cfg.gaussian_sigma)
            if np.asarray(measurement).ndim == 3 else cfg.gaussian_sigma
        )
        img = gaussian_filter_auto(measurement, sigma)  # returns a fresh array we own
    else:
        # MUST copy: with sigma=0 an asarray() would alias the measurement image,
        # and the in-place rescale below would silently corrupt the one array that
        # every reported number is measured from.
        img = np.array(measurement, dtype=np.float32)

    # In-place rescale. `(img - lo) / (hi - lo)` costs two extra full-size
    # temporaries; on a large volume that is the difference between running and
    # swapping.
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        np.subtract(img, lo, out=img)
        np.divide(img, hi - lo, out=img)
    else:
        img.fill(0.0)

    if cfg.use_clahe:
        img = _apply_clahe(img, cfg.clahe_clip_limit, per_plane_clahe)

    np.clip(img, 0.0, 1.0, out=img)
    return img


def _apply_clahe(img: np.ndarray, clip_limit: float, per_plane: bool) -> np.ndarray:
    """Apply CLAHE, guarding against degenerate flat inputs.

    ``equalize_adapthist`` raises or returns NaN on a constant array, which
    happens routinely for empty channels and blank planes.

    Args:
        img: Array already scaled to ``[0, 1]``.
        clip_limit: CLAHE clip limit.
        per_plane: Equalise each z-plane independently for 3D input.

    Returns:
        Equalised float32 array.
    """
    if float(img.max()) <= float(img.min()):
        return img  # flat image: nothing to equalise

    # scikit-image's equalize_adapthist peaks at ~17-19x the image size, measured,
    # and that is independent of kernel_size -- it is inherent to the implementation.
    # On a 12671x12671 plane that is 13 GB for one image. Warn rather than let a
    # large batch die halfway through, or silently run 5x slower than it needs to.
    if img.ndim == 2 and img.size > _CLAHE_WARN_PIXELS:
        warnings.warn(
            f"CLAHE on a {img.shape[0]}x{img.shape[1]} image will peak near "
            f"{img.nbytes * 18 / 1e9:.1f} GB (~18x the image; measured, and not reducible "
            "via kernel_size). Top-hat background subtraction has already removed uneven "
            "illumination, so CLAHE's marginal benefit here is often small -- on a "
            "high-contrast test plane it changed nothing but cost 2.7x the memory and "
            "1.5x the time. Compare object counts with use_clahe=False on YOUR data: if "
            "they agree, turn it off and fit ~3x more parallel workers.",
            stacklevel=3,
        )

    if img.ndim == 3 and per_plane:
        # Write each equalised plane straight back into the buffer we own.
        # equalize_adapthist returns float64, but only one PLANE of it exists at a
        # time here -- allocating a whole float64 volume would cost 2x the volume.
        for z in range(img.shape[0]):
            plane = img[z]
            if plane.max() > plane.min():
                img[z] = exposure.equalize_adapthist(plane, clip_limit=clip_limit)
        return img
    img[...] = exposure.equalize_adapthist(img, clip_limit=clip_limit)
    return img


def prepare_image(
    raw: np.ndarray, cfg: PreprocessConfig, channel: Channel = "auto",
) -> PreparedImage:
    """Prepare a 2D image for analysis, producing both images.

    Args:
        raw: Raw 2D or channel-last image, in its original intensity scale.
        cfg: Preprocessing parameters.
        channel: Channel to analyse.

    Returns:
        A :class:`PreparedImage`.

    Raises:
        ValueError: If the image shape is unsupported or contains NaN/Inf.
    """
    channel_img, channel_name = select_channel(raw, channel)
    # copy=False: select_channel already produced an array we own, so baseline
    # subtraction may work in place instead of making another full-size copy.
    measurement, baseline = apply_baseline_subtraction(
        channel_img,
        method=cfg.baseline_method,
        percentile=cfg.baseline_percentile,
        sigma=cfg.baseline_sigma,
        tophat_radius=cfg.tophat_radius,
        normalize=False,  # never rescale the measurement image
        return_baseline=cfg.keep_baseline,
        copy=False,
    )
    segmentation = build_segmentation_image(measurement, cfg)
    return PreparedImage(measurement, segmentation, channel_name, baseline)


def prepare_volume(
    raw: np.ndarray, cfg: PreprocessConfig, channel: Channel = "auto",
) -> PreparedImage:
    """Prepare a z-stack for analysis, producing both volumes.

    Order matters and is the fix for the legacy 3D crash: channel selection
    happens FIRST, reducing ``(Z, Y, X, C)`` to ``(Z, Y, X)``, and only then is
    baseline subtraction applied to a shape it accepts.

    Args:
        raw: Raw ``(Z, Y, X)`` or ``(Z, Y, X, C)`` stack.
        cfg: Preprocessing parameters. For 3D, prefer per-axis ``tophat_radius``
            (e.g. ``(2, 25, 25)``) since z-spacing rarely matches xy.
        channel: Channel to analyse.

    Returns:
        A :class:`PreparedImage` whose arrays are 3D volumes.

    Raises:
        ValueError: If the stack shape is unsupported or contains NaN/Inf.
    """
    volume, channel_name = select_channel_volume(raw, channel)
    # select_channel_volume always returns a freshly allocated array, so we can
    # hand ownership to the baseline step and let it work in place.
    measurement, baseline = apply_baseline_subtraction(
        volume,
        method=cfg.baseline_method,
        percentile=cfg.baseline_percentile,
        sigma=cfg.baseline_sigma,
        tophat_radius=cfg.tophat_radius,
        normalize=False,
        return_baseline=cfg.keep_baseline,
        copy=False,
    )
    segmentation = build_segmentation_image(measurement, cfg)
    return PreparedImage(measurement, segmentation, channel_name, baseline)
