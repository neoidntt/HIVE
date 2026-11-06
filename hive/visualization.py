"""Overlays, Sholl plots, and the interactive 3D mesh viewer.

Everything here is for looking at, not measuring. Display images are freely
contrast-stretched; quantitative values come from the CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import color

from .arborization import ShollResult
from .cleanup import normalize_for_display


def to_rgb_display(image: np.ndarray) -> np.ndarray:
    """Convert any image to a displayable RGB float array in ``[0, 1]``.

    Args:
        image: 2D grayscale or channel-last RGB(A).

    Returns:
        ``(H, W, 3)`` float32 array in ``[0, 1]``.

    Raises:
        ValueError: If the array cannot be interpreted as an image.
    """
    arr = np.asarray(image)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim == 2:
        gray = normalize_for_display(arr)
        return np.dstack([gray, gray, gray]).astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[:, :, :3].astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / rgb.max()
        return np.clip(rgb, 0, 1).astype(np.float32, copy=False)
    if arr.ndim == 3:
        return to_rgb_display(arr[:, :, 0])
    raise ValueError(f"Cannot display an array of shape {np.asarray(image).shape}")


def create_boundary_overlay(
    image: np.ndarray, labels: np.ndarray, color_rgb: tuple[float, float, float] = (1, 1, 0)
) -> np.ndarray:
    """Draw segmentation boundaries over an image.

    The most useful QC view: it shows where the segmentation thinks each object
    ends while leaving the underlying signal visible, so over- and
    under-segmentation are both obvious.

    Args:
        image: Background image (raw or measurement).
        labels: Integer label array.
        color_rgb: Boundary colour.

    Returns:
        ``(H, W, 3)`` float array in ``[0, 1]``.
    """
    # What skimage's mark_boundaries does, minus its extra full-size copy of the picture:
    # find the thick boundaries and paint them into the RGB float32 image made here.
    marked = to_rgb_display(image)
    marked[_thick_boundaries(np.asarray(labels))] = color_rgb
    return marked


def _thick_boundaries(labels: np.ndarray) -> np.ndarray:
    """``skimage.segmentation.find_boundaries(labels, mode="thick")``, computed with shifts.

    A pixel is on a boundary when the largest and the smallest label among itself and its
    edge neighbours differ (grey dilation != grey erosion with a cross). scikit-image
    reflects at the image border, which adds no new value, so neighbours past the border
    are simply left out.
    """
    largest, smallest = labels.copy(), labels.copy()
    for axis in range(labels.ndim):
        toward = [slice(None)] * labels.ndim
        away = [slice(None)] * labels.ndim
        toward[axis], away[axis] = slice(1, None), slice(None, -1)
        toward, away = tuple(toward), tuple(away)
        np.maximum(largest[toward], labels[away], out=largest[toward])
        np.maximum(largest[away], labels[toward], out=largest[away])
        np.minimum(smallest[toward], labels[away], out=smallest[toward])
        np.minimum(smallest[away], labels[toward], out=smallest[away])
    return largest != smallest


def create_label_overlay(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint each object a distinct colour over the image.

    Args:
        image: Background image.
        labels: Integer label array.
        alpha: Label opacity.

    Returns:
        ``(H, W, 3)`` float array in ``[0, 1]``.
    """
    return color.label2rgb(
        labels, image=normalize_for_display(image), bg_label=0, alpha=alpha
    )


def label_overlay_f32(
    image: np.ndarray, labels: np.ndarray, alpha: float = 0.45, band_pixels: int = 4_000_000,
) -> np.ndarray:
    """:func:`create_label_overlay` as float32, built in bands of rows to save memory.

    ``color.label2rgb`` makes several float64 RGB arrays at once, about 5 GB for a
    66-megapixel image. This gives the same pixel values (cast to float32, which is the first
    thing the PNG writer does anyway) with only the float32 result and one band of
    temporaries. Each label gets the colour ``label2rgb`` would give it: the colours are
    looked up once from ``label2rgb`` itself on the list of distinct labels, so the cycle
    matches whatever labels are present.

    Args:
        image: Background image.
        labels: Integer label array.
        alpha: Label opacity.

    Returns:
        ``(H, W, 3)`` float32 array in ``[0, 1]``.
    """
    labels = np.asarray(labels)
    gray = normalize_for_display(image)
    distinct = np.unique(labels)
    palette = color.label2rgb(distinct.reshape(-1, 1), bg_label=0)[:, 0, :]
    # Label -> row of the palette. A table lookup gives the same row as a binary search, much
    # faster, as long as the labels are small enough to index (they are: labels are counted
    # from 1); otherwise the binary search is used.
    table = None
    if labels.size and int(distinct[-1]) < 64 * labels.size + 1_000_000 and distinct[0] >= 0:
        table = np.zeros(int(distinct[-1]) + 1, dtype=np.int32)
        table[distinct] = np.arange(len(distinct), dtype=np.int32)
    out = np.empty(labels.shape + (3,), dtype=np.float32)
    rows = max(1, band_pixels // max(1, labels.shape[1]))
    for start in range(0, labels.shape[0], rows):
        band = slice(start, start + rows)
        lab = labels[band]
        # label2rgb's default background colour is black, so background pixels come out as
        # the image dimmed by (1 - alpha): the palette row for label 0 is black, and adding
        # its zero changes nothing.
        dimmed = gray[band][..., None] * (1 - alpha)
        out[band] = dimmed
        painted = lab > 0
        if painted.any():
            index = table[lab[painted]] if table is not None else np.searchsorted(distinct, lab[painted])
            base = np.repeat(gray[band][painted][:, None], 3, axis=1)
            out[band][painted] = palette[index] * alpha + base * (1 - alpha)
    return out


def create_skeleton_overlay(
    image: np.ndarray, skeleton: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Overlay a skeleton (and optionally its mask) on an image.

    Shows the mask in dim blue and the skeleton in bright red, which makes it
    obvious when the skeleton has wandered off the structure or sprouted spurs.

    Args:
        image: Background image.
        skeleton: Boolean skeleton.
        mask: Optional mask the skeleton was derived from.

    Returns:
        ``(H, W, 3)`` float array in ``[0, 1]``.
    """
    rgb = to_rgb_display(image)
    if mask is not None:
        rgb[np.asarray(mask, dtype=bool)] = rgb[np.asarray(mask, dtype=bool)] * 0.6 + np.array(
            [0.0, 0.0, 0.4], dtype=np.float32
        )
    rgb[np.asarray(skeleton, dtype=bool)] = [1.0, 0.1, 0.1]
    return np.clip(rgb, 0, 1, out=rgb)


def plot_sholl(sholl: ShollResult, unit: str = "px", title: str | None = None):
    """Plot a Sholl profile with its critical radius marked.

    Args:
        sholl: A :class:`~hive.arborization.ShollResult`.
        unit: Unit label for the x-axis.
        title: Plot title.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ImportError: If matplotlib is unavailable.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sholl.radii, sholl.intersections, marker="o", linewidth=1.5, markersize=3)
    if sholl.max_intersections > 0:
        ax.axvline(sholl.critical_radius, linestyle="--", alpha=0.6, color="crimson",
                   label=f"critical radius = {sholl.critical_radius:.1f} {unit}")
        ax.legend(frameon=False)
    ax.set_xlabel(f"Distance from centre ({unit})")
    ax.set_ylabel("Intersections N(r)")
    ax.set_title(title or "Sholl analysis")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def create_3d_mesh_html(
    path: str | Path,
    mask: np.ndarray,
    voxel_size_xy: float = 1.0,
    voxel_size_z: float = 1.0,
    step_size: int = 2,
    title: str = "HIVE 3D Reconstruction",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Write an interactive Plotly mesh viewer for a segmented volume.

    The surface is extracted with marching cubes (Lorensen & Cline, 1987) at the
    0.5 iso-level -- the midpoint between background and foreground of a binary
    mask, i.e. the mask boundary. ``spacing`` is passed so the mesh is in
    physical units and anisotropic stacks are not rendered squashed.

    Args:
        path: Output ``.html`` path.
        mask: Boolean 3D mask.
        voxel_size_xy: Physical pixel size in XY.
        voxel_size_z: Physical z-step.
        step_size: Marching-cubes stride; higher is coarser and faster.
        title: Figure title.

    Returns:
        Tuple of (vertices, faces), or ``(None, None)`` if the mask was empty --
        in which case a placeholder page is still written, so a downstream link
        never 404s.

    Raises:
        ImportError: If Plotly is unavailable.
        ValueError: If the mask is not 3D.
    """
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"create_3d_mesh_html expects a 3D mask, got shape {mask.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not mask.any():
        fig = go.Figure()
        fig.update_layout(title=f"{title}: no segmented volume detected")
        fig.write_html(str(path), include_plotlyjs="cdn")
        return None, None

    spacing = (float(voxel_size_z), float(voxel_size_xy), float(voxel_size_xy))
    verts, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=spacing, step_size=max(1, int(step_size))
    )

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.55,
            intensity=verts[:, 0],
            colorscale="Plasma",
            name="Segmented volume",
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")
    return verts, faces


def export_obj(path: str | Path, verts: np.ndarray, faces: np.ndarray) -> Path:
    """Write a triangular mesh as a Wavefront OBJ file.

    OBJ indices are 1-based, unlike NumPy's 0-based faces, hence the ``+ 1``.
    Vertices are written x, y, z while marching cubes returns them z, y, x.

    Args:
        path: Output ``.obj`` path.
        verts: ``(N, 3)`` vertex array in (z, y, x) order.
        faces: ``(M, 3)`` triangle index array.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# HIVE mesh export\n")
        for v in verts:
            f.write(f"v {v[2]:.6f} {v[1]:.6f} {v[0]:.6f}\n")
        for face in faces:
            a, b, c = np.asarray(face) + 1
            f.write(f"f {a} {b} {c}\n")
    return path
