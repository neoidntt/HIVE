"""Preview what each background-subtraction method does to an image.

Choosing a method by looking at the finished analysis is slow and easy to get wrong.
This applies all four estimators (:mod:`hive.baseline`) to a crop of one image with
the run's own settings, so the effect can be judged before the full analysis.

Read the two numbers beside each preview: *background removed* is the share of the
image's mean intensity taken away, and *pixels at zero* is the share of the corrected
image that is exactly zero. A high second number means the method is eating real
signal (for top-hat, the radius is smaller than the objects).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline import apply_baseline_subtraction
from .config import PreprocessConfig

METHODS: tuple[str, ...] = ("none", "percentile", "smooth", "tophat")


@dataclass
class BaselinePreview:
    """The result of one method on the preview crop.

    Attributes:
        method: Method name.
        background: The estimated background (an image, or a single number for
            ``percentile``, or ``None`` for ``none``).
        corrected: The image after subtraction, original intensity scale.
        removed_fraction: Mean background divided by mean original intensity.
        zero_fraction: Share of corrected pixels that are exactly zero.
    """

    method: str
    background: np.ndarray | float | None
    corrected: np.ndarray
    removed_fraction: float
    zero_fraction: float


def _xy(value):
    """The in-plane part of a scalar or per-axis (z, y, x) size."""
    return value[-1] if isinstance(value, (tuple, list)) else value


def center_crop(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Crop the middle of a 2D image so previews stay fast on huge images."""
    h, w = image.shape
    y0, x0 = max(0, (h - max_side) // 2), max(0, (w - max_side) // 2)
    return image[y0:y0 + max_side, x0:x0 + max_side]


def preview_methods(
    image: np.ndarray, cfg: PreprocessConfig, max_side: int = 512,
) -> list[BaselinePreview]:
    """Apply every background method to a crop of ``image``.

    The image itself is not modified.

    Args:
        image: One 2D plane, original intensity scale.
        cfg: Preprocessing settings; the percentile, sigma and radius are used as set.
        max_side: Longest edge of the central crop that is processed.

    Returns:
        One :class:`BaselinePreview` per method in :data:`METHODS`.

    Raises:
        ValueError: If ``image`` is not 2D.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"preview_methods expects one 2D plane, got shape {arr.shape}")
    crop = center_crop(arr, max_side)
    mean_original = float(crop.mean()) or 1.0
    previews = []
    for method in METHODS:
        corrected, background = apply_baseline_subtraction(
            crop, method=method, percentile=cfg.baseline_percentile,
            sigma=_xy(cfg.baseline_sigma), tophat_radius=_xy(cfg.tophat_radius),
            normalize=False, return_baseline=True, copy=True,
        )
        if background is None:
            removed = 0.0
        else:
            removed = float(np.mean(background)) / mean_original
        previews.append(BaselinePreview(
            method, background, corrected, removed, float(np.mean(corrected <= 0)),
        ))
    return previews


def preview_figure(original: np.ndarray, previews: list[BaselinePreview]):
    """Draw the previews: for each method, the original, its background and the result.

    All panels share one grey scale (the original's 1st to 99.5th percentile), so
    brightness can be compared between methods.

    Args:
        original: The cropped original plane (use :func:`center_crop`).
        previews: Output of :func:`preview_methods`.

    Returns:
        A matplotlib ``Figure`` (not attached to pyplot).
    """
    from matplotlib.figure import Figure

    lo, hi = np.percentile(original, [1, 99.5])
    fig = Figure(figsize=(7.5, 2.4 * len(previews)), layout="constrained")
    axes = fig.subplots(len(previews), 3)
    for row, preview in zip(axes, previews):
        background = preview.background
        if background is None:
            background = np.zeros_like(original)
        elif np.isscalar(background):
            background = np.full_like(original, float(background))
        panels = (
            (original, f"{preview.method}"),
            (background, f"background (removes {preview.removed_fraction:.0%})"),
            (preview.corrected, f"corrected ({preview.zero_fraction:.0%} of pixels at zero)"),
        )
        for ax, (image, title) in zip(row, panels):
            ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=8, fontweight="bold" if image is original else "normal")
            ax.axis("off")
    return fig
