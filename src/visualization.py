"""LEGACY (HIVE 1.x) QC overlay rendering.

.. deprecated:: 2.0
    Superseded by :mod:`hive.visualization`.

Everything here is contrast-stretched per image for visibility. These outputs
are QC previews for confirming that segmentation found the right objects --
never read quantitative values off them. Use the exported CSV.
"""

import numpy as np
from skimage import color, segmentation, exposure
from .preprocessing import ensure_2d


def to_rgb_display(image):
    """Coerce any image to a displayable RGB array in [0, 1].

    Args:
        image: 2D (replicated across R, G and B), RGB(A) (first three
            channels kept), or anything :func:`ensure_2d` can flatten.

    Returns:
        ``(H, W, 3)`` float array in [0, 1], stretched per image.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        img = ensure_2d(arr)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        return np.dstack([img, img, img])
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[:, :, :3].astype(np.float32)
        if rgb.max() > 1:
            rgb = rgb / rgb.max()
        return np.clip(rgb, 0, 1)
    return to_rgb_display(ensure_2d(arr))


def create_boundary_overlay(original, labeled, color_rgb=(1, 1, 0)):
    """Outline each detected object on the original image.

    Args:
        original: Image to draw on.
        labeled: Integer label image.
        color_rgb: Outline colour; defaults to yellow for visibility against
            the magenta/red fluorescence this pipeline usually handles.

    Returns:
        RGB overlay array. ``mode="thick"`` draws boundaries on both sides of
        the edge, so outlines stay visible when downscaled for a report.
    """
    rgb = to_rgb_display(original)
    return segmentation.mark_boundaries(rgb, labeled, color=color_rgb, mode="thick")


def create_label_overlay(intensity_image, labeled, alpha=0.45):
    """Tint each object a distinct colour over a greyscale base.

    Args:
        intensity_image: Greyscale base image.
        labeled: Integer label image.
        alpha: Colour opacity. The default keeps the underlying texture
            visible so under- and over-segmentation stay recognisable.

    Returns:
        RGB overlay array. Useful for spotting merged objects, which appear as
        one colour spanning two visually separate cells.
    """
    base = ensure_2d(intensity_image)
    if base.max() > base.min():
        base = (base - base.min()) / (base.max() - base.min())
    return color.label2rgb(labeled, image=base, bg_label=0, alpha=alpha)


def create_magenta_overlay(intensity_image, mask):
    """Render the image in magenta with masked pixels tinted toward white.

    Reproduces the usual magenta fluorescence look by writing the greyscale
    image into the red and blue channels, then lifting green to 0.25 inside
    the mask so detected pixels shift perceptibly without hiding the signal.

    Args:
        intensity_image: Greyscale image.
        mask: Boolean mask of detected pixels.

    Returns:
        ``(H, W, 3)`` RGB array in [0, 1].
    """
    img = ensure_2d(intensity_image)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    rgb = np.zeros((*img.shape, 3), dtype=np.float32)
    rgb[..., 0] = img
    rgb[..., 2] = img
    rgb[mask.astype(bool), 1] = 0.25
    return np.clip(rgb, 0, 1)
