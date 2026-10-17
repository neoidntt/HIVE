"""LEGACY (HIVE 1.x) 2D segmentation.

.. deprecated:: 2.0
    Superseded by :mod:`hive.segmentation`. Behavioural differences that make
    1.x and 2.0 object counts non-comparable are noted per function below.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation


def threshold_segmentation(image_2d, min_size=80, hole_size=50, threshold_scale=1.2):
    """Segment a 2D image by scaled Otsu thresholding plus mask cleanup.

    Args:
        image_2d: 2D array. Higher values are assumed to be signal.
        min_size: Objects below this area (px) are discarded as debris.
        hole_size: Interior holes below this area (px) are filled.
        threshold_scale: Multiplier applied to the Otsu threshold. ``1.0`` is
            unmodified Otsu (1979); this function defaults to ``1.2``.

    Returns:
        Boolean mask. An image with no intensity range returns all-``False``
        rather than raising, so a blank slice cannot abort a batch run.

    Note:
        ``threshold_scale`` is an empirical knob with no basis in Otsu's
        method. If object count varies steeply with it, the count reflects the
        threshold rather than the sample -- always report the value used.

        The final opening/closing with a radius-1 disk is unconditional here.
        It removes single-pixel noise but also erodes genuinely thin
        structures such as neurites, which is why 2.0 made cleanup
        configurable instead of mandatory.
    """
    img = np.asarray(image_2d, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"threshold_segmentation expects 2D image, got {img.shape}")
    # Flat image: Otsu is undefined and every pixel would land on one side.
    if float(img.max()) <= float(img.min()):
        return np.zeros_like(img, dtype=bool)
    try:
        thresh = filters.threshold_otsu(img) * float(threshold_scale)
    except ValueError:
        # Otsu needs a usable histogram; fall back to a mean + 1 SD cut.
        thresh = float(img.mean()) + float(img.std())
    mask = img > thresh
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_size))
    mask = morphology.remove_small_holes(mask, area_threshold=int(hole_size))
    # Opening then closing: despeckle, then bridge 1-px gaps. Thin processes
    # can be thinned or severed by this pair.
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.binary_closing(mask, morphology.disk(1))
    return mask.astype(bool)


def watershed_split(mask, image_2d):
    """Split touching objects via distance-transform watershed.

    Implements the classic marker-based approach (Beucher & Lantuejoul, 1979):
    seed at local maxima of the Euclidean distance transform, then flood the
    negated distance so basins meet along the constrictions between objects.

    Args:
        mask: Boolean mask to split.
        image_2d: Unused; accepted for call-signature compatibility. The split
            is driven purely by mask geometry, not by intensity.

    Returns:
        Boolean mask with watershed lines removed between split objects.

    Warning:
        Every local maximum becomes a marker, with no smoothing or minimum
        separation, so a lumpy distance transform over-segments badly. The
        method also assumes roughly convex objects -- on a branched cell it
        shatters the arbor into fragments. Leave it off for neurons.

        Returning ``labels > 0`` discards the split labels themselves, so the
        caller must relabel; adjacent objects separated only by a watershed
        line may then merge again in :func:`label_mask`.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    distance = ndi.distance_transform_edt(mask)
    coords = morphology.local_maxima(distance)
    markers = measure.label(coords)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    return labels > 0


def label_mask(mask):
    """Assign a unique integer label to each connected component.

    Args:
        mask: Boolean mask.

    Returns:
        Integer label image; ``0`` is background.
    """
    return measure.label(mask.astype(bool))
