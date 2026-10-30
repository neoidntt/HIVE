"""Object detection: thresholding, watershed splitting, and Cellpose.

All functions here consume the **segmentation image** (contrast-enhanced,
``[0, 1]``) and return masks or label arrays. None of them read intensity for
measurement purposes -- see :mod:`hive.preprocessing` for why that separation
matters.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation as skseg
from skimage.feature import peak_local_max

from .config import SegmentationConfig


class SegmentationError(RuntimeError):
    """Raised when segmentation cannot proceed (e.g. an empty channel)."""


def otsu_threshold(image: np.ndarray, scale: float = 1.0) -> float:
    """Compute a (scaled) Otsu threshold.

    Otsu (1979) selects the threshold minimising intra-class variance of a
    bimodal intensity histogram.

    The ``scale`` multiplier is an **empirical knob, not part of Otsu's method**.
    The legacy default of ``1.2`` silently made every run 20% more conservative
    than the published algorithm, which is defensible as a practical adjustment
    for the long bright tail of fluorescence histograms but must be reported.
    HIVE 2.0 defaults to ``1.0`` (true Otsu) and records the value used.

    Args:
        image: Input image or volume.
        scale: Multiplier on the Otsu value. ``1.0`` is unmodified Otsu.

    Returns:
        The threshold value.

    Raises:
        SegmentationError: If the image is flat (max == min), which usually
            means an empty channel rather than a genuinely uniform sample.
    """
    img = np.asarray(image, dtype=np.float32)
    if float(img.max()) <= float(img.min()):
        raise SegmentationError(
            f"Image is uniform (all values = {float(img.min()):.4g}); no threshold exists. "
            "This usually means an empty channel -- check the channel selection."
        )
    return float(filters.threshold_otsu(img) * float(scale))


def threshold_segmentation(image: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Segment by (scaled) Otsu thresholding, then clean the mask.

    Works on both 2D images and 3D volumes; the structuring element for the
    morphological polish adapts to dimensionality.

    Args:
        image: Segmentation image (2D or 3D), already enhanced.
        cfg: Segmentation parameters.

    Returns:
        Boolean mask, same shape as ``image``. An all-False mask is a valid
        result meaning "nothing passed threshold".

    Raises:
        SegmentationError: If the image is uniform.
        ValueError: If the image is not 2D or 3D.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim not in (2, 3):
        raise ValueError(f"threshold_segmentation expects a 2D or 3D array, got {img.shape}")

    try:
        threshold = otsu_threshold(img, cfg.threshold_scale)
    except SegmentationError as exc:
        # A blank field is a legitimate result ("no objects here"), not a failure.
        # Raising would drop the image from the batch summary entirely, turning a
        # recorded zero into a missing row -- and a zero count is data. Warn so a
        # genuinely wrong channel (every image blank) is still visible.
        warnings.warn(f"{exc} Reporting 0 objects for this image.", stacklevel=2)
        return np.zeros(img.shape, dtype=bool)

    mask = img > threshold
    if not mask.any():
        return mask

    # A radius-1 opening then closing removes single-pixel noise and bridges
    # 1-px gaps without materially altering object size. `opening`/`closing` are
    # used rather than `binary_opening`/`binary_closing`, which are deprecated
    # and removed in scikit-image 0.28; on a boolean array they are equivalent.
    footprint = morphology.disk(1) if img.ndim == 2 else morphology.ball(1)
    mask = morphology.opening(mask, footprint)
    mask = morphology.closing(mask, footprint)
    return mask.astype(bool)


def watershed_split(mask: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """Split touching convex objects via distance-transform watershed.

    Method (Beucher & Lantuejoul, 1979): the Euclidean distance transform peaks
    at each object's centre; treating those peaks as markers and flooding the
    inverted distance map places boundaries at the necks between objects.

    Two bugs in the legacy implementation are fixed here:

    1. It ended ``return labels > 0``, discarding the split and returning the
       input mask unchanged. Verified: two touching discs produced one label
       before *and* after, with output ``array_equal`` to the input. The
       function was a no-op (and was never called).
    2. It used ``morphology.local_maxima``, which flags every pixel of a
       maximal plateau. A flat-topped distance ridge yields dozens of markers
       and shatters one cell into dozens of fragments. ``peak_local_max`` with
       a ``min_distance`` guard yields one marker per object.

    This assumes roughly convex, similarly sized objects. It will over-split
    genuinely branched structures such as neurons -- leave it off for
    arborisation work.

    Args:
        mask: Boolean mask.
        min_distance: Minimum separation (px) between markers. Roughly the
            smallest expected object radius.

    Returns:
        Integer label array (0 = background), NOT a boolean mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(
        distance, labels=mask, min_distance=int(min_distance), exclude_border=False
    )
    if len(coords) == 0:
        return measure.label(mask).astype(np.int32)

    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return skseg.watershed(-distance, markers, mask=mask).astype(np.int32)


def label_mask(mask: np.ndarray, connectivity: int | None = None) -> np.ndarray:
    """Label connected components of a boolean mask.

    Args:
        mask: Boolean mask.
        connectivity: Passed to :func:`skimage.measure.label`. ``None`` uses
            full connectivity (8 in 2D, 26 in 3D).

    Returns:
        Integer label array.
    """
    return measure.label(np.asarray(mask, dtype=bool), connectivity=connectivity).astype(np.int32)


def segment_with_cellpose(
    image: np.ndarray, cfg: SegmentationConfig, gpu: bool = False
) -> np.ndarray:
    """Segment a 2D image with Cellpose.

    Cellpose (Stringer et al., 2021, *Nat Methods*) predicts spatial flow fields
    and follows them to cell centres, so it handles touching cells that
    thresholding merges. It is trained on cell-like objects and tends to
    underperform on connected tissue such as whole-brain neuropil, where a
    threshold is often better -- as the project README already notes.

    Args:
        image: 2D segmentation image.
        cfg: Segmentation parameters (``cellpose_*`` fields).
        gpu: Request GPU inference.

    Returns:
        Integer label array.

    Raises:
        ImportError: If Cellpose is not installed.
        ValueError: If the image is not 2D.
        SegmentationError: If Cellpose fails at runtime.
    """
    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError(
            "Cellpose is not installed. Install with: pip install cellpose torch torchvision"
        ) from exc

    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"segment_with_cellpose expects a 2D image, got shape {img.shape}")

    diameter = None if cfg.cellpose_diameter <= 0 else float(cfg.cellpose_diameter)
    try:
        try:
            model = models.CellposeModel(gpu=bool(gpu), model_type=cfg.cellpose_model)
        except (TypeError, AttributeError):
            # Cellpose >=4 renamed/removed several constructor arguments.
            model = models.Cellpose(gpu=bool(gpu), model_type=cfg.cellpose_model)
        result = model.eval(
            img,
            channels=[0, 0],
            diameter=diameter,
            cellprob_threshold=float(cfg.cellpose_cellprob),
            flow_threshold=float(cfg.cellpose_flow),
        )
    except Exception as exc:
        raise SegmentationError(f"Cellpose failed: {exc}") from exc
    return np.asarray(result[0], dtype=np.int32)


def segment(image: np.ndarray, cfg: SegmentationConfig, gpu: bool = False) -> np.ndarray:
    """Dispatch to the configured segmentation method and return labels.

    Args:
        image: Segmentation image (2D, or 3D for ``method="threshold"``).
        cfg: Segmentation parameters.
        gpu: Request GPU inference for Cellpose.

    Returns:
        Integer label array (0 = background).

    Raises:
        ValueError: On an unknown method, or 3D input to Cellpose.
        SegmentationError: If the image is uniform or Cellpose fails.
    """
    img = np.asarray(image, dtype=np.float32)

    if cfg.method == "threshold":
        mask = threshold_segmentation(img, cfg)
        if cfg.watershed_split and mask.any():
            return watershed_split(mask)
        return label_mask(mask)

    if cfg.method == "cellpose":
        if img.ndim != 2:
            raise ValueError(
                f"Cellpose in HIVE handles 2D images only, got shape {img.shape}. "
                "Use method='threshold' for volumes."
            )
        return segment_with_cellpose(img, cfg, gpu=gpu)

    raise ValueError(f"Unknown segmentation method {cfg.method!r}; use 'threshold' or 'cellpose'")
