"""LEGACY (HIVE 1.x) per-object morphology and intensity measurement.

.. deprecated:: 2.0
    Superseded by :mod:`hive.measurements`.

Warning:
    **Intensity values produced through this module are not fluorescence.**
    The legacy pipeline (:func:`src.pipeline.analyze_single_image`) passes its
    *preprocessed* image as ``intensity_image``. That image has been min-max
    normalised per image and CLAHE-equalised by :func:`src.preprocessing.preprocess`,
    so every image is rescaled to span [0, 1] regardless of how bright it
    actually was. A genuine 3x difference in brightness between two samples is
    reported as 1.0x. HIVE 2.0's two-image contract exists to fix precisely
    this. Object counts and areas from this module remain valid.
"""

import pandas as pd
from skimage import measure, morphology
from .preprocessing import ensure_2d


def measure_objects(labeled, intensity_image, image_name=""):
    """Build a per-object table of shape and intensity statistics.

    Args:
        labeled: Integer label image (``0`` = background).
        intensity_image: Image sampled for intensity statistics. See the
            module-level warning about what the legacy pipeline passes here.
        image_name: Source filename, copied into the ``image`` column so
            per-image tables can be concatenated across a batch.

    Returns:
        :class:`pandas.DataFrame` with one row per labelled object. Empty if
        the label image contains no objects.

    Note:
        Areas are reported in pixels only -- this module has no access to
        voxel size, so nothing here can be converted to physical units.
        ``eccentricity`` and ``perimeter`` are fetched with ``getattr``
        defaults because scikit-image omits them for some region shapes.
    """
    intensity = ensure_2d(intensity_image)
    rows = []
    for region in measure.regionprops(labeled, intensity_image=intensity):
        rows.append({
            "image": image_name,
            "label": int(region.label),
            "area_px": int(region.area),
            "bbox_min_row": int(region.bbox[0]),
            "bbox_min_col": int(region.bbox[1]),
            "bbox_max_row": int(region.bbox[2]),
            "bbox_max_col": int(region.bbox[3]),
            "centroid_row": float(region.centroid[0]),
            "centroid_col": float(region.centroid[1]),
            "mean_intensity": float(region.mean_intensity),
            "max_intensity": float(region.max_intensity),
            "eccentricity": float(getattr(region, "eccentricity", 0.0)),
            "perimeter": float(getattr(region, "perimeter", 0.0)),
        })
    return pd.DataFrame(rows)


def summarize_image(measurements_df, image_name=""):
    """Collapse a per-object table into a one-row per-image summary.

    Args:
        measurements_df: Output of :func:`measure_objects`. ``None`` or empty
            yields an explicit all-zero row so images with no detections still
            appear in the batch summary instead of vanishing.
        image_name: Value for the ``image`` column.

    Returns:
        Single-row :class:`pandas.DataFrame`.

    Note:
        ``mean_intensity`` here is the unweighted mean *across objects*, not
        the intensity-weighted mean across pixels; a tiny object counts as
        much as a large one.
    """
    if measurements_df is None or measurements_df.empty:
        return pd.DataFrame([{
            "image": image_name,
            "object_count": 0,
            "total_area_px": 0,
            "mean_object_area_px": 0,
            "mean_intensity": 0,
        }])
    return pd.DataFrame([{
        "image": image_name,
        "object_count": int(len(measurements_df)),
        "total_area_px": float(measurements_df["area_px"].sum()),
        "mean_object_area_px": float(measurements_df["area_px"].mean()),
        "mean_intensity": float(measurements_df["mean_intensity"].mean()),
    }])


def skeletonize_mask(mask):
    """Reduce a mask to its 1-pixel-wide medial axis.

    Args:
        mask: Boolean mask.

    Returns:
        Boolean skeleton.

    Note:
        The legacy pipeline saves this as a PNG but never measures it. Skeleton
        length, branch points and Sholl metrics arrived in
        :mod:`hive.arborization`; there is no branch topology here.
    """
    return morphology.skeletonize(mask.astype(bool))
