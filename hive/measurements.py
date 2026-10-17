"""Morphology and intensity measurement.

Everything here reads the **measurement image** -- background-corrected and in
its original intensity scale -- never the contrast-enhanced segmentation image.
See :mod:`hive.preprocessing` for the measured consequences of confusing them.

Implementation note: measurements use :func:`skimage.measure.regionprops_table`
rather than a Python loop over ``regionprops``. The legacy code appended one
dict per object inside a for-loop; the table form pushes the per-object work
into scikit-image's compiled code and returns columns directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure

# Shape descriptors defined only for 2D regions; requesting them for a 3D
# label image raises inside scikit-image.
_PROPS_2D: tuple[str, ...] = (
    "label", "area", "bbox", "centroid", "eccentricity", "perimeter", "solidity",
    "equivalent_diameter_area", "axis_major_length", "axis_minor_length",
    "orientation", "extent",
)
_PROPS_3D: tuple[str, ...] = (
    "label", "area", "bbox", "centroid", "equivalent_diameter_area", "extent", "solidity",
)
_INTENSITY_PROPS: tuple[str, ...] = ("intensity_mean", "intensity_max", "intensity_min")


def measure_objects(
    labels: np.ndarray,
    measurement_image: np.ndarray,
    image_name: str = "",
    voxel_size_xy: float = 1.0,
    voxel_size_z: float = 1.0,
    unit: str = "px",
) -> pd.DataFrame:
    """Measure morphology and intensity of every labelled object.

    Intensity columns are computed on ``measurement_image``, which must be the
    background-corrected image in its original scale. Passing a normalised or
    CLAHE-equalised image here silently produces numbers that cannot be compared
    between images -- the exact defect this package was refactored to fix.

    Physical units are derived from voxel size. With the defaults
    (``voxel_size_*=1.0``) values are in pixels and ``unit="px"``.

    Columns include ``integrated_intensity``, the sum of corrected intensity over
    the object. Because the background is already removed, this is equivalent to
    the corrected total cell fluorescence (CTCF) commonly reported in the
    fluorescence literature, and is usually the quantity of interest: mean
    intensity alone confounds a bright small object with a dim large one.

    Args:
        labels: Integer label array (2D or 3D).
        measurement_image: Background-corrected intensity image, same shape as
            ``labels``.
        image_name: Source image name, recorded in the ``image`` column so
            concatenated batch tables stay attributable.
        voxel_size_xy: Physical pixel size in XY.
        voxel_size_z: Physical z-step (3D only).
        unit: Unit label used in column names.

    Returns:
        One row per object. Empty (but correctly typed) when no objects exist.

    Raises:
        ValueError: If ``labels`` and ``measurement_image`` shapes differ, or
            the arrays are not 2D/3D.
    """
    labels = np.asarray(labels)
    intensity = np.asarray(measurement_image, dtype=np.float32)

    if labels.shape != intensity.shape:
        raise ValueError(
            f"labels shape {labels.shape} != measurement image shape {intensity.shape}. "
            "Measurements must be taken on an image matching the label array."
        )
    if labels.ndim not in (2, 3):
        raise ValueError(f"measure_objects expects 2D or 3D data, got shape {labels.shape}")
    if labels.max() == 0:
        return pd.DataFrame()

    is_3d = labels.ndim == 3
    props = (_PROPS_3D if is_3d else _PROPS_2D) + _INTENSITY_PROPS
    table = measure.regionprops_table(labels, intensity_image=intensity, properties=props)
    df = pd.DataFrame(table)
    df.insert(0, "image", image_name)

    if is_3d:
        df = df.rename(columns={"area": "volume_voxels"})
        voxel_volume = float(voxel_size_xy) ** 2 * float(voxel_size_z)
        df[f"volume_{unit}3"] = df["volume_voxels"] * voxel_volume
        size_px = df["volume_voxels"]
    else:
        df = df.rename(columns={"area": "area_px"})
        df[f"area_{unit}2"] = df["area_px"] * float(voxel_size_xy) ** 2
        df[f"perimeter_{unit}"] = df["perimeter"] * float(voxel_size_xy)
        size_px = df["area_px"]

    # Integrated (total) corrected fluorescence -- mean x size, in raw intensity
    # units. This is the background-corrected integrated density.
    df["integrated_intensity"] = df["intensity_mean"] * size_px

    return df


def summarize_measurements(
    measurements: pd.DataFrame, image_name: str = "", unit: str = "px"
) -> pd.DataFrame:
    """Reduce a per-object table to a one-row per-image summary.

    Reports both mean and median object size. They diverge when the size
    distribution is skewed -- a couple of merged clumps drag the mean up while
    the median holds -- so a large gap between them is a useful segmentation
    warning rather than a statistic to average away.

    Args:
        measurements: Output of :func:`measure_objects`.
        image_name: Image name for the summary row.
        unit: Unit label, must match the one used for measurement.

    Returns:
        Single-row DataFrame. Zero-object images yield a valid row with zero
        counts rather than an empty frame, so a batch summary keeps one row per
        image and "no objects found" stays visible instead of vanishing.
    """
    size_col = f"volume_{unit}3" if f"volume_{unit}3" in measurements.columns else f"area_{unit}2"

    if measurements is None or measurements.empty:
        return pd.DataFrame([{
            "image": image_name,
            "object_count": 0,
            f"total_{size_col}": 0.0,
            f"mean_{size_col}": 0.0,
            f"median_{size_col}": 0.0,
            "mean_intensity": 0.0,
            "total_integrated_intensity": 0.0,
        }])

    return pd.DataFrame([{
        "image": image_name,
        "object_count": int(len(measurements)),
        f"total_{size_col}": float(measurements[size_col].sum()),
        f"mean_{size_col}": float(measurements[size_col].mean()),
        f"median_{size_col}": float(measurements[size_col].median()),
        "mean_intensity": float(measurements["intensity_mean"].mean()),
        "total_integrated_intensity": float(measurements["integrated_intensity"].sum()),
    }])
