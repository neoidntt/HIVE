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

import warnings

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.morphology.convex_hull import _offsets_diamond
from skimage.util import unique_rows

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


_HULL_TOLERANCE = 1e-10      # scikit-image's own value for "inside the hull"


def convex_voxel_count(mask: np.ndarray) -> int:
    """Voxels inside the convex hull of a 3D mask: scikit-image's number, much faster.

    Same hull as :func:`skimage.morphology.convex_hull_image` and the same inequality per
    facet and voxel; only the way the voxels are counted differs.
    """
    try:
        coords = np.transpose(np.nonzero(mask))
        hull0 = ConvexHull(coords)
        coords = hull0.points[hull0.vertices]
        coords = unique_rows((coords[:, np.newaxis, :] + _offsets_diamond(3)).reshape(-1, 3))
        equations = ConvexHull(coords).equations
    except (QhullError, ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return int(convex_hull_image(mask).sum())
    # Each facet is  e0*z + e1*y + e2*x + e3 < tol.  For one (z, y) row that is a bound on
    # x, so the allowed x-range of every row is found directly and its integers are counted,
    # instead of testing every voxel of the bounding box against every facet.
    depth, height, width = mask.shape
    zz, yy = np.meshgrid(np.arange(depth, dtype=np.float64), np.arange(height, dtype=np.float64),
                         indexing="ij")
    zz, yy = zz.ravel(), yy.ravel()
    low = np.full(zz.size, -np.inf)
    high = np.full(zz.size, np.inf)
    row_ok = np.ones(zz.size, dtype=bool)
    for e0, e1, e2, e3 in equations:
        rest = (e0 * zz + e1 * yy + e3)
        if e2 > 0:
            np.minimum(high, (_HULL_TOLERANCE - rest) / e2, out=high)
        elif e2 < 0:
            np.maximum(low, (_HULL_TOLERANCE - rest) / e2, out=low)
        else:
            row_ok &= rest < _HULL_TOLERANCE
    first = np.floor(np.clip(low, -1.0, width)) + 1      # smallest integer x above `low`
    last = np.ceil(np.clip(high, -1.0, width)) - 1       # largest integer x below `high`
    count = np.clip(np.minimum(last, width - 1) - np.maximum(first, 0) + 1, 0, None)
    return int(count[row_ok].sum())


def _solidity_3d(labels: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Object volume divided by its convex-hull volume, one value per label in ``index``."""
    boxes = ndimage.find_objects(labels)
    out = np.full(len(index), np.nan)
    for i, label in enumerate(index):
        box = boxes[label - 1] if 0 < label <= len(boxes) else None
        if box is None:
            continue
        mask = labels[box] == label
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = np.float64(mask.sum()) / np.float64(convex_voxel_count(mask))
    return out


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
    if is_3d:
        props = tuple(p for p in props if p != "solidity")
    table = measure.regionprops_table(labels, intensity_image=intensity, properties=props)
    if is_3d:
        solidity = _solidity_3d(labels, table["label"])
        table = {
            name: column for key, column in table.items()
            for name, column in (((key, column), ("solidity", solidity)) if key == "extent"
                                 else ((key, column),))
        }
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
