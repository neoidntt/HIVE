"""Post-processing filters for noisy segmentation output (Phase 3.2).

These operate on **label arrays**, not boolean masks, so an object survives or
dies as a whole. Filtering a boolean mask and re-labelling afterwards can split
one rejected object into several surviving fragments.

Every filter reports what it removed. Silent filtering is how a "cell count"
becomes an artifact of a debris threshold nobody wrote down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation as skseg

from .config import SegmentationConfig


@dataclass
class CleanupReport:
    """Audit trail of what mask cleanup removed.

    Attributes:
        initial_count: Objects before filtering.
        final_count: Objects after filtering.
        removed_small: Objects dropped as debris (below ``min_size``).
        removed_large: Objects dropped as oversized (above ``max_size``).
        removed_border: Objects dropped for touching the image edge.
        holes_filled: Holes filled, in pixels/voxels.
        steps: Human-readable log of each step applied.
    """

    initial_count: int = 0
    final_count: int = 0
    removed_small: int = 0
    removed_large: int = 0
    removed_border: int = 0
    holes_filled: int = 0
    steps: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Render a one-line human-readable summary.

        Returns:
            Summary string suitable for logs or a UI caption.
        """
        removed = self.initial_count - self.final_count
        parts = [f"{self.initial_count} -> {self.final_count} objects ({removed} removed)"]
        if self.removed_small:
            parts.append(f"{self.removed_small} debris")
        if self.removed_large:
            parts.append(f"{self.removed_large} oversized")
        if self.removed_border:
            parts.append(f"{self.removed_border} edge-touching")
        return "; ".join(parts)


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to be consecutive from 1 with no gaps.

    Removing objects leaves holes in the label sequence; consecutive labels keep
    the exported table's ``label`` column meaningful.

    Args:
        labels: Integer label array.

    Returns:
        Relabelled array of the same shape.
    """
    out, _, _ = skseg.relabel_sequential(labels)
    return out.astype(np.int32)


def filter_by_size(
    labels: np.ndarray, min_size: int = 0, max_size: int = 0
) -> tuple[np.ndarray, int, int]:
    """Remove objects outside a size window.

    Small objects are typically shot noise, dust or out-of-focus debris; very
    large ones are usually merged clumps, saturated bleed or a coverslip edge.

    Args:
        labels: Integer label array.
        min_size: Minimum area/volume in px/voxels. ``0`` disables.
        max_size: Maximum area/volume. ``0`` disables (no upper limit).

    Returns:
        Tuple of (filtered labels, n removed as small, n removed as large).
    """
    if labels.max() == 0 or (min_size <= 0 and max_size <= 0):
        return labels, 0, 0

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background is not an object

    too_small = (counts < min_size) & (counts > 0) if min_size > 0 else np.zeros_like(counts, bool)
    too_large = (counts > max_size) & (counts > 0) if max_size > 0 else np.zeros_like(counts, bool)

    doomed = too_small | too_large
    if not doomed.any():
        return labels, 0, 0

    out = labels.copy()
    out[doomed[labels]] = 0
    return _relabel_sequential(out), int(too_small.sum()), int(too_large.sum())


def remove_edge_objects(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove objects touching the image border.

    Border objects are truncated by the field of view, so their area, perimeter
    and morphology are lower bounds rather than measurements. Including them
    biases mean size downward. Counting them but not measuring them is also
    valid -- this drops them entirely, so use it when morphology matters more
    than absolute count.

    Args:
        labels: Integer label array.

    Returns:
        Tuple of (filtered labels, number removed).
    """
    if labels.max() == 0:
        return labels, 0
    before = len(np.unique(labels)) - 1
    cleared = skseg.clear_border(labels)
    after = len(np.unique(cleared)) - 1
    return _relabel_sequential(cleared), int(before - after)


def fill_object_holes(labels: np.ndarray, hole_size: int = 0) -> tuple[np.ndarray, int]:
    """Fill interior holes in labelled objects.

    Holes arise where a nucleus is dimmer than its cytoplasm, or from threshold
    speckle. ``hole_size`` bounds what may be filled, so genuine lumens (a
    vessel, a ventricle) are preserved.

    The legacy 3D path accepted a ``hole_size`` argument and ignored it, calling
    ``binary_fill_holes`` which fills every hole regardless of size -- so the
    CLI and UI knob did nothing in 3D.

    Args:
        labels: Integer label array.
        hole_size: Maximum hole area/volume to fill. ``0`` disables filling;
            a negative value fills all holes regardless of size.

    Returns:
        Tuple of (labels with holes filled, pixels/voxels filled).
    """
    if hole_size == 0 or labels.max() == 0:
        return labels, 0

    mask = labels > 0
    # Hole filling is done directly rather than via skimage's remove_small_holes,
    # whose `area_threshold` argument is deprecated and slated for removal in
    # scikit-image 2.0. binary_fill_holes finds every enclosed region; we then
    # keep only those small enough to be artifacts.
    filled_all = ndi.binary_fill_holes(mask)
    all_holes = filled_all & ~mask

    if hole_size < 0 or not all_holes.any():
        added = all_holes
    else:
        # ndi.label with full connectivity labels the same holes as measure.label but in int32
        # instead of int64 (half the memory); only the size of each hole is used, so the
        # numbering does not matter.
        hole_labels, _ = ndi.label(all_holes, structure=np.ones((3,) * all_holes.ndim, dtype=bool))
        hole_sizes = np.bincount(hole_labels.ravel())
        keep = hole_sizes <= int(hole_size)
        keep[0] = False  # index 0 is "not a hole"
        added = keep[hole_labels] & all_holes

    n_added = int(added.sum())
    if n_added == 0:
        return labels, 0

    # Assign each filled pixel to the object enclosing it via nearest-label lookup.
    out = labels.copy()
    if not _assign_holes_locally(out, labels, mask, added):
        # Only the nearest-pixel indices are needed; asking for the distances too allocates a
        # float64 array the size of the image (0.5 GB at 66 megapixels) that is thrown away.
        indices = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        out[added] = labels[tuple(idx[added] for idx in indices)]
    return out, n_added


_MAX_LOCAL_HOLES = 20_000


def _assign_holes_locally(out, labels, mask, added) -> bool:
    """Give each hole voxel the label of its nearest object, looking only near each hole.

    A hole is enclosed by object pixels, so every voxel in it has an object pixel within one
    pixel of the hole's bounding box; the nearest one is therefore inside that box grown by
    two pixels, and a distance transform of that small crop finds it. This avoids transforming
    the whole image (13 s for a 28-million-voxel volume) to fill a few hundred voxels.

    Returns:
        ``False`` (nothing written) when there are too many holes for this to pay off.
    """
    holes, count = ndi.label(added, structure=np.ones((3,) * added.ndim, dtype=bool))
    if count > _MAX_LOCAL_HOLES:
        return False
    boxes = ndi.find_objects(holes)
    if sum(int(np.prod([s.stop - s.start for s in b])) for b in boxes) > 0.5 * mask.size:
        return False
    for number, box in enumerate(boxes, start=1):
        grown = tuple(slice(max(0, s.start - 2), min(n, s.stop + 2))
                      for s, n in zip(box, mask.shape))
        indices = ndi.distance_transform_edt(~mask[grown], return_distances=False,
                                             return_indices=True)
        target = holes[grown] == number
        out[grown][target] = labels[grown][tuple(idx[target] for idx in indices)]
    return True


def clean_labels(
    labels: np.ndarray, cfg: SegmentationConfig, hole_size: int | None = None
) -> tuple[np.ndarray, CleanupReport]:
    """Apply the configured cleanup filters in a defensible order.

    Order: fill holes (so filled objects are sized correctly) -> drop border
    objects -> filter by size (last, so it sees final sizes). Filtering by size
    before filling would judge objects on a hole-riddled area.

    Args:
        labels: Integer label array from segmentation.
        cfg: Segmentation parameters supplying the thresholds.
        hole_size: Override ``cfg.hole_size``.

    Returns:
        Tuple of (cleaned labels, :class:`CleanupReport`).
    """
    report = CleanupReport(initial_count=int(labels.max()))
    out = labels

    holes = cfg.hole_size if hole_size is None else hole_size
    if holes:
        out, filled = fill_object_holes(out, holes)
        report.holes_filled = filled
        if filled:
            report.steps.append(f"filled {filled} px of holes (<= {holes} px)")

    if cfg.clear_border:
        out, n_border = remove_edge_objects(out)
        report.removed_border = n_border
        report.steps.append(f"removed {n_border} edge-touching objects")

    if cfg.min_size > 0 or cfg.max_size > 0:
        out, n_small, n_large = filter_by_size(out, cfg.min_size, cfg.max_size)
        report.removed_small, report.removed_large = n_small, n_large
        if n_small:
            report.steps.append(f"removed {n_small} objects < {cfg.min_size} px (debris)")
        if n_large:
            report.steps.append(f"removed {n_large} objects > {cfg.max_size} px (clumps)")

    report.final_count = int(out.max())
    return out, report


def normalize_for_display(
    image: np.ndarray, method: str = "percentile", low: float = 1.0, high: float = 99.5
) -> np.ndarray:
    """Rescale an image to ``[0, 1]`` for DISPLAY only.

    WARNING -- do not feed the output to any measurement. Every mode here is
    per-image, so an object's value depends on its own frame's extremes. This is
    precisely the operation that made the legacy pipeline report a real 3x
    biological difference as 1.0x. It is exposed because contrast-stretching for
    a figure is legitimate; quantifying the stretched result is not.

    Args:
        image: Input image.
        method: ``"percentile"`` clips to the ``low``/``high`` percentiles
            (robust to hot pixels); ``"minmax"`` uses true extremes (one dead
            pixel at 0 and one hot pixel at max destroys the range).
        low: Lower percentile for ``"percentile"``.
        high: Upper percentile for ``"percentile"``.

    Returns:
        float32 array in ``[0, 1]``.

    Raises:
        ValueError: On an unknown method or an invalid percentile pair.
    """
    arr = np.asarray(image, dtype=np.float32)
    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
    elif method == "percentile":
        if not 0 <= low < high <= 100:
            raise ValueError(f"Need 0 <= low < high <= 100, got low={low}, high={high}")
        lo, hi = (float(v) for v in np.percentile(arr, [low, high]))
    else:
        raise ValueError(f"Unknown method {method!r}; use 'percentile' or 'minmax'")

    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
