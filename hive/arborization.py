"""Arborization analysis: skeletonisation, branch metrics, and Sholl analysis.

This module is **new in HIVE 2.0**. The legacy pipeline advertised arborisation
analysis but only ever called ``skeletonize_mask``, saved the result as a PNG,
and never read it back: the measurement table contained no branch, junction,
neurite-length or Sholl column.

Methods implemented
-------------------
Skeletonisation
    Medial-axis thinning. 2D uses Zhang & Suen (1984)-family thinning; 3D uses
    Lee, Kashyap & Chu (1994), both via scikit-image. The skeleton is a 1-px
    centreline preserving the arbor's topology.
Branch topology
    Classified by 8-/26-neighbour count on the skeleton: exactly 1 neighbour is
    a terminal tip, >= 3 is a bifurcation. This is the standard digital
    approximation used by Fiji's AnalyzeSkeleton (Arganda-Carreras et al., 2010).
Sholl analysis
    Sholl (1953). Concentric shells at increasing radius from the soma; the
    number of skeleton crossings at each radius profiles branching complexity
    against distance. Derived scalars follow the conventions in Sholl (1953) and
    the Fiji Sholl plugin (Ferreira et al., 2014, *Nat Methods*).

Assumption: one arbor per labelled object, with a soma-like centre. For densely
overlapping neuropil where processes from many cells intermingle, per-object
Sholl is not meaningful -- segmentation cannot assign a neurite to its parent
soma, and no amount of skeleton analysis fixes that.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology

from .config import ArborConfig


@dataclass
class ShollResult:
    """Result of a Sholl analysis on one arbor.

    Attributes:
        radii: Shell radii, in physical units.
        intersections: Crossing count at each radius (``N(r)``).
        center: Coordinates of the shell centre, in array index space.
        max_intersections: Peak of ``N(r)``; the arbor's maximum branching density.
        critical_radius: Radius at ``max_intersections`` -- where the arbor is
            most complex.
        enclosing_radius: Largest radius with any crossing; the arbor's reach.
        n_primary: Crossings at the innermost shell, i.e. processes leaving the
            soma.
        ramification_index: ``max_intersections / n_primary``. Branching gain
            relative to the number of primary processes; dimensionless, so it
            compares neurons with different numbers of primary dendrites.
        regression_coefficient: Sholl regression coefficient ``k`` from the
            semi-log fit ``log10(N(r) / area(r)) = log10(K) - k*r``. Higher ``k``
            means intersection density falls off faster with distance, i.e. a
            more compact arbor. ``nan`` when fewer than 3 non-zero shells exist.
        total_intersections: Sum of ``N(r)``, a coarse complexity proxy.
    """

    radii: np.ndarray
    intersections: np.ndarray
    center: tuple[float, ...]
    max_intersections: int = 0
    critical_radius: float = 0.0
    enclosing_radius: float = 0.0
    n_primary: int = 0
    ramification_index: float = float("nan")
    regression_coefficient: float = float("nan")
    total_intersections: int = 0

    def to_frame(self) -> pd.DataFrame:
        """Return the full radius-vs-intersections profile.

        Returns:
            DataFrame with ``radius`` and ``intersections`` columns -- the raw
            Sholl curve, for plotting or re-fitting.
        """
        return pd.DataFrame({"radius": self.radii, "intersections": self.intersections})


@dataclass
class SkeletonMetrics:
    """Topological measurements of one skeletonised arbor.

    Attributes:
        skeleton: The (pruned) skeleton, same shape as the input mask.
        total_length: Summed skeleton length in physical units.
        n_branch_points: Bifurcation count.
        n_endpoints: Terminal tip count.
        n_branches: Distinct branch segments between nodes.
        max_branch_length: Longest single segment.
        mean_branch_length: Mean segment length.
        pruned_spurs: Short terminal spurs removed before counting.
    """

    skeleton: np.ndarray = field(repr=False)
    total_length: float = 0.0
    n_branch_points: int = 0
    n_endpoints: int = 0
    n_branches: int = 0
    max_branch_length: float = 0.0
    mean_branch_length: float = 0.0
    pruned_spurs: int = 0


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce a mask to its 1-px-wide medial-axis skeleton.

    Args:
        mask: Boolean mask, 2D or 3D.

    Returns:
        Boolean skeleton of the same shape.

    Raises:
        ValueError: If the mask is not 2D or 3D.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim not in (2, 3):
        raise ValueError(f"skeletonize_mask expects a 2D or 3D mask, got shape {m.shape}")
    if not m.any():
        return m
    return morphology.skeletonize(m).astype(bool)


# The 8-neighbourhood traversed as a closed ring, starting north and going
# clockwise. Order matters: the crossing number counts transitions between
# *circularly adjacent* neighbours, so the sequence must walk the ring.
_RING_2D: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)


def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    """Count skeleton neighbours of every pixel (8-connected 2D, 26-connected 3D).

    Args:
        skeleton: Boolean skeleton.

    Returns:
        Integer array of neighbour counts; meaningful only where the skeleton
        is True.
    """
    kernel = np.ones((3,) * skeleton.ndim, dtype=np.uint8)
    kernel[(1,) * skeleton.ndim] = 0  # exclude the centre pixel itself
    return ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)


def crossing_number(skeleton: np.ndarray) -> np.ndarray:
    """Compute the Rutovitz crossing number of every 2D skeleton pixel.

    The crossing number counts how many separate branches emanate from a pixel::

        CN(p) = 0.5 * sum_i |P(i+1) - P(i)|

    over the 8-neighbourhood traversed as a closed ring. It classifies a pixel
    by topology rather than by raw neighbour count: ``1`` = terminal tip,
    ``2`` = ordinary path pixel, ``>=3`` = true bifurcation.

    Why this replaced neighbour-counting
    ------------------------------------
    Counting neighbours misclassifies a **bump** -- a single pixel resting on a
    straight line. It touches 3 line pixels, so ``degree >= 3`` calls it a
    junction, yet its neighbours are all mutually adjacent, so no branch
    actually departs there. Verified during development: pruning a spur left its
    base pixel behind, that pixel was mislabelled a junction, and being a
    "junction" excluded it from the prunable segments -- so it survived every
    iteration and held a phantom bifurcation open. Branch counts stayed at 3
    when the ground truth was 1.

    The crossing number gets both cases right: for the bump the ring is
    ``00011100``, a single 0->1 transition, so ``CN=1`` (a tip); for a genuine
    T-junction the ring is ``10100010``, three transitions, so ``CN=3``.

    Args:
        skeleton: Boolean 2D skeleton.

    Returns:
        Integer crossing numbers; meaningful only where the skeleton is True.

    Raises:
        ValueError: If the skeleton is not 2D. The crossing number is a planar
            construction with no direct 26-neighbourhood analogue.
    """
    skel = np.asarray(skeleton, dtype=bool)
    if skel.ndim != 2:
        raise ValueError(f"crossing_number is 2D-only, got shape {skel.shape}")

    padded = np.pad(skel.astype(np.int16), 1)
    rows, cols = skel.shape
    ring = [padded[1 + dr: 1 + dr + rows, 1 + dc: 1 + dc + cols] for dr, dc in _RING_2D]

    transitions = np.zeros(skel.shape, dtype=np.int16)
    for i in range(len(_RING_2D)):
        transitions += np.abs(ring[(i + 1) % len(_RING_2D)] - ring[i])
    return (transitions // 2).astype(np.int16)


def find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Locate terminal tips of a skeleton.

    2D uses the crossing number (``CN == 1``), which correctly treats a pixel
    resting on a line as a tip. 3D falls back to ``degree == 1``.

    Args:
        skeleton: Boolean skeleton.

    Returns:
        Boolean array marking endpoints.
    """
    skel = np.asarray(skeleton, dtype=bool)
    if skel.ndim == 2:
        return skel & (crossing_number(skel) == 1)
    return skel & (_neighbour_count(skel) == 1)


def find_branch_points(skeleton: np.ndarray) -> np.ndarray:
    """Locate bifurcations of a skeleton.

    2D uses the crossing number (``CN >= 3``) -- see :func:`crossing_number` for
    why raw neighbour-counting is unsafe.

    3D uses ``degree >= 3``, the same criterion as Fiji's AnalyzeSkeleton
    (Arganda-Carreras et al., 2010). This is the accepted standard but inherits
    the bump false-positive, since the crossing number has no direct
    26-neighbourhood analogue. In practice adjacent junction voxels are grouped
    into one junction by :func:`skeleton_metrics`, which absorbs most of the
    error; treat 3D junction counts as slightly upward-biased.

    A digital bifurcation often spans a small cluster of adjacent pixels rather
    than one, so :func:`skeleton_metrics` counts connected *clusters*, not pixels.

    Args:
        skeleton: Boolean skeleton.

    Returns:
        Boolean array marking branch-point pixels.
    """
    skel = np.asarray(skeleton, dtype=bool)
    if skel.ndim == 2:
        return skel & (crossing_number(skel) >= 3)
    return skel & (_neighbour_count(skel) >= 3)


def _branch_segments(skeleton: np.ndarray, branch_points: np.ndarray) -> np.ndarray:
    """Split a skeleton into branch segments by cutting it at its junctions.

    The cut set is the junction pixels **dilated by one**, which is essential
    rather than cosmetic. Deleting a single junction pixel does not disconnect
    an 8-connected skeleton: the arms meeting there remain diagonally adjacent
    around the one-pixel gap and re-merge into a single component. Verified on a
    4-arm star, where cutting the exact centre pixel yielded 1 segment instead
    of 4, and a spur stayed attached to its parent because its base reconnected
    diagonally around the removed junction.

    Cost: each segment loses roughly a pixel at every junction end, so branch
    lengths are marginally underestimated. Total skeleton length is measured on
    the uncut skeleton and is unaffected.

    Args:
        skeleton: Boolean skeleton.
        branch_points: Boolean array marking junction pixels.

    Returns:
        Integer label array of branch segments.
    """
    skel = np.asarray(skeleton, dtype=bool)
    if not branch_points.any():
        return measure.label(skel)
    footprint = np.ones((3,) * skel.ndim, dtype=bool)
    cut = ndi.binary_dilation(branch_points, structure=footprint) & skel
    return measure.label(skel & ~cut)


def _as_spacing(ndim: int, spacing: tuple[float, ...] | float) -> np.ndarray:
    """Normalise a scalar-or-tuple spacing into a per-axis array.

    Args:
        ndim: Array dimensionality.
        spacing: Scalar (isotropic) or one value per axis.

    Returns:
        Float array of length ``ndim``.

    Raises:
        ValueError: If a sequence is given whose length differs from ``ndim``.
    """
    if np.isscalar(spacing):
        return np.full(ndim, float(spacing))  # type: ignore[arg-type]
    spac = np.asarray(spacing, dtype=float)
    if spac.size != ndim:
        raise ValueError(f"spacing has {spac.size} entries but the array is {ndim}-D")
    return spac


def _forward_offsets(ndim: int) -> list[tuple[int, ...]]:
    """Neighbour offsets covering each adjacent pair exactly once.

    Taking only lexicographically-positive offsets counts the pair (p, p+off)
    but not (p+off, p), so adjacencies are not double-counted.

    Args:
        ndim: Array dimensionality.

    Returns:
        List of offset tuples.
    """
    offsets = []
    for off in itertools.product((-1, 0, 1), repeat=ndim):
        if any(off) and next(v for v in off if v != 0) > 0:
            offsets.append(off)
    return offsets


def skeleton_length(skeleton: np.ndarray, spacing: tuple[float, ...] | float = 1.0) -> float:
    """Measure total skeleton length with diagonal correction.

    Counting skeleton pixels underestimates length, because a diagonal step
    spans sqrt(2) px but counts as 1. This sums the true Euclidean distance of
    every adjacent pixel pair instead: 1 for an orthogonal step, sqrt(2) for a
    face diagonal, sqrt(3) for a 3D body diagonal, each scaled by voxel spacing.

    This slightly *over*estimates where an 8-connected skeleton forms a
    2-px triangle (a diagonal plus its two orthogonal legs), which thinning
    makes rare but not impossible. The bias is small and upward; report it as an
    approximation rather than an exact arc length.

    Args:
        skeleton: Boolean skeleton.
        spacing: Physical size per axis, or a scalar for isotropic data. For a
            z-stack use ``(voxel_size_z, voxel_size_xy, voxel_size_xy)``.

    Returns:
        Total length in physical units.
    """
    skel = np.asarray(skeleton, dtype=bool)
    if not skel.any():
        return 0.0

    spac = _as_spacing(skel.ndim, spacing)

    total = 0.0
    for off in _forward_offsets(skel.ndim):
        # Overlap of the skeleton with itself shifted by `off` = adjacent pairs.
        src = tuple(slice(max(o, 0), skel.shape[i] + min(o, 0)) for i, o in enumerate(off))
        dst = tuple(slice(max(-o, 0), skel.shape[i] - max(o, 0)) for i, o in enumerate(off))
        n_pairs = int(np.count_nonzero(skel[src] & skel[dst]))
        if n_pairs:
            total += n_pairs * float(np.linalg.norm(np.asarray(off) * spac))
    return total


def _neighbours(skeleton: np.ndarray, point: tuple[int, ...]) -> list[tuple[int, ...]]:
    """List in-bounds skeleton neighbours of a pixel.

    Args:
        skeleton: Boolean skeleton.
        point: Pixel coordinates.

    Returns:
        Coordinates of each True neighbour.
    """
    out: list[tuple[int, ...]] = []
    for off in itertools.product((-1, 0, 1), repeat=skeleton.ndim):
        if not any(off):
            continue
        q = tuple(point[i] + off[i] for i in range(len(point)))
        if all(0 <= q[i] < skeleton.shape[i] for i in range(len(q))) and skeleton[q]:
            out.append(q)
    return out


def _walk_spur(
    skeleton: np.ndarray,
    junctions: np.ndarray,
    start: tuple[int, ...],
    min_length: float,
    spacing: np.ndarray,
) -> list[tuple[int, ...]] | None:
    """Walk from a terminal tip toward its parent, returning the spur if short.

    Follows the unambiguous degree-2 path away from ``start``. The walk stops
    when it reaches the parent structure -- signalled either by stepping to a
    junction pixel, or by the path fanning out to several neighbours at once
    (which is what a spur base resting on a line looks like). If the tip was
    reached within ``min_length``, every pixel walked is a spur.

    Bailing out as soon as ``min_length`` is exceeded is what keeps this cheap:
    long real neurites are abandoned after a few steps rather than traced.

    Args:
        skeleton: Boolean skeleton.
        junctions: Boolean array marking junction pixels.
        start: Coordinates of the terminal tip.
        min_length: Spur length threshold in physical units.
        spacing: Physical size per axis.

    Returns:
        The spur's pixel coordinates (including ``start``), or ``None`` if this
        is a real branch, an isolated fragment, or an ambiguous path.
    """
    path = [start]
    visited = {start}
    current = start
    length = 0.0

    while True:
        nbrs = [n for n in _neighbours(skeleton, current) if n not in visited]
        if not nbrs:
            return None  # isolated fragment: it has no parent, so it is not a spur
        if len(nbrs) > 1 or any(junctions[n] for n in nbrs):
            # Reached the parent structure -- the spur ends at `current`.
            return path if length < min_length else None

        nxt = nbrs[0]
        step = float(np.linalg.norm((np.asarray(nxt) - np.asarray(current)) * spacing))
        if length + step >= min_length:
            return None  # long enough to be a real branch; leave it alone
        length += step
        path.append(nxt)
        visited.add(nxt)
        current = nxt


def prune_skeleton(
    skeleton: np.ndarray, min_length: int = 3, spacing: tuple[float, ...] | float = 1.0
) -> tuple[np.ndarray, int]:
    """Remove short terminal spurs from a skeleton.

    Thinning a mask with a ragged boundary sprouts short false twigs, each
    adding a spurious junction and endpoint. This traces each terminal tip back
    to its parent and deletes the whole path when it is shorter than
    ``min_length``. Internal segments are never walked (they have no tip), so a
    genuinely short branch *between* two junctions survives.

    Why a walk and not a mask operation
    -----------------------------------
    Two simpler designs were tried and both failed on a 4-arm star with two
    3-px twigs:

    * Eroding endpoints ``min_length`` times also shortens real neurites.
    * Deleting short terminal *segments* (skeleton minus junction pixels) leaves
      the spur's base behind, because that base is part of the junction cut set
      and therefore belongs to no segment. The residual stub kept the phantom
      junction alive and the branch count stuck at 3 instead of 1.

    Walking pixel-by-pixel has no immune set: every pixel on the path is removed.

    Args:
        skeleton: Boolean skeleton.
        min_length: Terminal spurs shorter than this (physical units) are cut.
            ``0`` disables pruning.
        spacing: Physical size per axis.

    Returns:
        Tuple of (pruned skeleton, number of spurs removed).
    """
    skel = np.asarray(skeleton, dtype=bool).copy()
    if min_length <= 0 or not skel.any():
        return skel, 0

    spac = _as_spacing(skel.ndim, spacing)

    total_removed = 0
    for _ in range(10):  # cutting a spur can expose another; converges in 1-2 passes
        endpoints = find_endpoints(skel)
        if not endpoints.any():
            break
        junctions = find_branch_points(skel)

        doomed: set[tuple[int, ...]] = set()
        for tip in map(tuple, np.argwhere(endpoints)):
            if tip in doomed:
                continue
            spur = _walk_spur(
                skel, junctions, tip, float(min_length), spac,  # type: ignore[arg-type]
            )
            if spur is not None:
                doomed.update(spur)
                total_removed += 1

        if not doomed:
            break
        for pixel in doomed:
            skel[pixel] = False
    return skel, total_removed


def skeleton_metrics(
    mask: np.ndarray, prune_length: int = 3, spacing: tuple[float, ...] | float = 1.0
) -> SkeletonMetrics:
    """Skeletonise a mask and measure its branching topology.

    Junctions are counted as connected *clusters* of high-degree pixels, not raw
    pixels: a single Y-bifurcation frequently thins to 2-3 mutually adjacent
    pixels each having 3+ neighbours, which would otherwise count as 2-3
    junctions.

    Args:
        mask: Boolean mask of one object.
        prune_length: Terminal spurs below this length are removed first.
        spacing: Physical size per axis.

    Returns:
        A :class:`SkeletonMetrics`.
    """
    skel = skeletonize_mask(mask)
    if not skel.any():
        return SkeletonMetrics(skeleton=skel)

    skel, n_pruned = prune_skeleton(skel, prune_length, spacing)
    if not skel.any():
        return SkeletonMetrics(skeleton=skel, pruned_spurs=n_pruned)

    branch_pts = find_branch_points(skel)
    endpoints = find_endpoints(skel)

    n_junctions = int(measure.label(branch_pts).max()) if branch_pts.any() else 0

    segments = _branch_segments(skel, branch_pts)
    lengths = [
        skeleton_length(segments == region.label, spacing)
        for region in measure.regionprops(segments)
    ]
    lengths = [length for length in lengths if length > 0]

    return SkeletonMetrics(
        skeleton=skel,
        total_length=skeleton_length(skel, spacing),
        n_branch_points=n_junctions,
        n_endpoints=int(measure.label(endpoints).max()) if endpoints.any() else 0,
        n_branches=len(lengths),
        max_branch_length=float(max(lengths)) if lengths else 0.0,
        mean_branch_length=float(np.mean(lengths)) if lengths else 0.0,
        pruned_spurs=n_pruned,
    )


def _distance_map(shape: tuple[int, ...], center: tuple[float, ...],
                  spacing: np.ndarray) -> np.ndarray:
    """Euclidean distance from ``center`` to every voxel, in physical units.

    Args:
        shape: Array shape.
        center: Centre coordinates in index space.
        spacing: Physical size per axis.

    Returns:
        Float32 distance array.
    """
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    squared = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, center, spacing))
    return np.sqrt(squared).astype(np.float32)


def sholl_analysis(
    skeleton: np.ndarray,
    center: tuple[float, ...],
    step: float = 5.0,
    max_radius: float = 0.0,
    spacing: tuple[float, ...] | float = 1.0,
    shell_thickness: float | None = None,
) -> ShollResult:
    """Run a Sholl analysis on a skeletonised arbor.

    Method (Sholl, 1953): place concentric shells -- circles in 2D, spheres in
    3D -- at increasing radius from the soma and count how many times the arbor
    crosses each. ``N(r)`` profiles branching complexity against distance from
    the soma, distinguishing (for example) a compact bushy arbor from a sparse
    far-reaching one even when both have identical total branch counts.

    Crossings at radius ``r`` are counted as the number of connected components
    of ``skeleton AND shell(r)`` -- the standard digital analogue of counting
    intersections with an ideal circle.

    Args:
        skeleton: Boolean skeleton of one arbor.
        center: Soma centre in index space; must match ``skeleton.ndim``.
        step: Radial spacing between shells, in physical units.
        max_radius: Outermost radius. ``0`` auto-selects the arbor's furthest
            skeleton pixel from the centre.
        spacing: Physical size per axis.
        shell_thickness: Shell width. Defaults to one voxel diagonal, the
            thinnest shell that a 1-px skeleton cannot slip through unseen.

    Returns:
        A :class:`ShollResult`.

    Raises:
        ValueError: If ``center`` length differs from the skeleton's
            dimensionality, or ``step`` is not positive.
    """
    skel = np.asarray(skeleton, dtype=bool)
    if len(center) != skel.ndim:
        raise ValueError(
            f"center has {len(center)} coordinates but skeleton is {skel.ndim}-D"
        )
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    spac = _as_spacing(skel.ndim, spacing)

    if not skel.any():
        return ShollResult(np.array([]), np.array([], dtype=int), tuple(center))

    dist = _distance_map(skel.shape, center, spac)
    if max_radius <= 0:
        max_radius = float(dist[skel].max())
    if shell_thickness is None:
        shell_thickness = float(np.linalg.norm(spac))

    radii = np.arange(step, max_radius + step, step, dtype=np.float32)
    if radii.size == 0:
        return ShollResult(np.array([]), np.array([], dtype=int), tuple(center))

    intersections = np.zeros(radii.size, dtype=int)
    for i, radius in enumerate(radii):
        shell = np.abs(dist - radius) <= (shell_thickness / 2.0)
        crossing = skel & shell
        if crossing.any():
            intersections[i] = int(measure.label(crossing).max())

    result = ShollResult(radii=radii, intersections=intersections, center=tuple(center))
    result.total_intersections = int(intersections.sum())
    if intersections.max() == 0:
        return result

    result.max_intersections = int(intersections.max())
    result.critical_radius = float(radii[int(np.argmax(intersections))])
    result.enclosing_radius = float(radii[np.nonzero(intersections)[0][-1]])
    result.n_primary = int(intersections[0])
    if result.n_primary > 0:
        result.ramification_index = result.max_intersections / result.n_primary
    result.regression_coefficient = _sholl_regression(radii, intersections, skel.ndim)
    return result


def _sholl_regression(radii: np.ndarray, intersections: np.ndarray, ndim: int) -> float:
    """Fit the Sholl regression coefficient ``k`` by the semi-log method.

    Sholl (1953) normalises the crossing count by the shell's area (2D) or
    surface area (3D) before fitting, because a larger shell samples more space
    and would otherwise show more crossings for purely geometric reasons::

        log10(N(r) / area(r)) = log10(K) - k * r

    ``k`` is the negated slope: a steeper decay means a more compact arbor.
    Only shells with ``N(r) > 0`` are fitted, since ``log10(0)`` is undefined.

    Args:
        radii: Shell radii.
        intersections: Crossing counts.
        ndim: 2 for circles, 3 for spheres.

    Returns:
        The coefficient ``k``, or ``nan`` if fewer than 3 non-zero shells exist
        (a fit through 2 points is not a regression).
    """
    valid = intersections > 0
    if int(valid.sum()) < 3:
        return float("nan")

    r = radii[valid].astype(np.float64)
    n = intersections[valid].astype(np.float64)
    area = np.pi * r ** 2 if ndim == 2 else 4.0 * np.pi * r ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(n / area)
    finite = np.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan")

    slope = np.polyfit(r[finite], y[finite], 1)[0]
    return float(-slope)


def _resolve_center(
    obj_mask: np.ndarray, intensity: np.ndarray | None, strategy: str
) -> tuple[float, ...]:
    """Choose the Sholl centre for one object.

    Args:
        obj_mask: Boolean mask of the object (cropped).
        intensity: Matching measurement-image crop, or ``None``.
        strategy: ``"centroid"`` or ``"max_intensity"``.

    Returns:
        Centre coordinates in index space.
    """
    if strategy == "max_intensity" and intensity is not None:
        masked = np.where(obj_mask, intensity, -np.inf)
        return tuple(float(c) for c in np.unravel_index(np.argmax(masked), masked.shape))
    return tuple(float(c) for c in ndi.center_of_mass(obj_mask))


def analyze_arbor(
    labels: np.ndarray,
    cfg: ArborConfig,
    intensity: np.ndarray | None = None,
    spacing: tuple[float, ...] | float = 1.0,
    unit: str = "px",
) -> tuple[pd.DataFrame, dict[int, ShollResult]]:
    """Measure arborisation for every labelled object.

    Each object is cropped to its bounding box before skeletonisation, so cost
    scales with total object area rather than image area -- important for sparse
    labelling in a large field of view.

    Args:
        labels: Integer label array (2D or 3D).
        cfg: Arborisation parameters.
        intensity: Measurement image, required for ``sholl_center="max_intensity"``.
        spacing: Physical size per axis.
        unit: Unit label used in column names.

    Returns:
        Tuple of (per-object metrics DataFrame, mapping of label -> full
        :class:`ShollResult` for plotting the curves).
    """
    rows: list[dict[str, float | int]] = []
    curves: dict[int, ShollResult] = {}

    if labels.max() == 0:
        return pd.DataFrame(), curves

    for region in measure.regionprops(labels):
        obj_mask = region.image  # cropped to the bounding box
        crop_intensity = intensity[region.slice] if intensity is not None else None

        metrics = skeleton_metrics(obj_mask, cfg.prune_length, spacing)
        row: dict[str, float | int] = {
            "label": int(region.label),
            f"neurite_length_{unit}": metrics.total_length,
            "n_branch_points": metrics.n_branch_points,
            "n_endpoints": metrics.n_endpoints,
            "n_branches": metrics.n_branches,
            f"max_branch_length_{unit}": metrics.max_branch_length,
            f"mean_branch_length_{unit}": metrics.mean_branch_length,
            "pruned_spurs": metrics.pruned_spurs,
        }

        if metrics.skeleton.any():
            center = _resolve_center(obj_mask, crop_intensity, cfg.sholl_center)
            sholl = sholl_analysis(
                metrics.skeleton, center, cfg.sholl_step, cfg.sholl_max_radius, spacing
            )
            curves[int(region.label)] = sholl
            row.update({
                "sholl_max_intersections": sholl.max_intersections,
                f"sholl_critical_radius_{unit}": sholl.critical_radius,
                f"sholl_enclosing_radius_{unit}": sholl.enclosing_radius,
                "sholl_n_primary": sholl.n_primary,
                "sholl_ramification_index": sholl.ramification_index,
                "sholl_regression_coeff_k": sholl.regression_coefficient,
                "sholl_total_intersections": sholl.total_intersections,
            })
        rows.append(row)

    return pd.DataFrame(rows), curves
