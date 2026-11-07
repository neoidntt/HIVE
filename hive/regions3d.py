"""Real 3D surface meshes of every segmented region, built from the full z-stack volume.

Each region (an object of the label volume) gets its own closed triangle mesh from marching cubes
(Lorensen and Cline, 1987) on that region's voxels in all the z-planes it spans. The meshes carry
true physical coordinates, so anisotropic stacks are not squashed, and a surface normal at every
vertex, so a viewer can shade them with light.

Why this exists: the earlier "3D segmentation" view drew a random 12,000 of the segmented voxels as
dots. Dots have no surface, no light and no depth cue, and a thin stack drawn true to scale puts them
all in a slab a few percent as deep as it is wide, which reads as a flat picture on a plane.

The region's own bounding box (the ``bbox-0..5`` columns of the measurement table) is used to read
only the voxels around it, so this also works on a label image too large for memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

_TARGET_DIMS = (72, 48, 32, 20)           # largest mesh grid side tried, coarser until under budget


@dataclass
class RegionMeshes:
    """The meshes of many regions in one set of arrays.

    Attributes:
        positions: ``(n, 3)`` float32 vertex positions in physical units, ordered ``(x, y, z)``.
        normals: ``(n, 3)`` float32 outward unit normals, ordered ``(x, y, z)``.
        faces: ``(m, 3)`` uint32 triangle corners (indices into ``positions``).
        region: ``(n,)`` uint32 index into ``table`` of the region each vertex belongs to.
        table: One dict per region: ``label`` (the id in the label volume), ``voxels``, ``volume``
            (physical), ``triangles``; in the order of ``region`` (largest region first).
        skipped: Labels that gave no surface (nothing to draw).
        grid_side: Largest mesh grid side that fitted the triangle budget (smaller = coarser).
        omitted: Regions left out because there were more than ``max_regions`` (the smallest).
    """

    positions: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    region: np.ndarray
    table: list[dict] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    grid_side: int = 0
    omitted: int = 0


def region_boxes(measurements: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    """Bounding boxes ``label -> (z0, y0, x0, z1, y1, x1)`` from a measurement table.

    Raises:
        ValueError: If the table has no ``bbox-0 .. bbox-5`` columns (3D measurements have them).
    """
    columns = [f"bbox-{i}" for i in range(6)]
    if measurements is None or any(c not in measurements for c in columns) or "label" not in measurements:
        raise ValueError("The measurement table needs 'label' and 'bbox-0' to 'bbox-5' columns.")
    boxes = measurements[["label"] + columns].to_numpy()
    return {int(row[0]): tuple(int(v) for v in row[1:]) for row in boxes}


def boxes_from_volume(labels: np.ndarray) -> dict[int, tuple[int, ...]]:
    """Bounding boxes found by scanning a label volume (for a volume held in memory)."""
    found = ndimage.find_objects(np.asarray(labels))
    return {i + 1: (b[0].start, b[1].start, b[2].start, b[0].stop, b[1].stop, b[2].stop)
            for i, b in enumerate(found) if b is not None}


def _one_region(labels, label: int, box: tuple[int, ...], spacing: tuple[float, float, float],
                smooth: float, grid_side: int):
    """Mesh of one region: ``(vertices xyz, normals xyz, faces, voxel count)`` or ``None``."""
    z0, y0, x0, z1, y1, x1 = box
    crop = np.asarray(labels[z0:z1, y0:y1, x0:x1]) == label
    voxels = int(crop.sum())
    if voxels == 0:
        return None
    padded = np.pad(crop, 1).astype(np.float32)      # a closed surface even at the volume's edge
    sz, sy, sx = spacing
    # Each axis is thinned on its own, only as far as it needs to be: a thin sheet (a few planes deep,
    # hundreds of pixels wide) keeps every plane while its width is thinned. A blur proportional to the
    # stride first keeps the thinning from aliasing, and also takes the stair-steps off the surface.
    strides = tuple(max(1, int(np.ceil(n / grid_side))) for n in padded.shape)
    base = (max(0.3, smooth * min(sy, sx) / sz), smooth, smooth) if smooth > 0 else (0.0, 0.0, 0.0)
    sigma = tuple(max(b, 0.5 * (st - 1)) for b, st in zip(base, strides))
    thin = tuple(slice(None, None, st) for st in strides)
    scaled = (sz * strides[0], sy * strides[1], sx * strides[2])
    attempts = []
    if any(sigma):
        attempts.append((ndimage.gaussian_filter(padded, sigma)[thin], scaled))
    attempts.append((padded[thin], scaled))          # a region too thin to survive the blur
    for volume, voxel in attempts:
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                volume, level=0.5, spacing=voxel, allow_degenerate=False)
        except (ValueError, RuntimeError):
            continue
        verts = verts + np.array([z0 - 1, y0 - 1, x0 - 1], dtype=np.float64) * np.array([sz, sy, sx])
        return verts[:, ::-1], normals[:, ::-1], faces, voxels        # (z, y, x) -> (x, y, z)
    return None


def build_region_meshes(
    labels,
    boxes: Mapping[int, tuple[int, ...]],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    smooth: float = 0.8,
    max_triangles: int = 1_500_000,
    max_regions: int = 4000,
    progress: Callable[[float, str], None] | None = None,
) -> RegionMeshes:
    """Build a surface mesh for every region in ``boxes``.

    Args:
        labels: Label volume ``(Z, Y, X)``; an array or a memory map (only the voxels inside each
            region's box are read).
        boxes: ``label -> (z0, y0, x0, z1, y1, x1)``; see :func:`region_boxes`.
        spacing: Physical voxel size ``(z, y, x)``.
        smooth: Blur (in voxels) applied to each region before the surface is found; 0 for none.
        max_triangles: Budget for the whole set. If a first pass is over it, regions are meshed on a
            coarser grid until it fits (the table's ``grid_side`` says what was used).
        max_regions: At most this many regions are meshed (the largest); the rest are counted in
            ``omitted``. A browser cannot usefully draw tens of thousands of separate surfaces.
        progress: Optional ``progress(fraction, message)`` callback.

    Returns:
        A :class:`RegionMeshes`, largest region first. Regions that give no surface are listed in
        ``skipped``.
    """
    labels_list = list(boxes)
    sizes = {lab: int(np.prod([boxes[lab][i + 3] - boxes[lab][i] for i in range(3)])) for lab in labels_list}
    order = sorted(labels_list, key=lambda lab: -sizes[lab])
    omitted = max(0, len(order) - max_regions)
    order = order[:max_regions]
    best = None
    for pass_number, side in enumerate(_TARGET_DIMS):
        pieces, skipped, total = [], [], 0
        for index, lab in enumerate(order):
            if progress and index % 25 == 0:
                progress((pass_number + index / max(1, len(order))) / len(_TARGET_DIMS),
                         f"surfaces: region {index + 1} of {len(order)}")
            made = _one_region(labels, lab, boxes[lab], spacing, smooth, side)
            if made is None:
                skipped.append(lab)
                continue
            pieces.append((lab, made))
            total += len(made[2])
        best = (pieces, skipped, side, total)
        if total <= max_triangles:
            break
    pieces, skipped, side, _ = best
    volume_per_voxel = float(np.prod(spacing))
    positions, normals, faces, region, table, offset = [], [], [], [], [], 0
    for table_index, (lab, (verts, norms, tri, voxels)) in enumerate(pieces):
        positions.append(verts.astype(np.float32))
        norms = norms / np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-9)
        normals.append(norms.astype(np.float32))
        faces.append((tri + offset).astype(np.uint32))
        region.append(np.full(len(verts), table_index, dtype=np.uint32))
        table.append({"label": int(lab), "voxels": voxels, "volume": voxels * volume_per_voxel,
                      "triangles": int(len(tri))})
        offset += len(verts)
    if not pieces:
        empty = np.zeros((0, 3), np.float32)
        return RegionMeshes(empty, empty, np.zeros((0, 3), np.uint32), np.zeros(0, np.uint32), [], skipped,
                            side, omitted)
    return RegionMeshes(np.concatenate(positions), np.concatenate(normals), np.concatenate(faces),
                        np.concatenate(region), table, skipped, side, omitted)
