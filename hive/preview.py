"""Looking at a tiled result without loading the whole volume.

A stack that was analysed in tiles is too big to hold in memory, but the label image it
produced sits on disk, and so does the source. Two kinds of view are built from those files:

* :func:`read_region` - a rectangle of one plane at full resolution (raw pixels and labels).
* :func:`overview` - the whole stack, reduced by keeping every ``stride``-th pixel in y and x,
  small enough to hold in memory (about 30 million voxels). It feeds the max projection, the
  3D point view and the rotation GIF.

Both are for looking only. Measurements come from the analysis, never from these views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .tiled import open_source

DEFAULT_MAX_VOXELS = 30_000_000


@dataclass
class Overview:
    """A reduced copy of a stack and its labels.

    Attributes:
        raw: ``(Z, h, w)`` float32 intensities, every ``stride``-th pixel of the source.
        labels: ``(Z, h, w)`` uint32 labels at the same pixels.
        stride: Source pixels per overview pixel in y and x (1 means full resolution).
        full_shape: ``(Z, Y, X)`` of the source.
    """

    raw: np.ndarray
    labels: np.ndarray
    stride: int
    full_shape: tuple[int, int, int]


def overview(path: str | Path, labels_path: str | Path, channel: str = "auto",
             max_voxels: int = DEFAULT_MAX_VOXELS) -> Overview:
    """Read the whole stack and its labels, keeping every ``stride``-th pixel in y and x.

    One plane is read at a time, so memory use is the overview itself plus one plane.

    Args:
        path: The stack that was analysed.
        labels_path: Its ``HIVE_3D_labels.ome.tif``.
        channel: Channel choice, as for the analysis.
        max_voxels: Upper bound on the overview's size.
    """
    labels_map = tifffile.memmap(str(labels_path), mode="r")
    with open_source(str(path), channel) as src:
        z, h, w = src.shape
        stride = max(1, math.ceil(math.sqrt(z * h * w / max_voxels)))
        rows, cols = -(-h // stride), -(-w // stride)
        raw = np.empty((z, rows, cols), np.float32)
        labels = np.empty((z, rows, cols), np.uint32)
        for index in range(z):
            raw[index] = src.read(index, slice(None), slice(None))[::stride, ::stride]
            labels[index] = labels_map[index, ::stride, ::stride]
    return Overview(raw, labels, stride, (z, h, w))


def read_region(path: str | Path, labels_path: str | Path, z: int, y0: int, x0: int,
                size: int, channel: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """One rectangle of one plane at full resolution: raw pixels and labels.

    Args:
        z: Plane index.
        y0, x0: Top-left corner (clipped to the image).
        size: Side of the square to read (clipped at the image edge).

    Returns:
        ``(raw, labels)``, both ``(h, w)``.
    """
    labels_map = tifffile.memmap(str(labels_path), mode="r")
    with open_source(str(path), channel) as src:
        _, height, width = src.shape
        y0, x0 = min(max(0, y0), height - 1), min(max(0, x0), width - 1)
        ys, xs = slice(y0, min(height, y0 + size)), slice(x0, min(width, x0 + size))
        raw = src.read(z, ys, xs)
    return raw, np.asarray(labels_map[z, ys, xs])


def max_projection(view: Overview) -> tuple[np.ndarray, np.ndarray]:
    """Brightest-voxel projection of the overview, with the label at each brightest voxel."""
    brightest = np.argmax(view.raw, axis=0)
    return (np.take_along_axis(view.raw, brightest[None], axis=0)[0],
            np.take_along_axis(view.labels, brightest[None], axis=0)[0])
