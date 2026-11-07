"""Tiled 3D analysis, for stacks that do not fit in memory.

The whole-volume pipeline needs about ten times the stack's size in RAM
(:data:`hive.io_utils.PIPELINE_MEMORY_FACTOR`). Here the stack is cut into XY tiles that
each carry the full Z range, and one tile at a time is read, analysed and discarded.

How the results are kept consistent with a whole-volume run:

* **One threshold for the whole volume.** A first pass samples the smoothed,
  background-corrected intensities of every tile (a seeded random subsample for very
  large volumes) and takes a single Otsu threshold. Cutting each tile at its own Otsu
  value would make the same object count in one tile and not in the next.
* **A halo around each tile.** Each tile is read with extra pixels on every side so
  background subtraction and smoothing see the same neighbourhood as they would in a
  whole-volume run.
* **Each object is kept once.** An object belongs to the tile whose core contains its
  centroid. An object reaching further than the halo from its centre could be cut at
  the edge of the tile that owns it; such objects are marked ``tile_clipped`` and
  counted in the warning, so they can be excluded or re-run with a larger halo.

What is different from a whole-volume run, and why (all recorded in ``hive_run.json``):

* CLAHE is switched off. It rescales intensities per image, so one threshold would not
  mean the same thing in every tile. Compare object counts with ``use_clahe=False`` on
  your own data before relying on this.
* Arborization / Sholl, extra channels, the 3D mesh viewer, the rotating GIF and the
  ImageJ ROI zip are not produced: they need the whole volume at once. The label image
  is written as an OME-TIFF, which Fiji and napari read without loading it all.
* Only Cellpose-free threshold segmentation is used (Cellpose is 2D only anyway).
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage import filters

from . import reporting
from .baseline import apply_baseline_subtraction
from .cleanup import CleanupReport, clean_labels
from .config import HiveConfig, PreprocessConfig
from .formats import read_microscope
from .io_utils import (
    PIPELINE_MEMORY_FACTOR, ImageLoadError, _available_memory_bytes, check_slice_files, collect_image_paths,
    load_image,
)
from .measurements import measure_objects, summarize_measurements
from .preprocessing import build_segmentation_image, select_channel
from .segmentation import label_mask, threshold_segmentation, watershed_split

logger = logging.getLogger(__name__)

# Voxels sampled for the global threshold (80 MB as float32). Smaller volumes use all.
_THRESHOLD_SAMPLE = 20_000_000
_MIN_HALO = 16
_COLOUR_NAMES = {"auto", "red", "green", "blue", "magenta", "pink", "gray"}


@dataclass
class _Source:
    """A stack that hands out rectangular pieces of single planes."""

    shape: tuple[int, int, int]
    read: Callable[[int, slice, slice], np.ndarray]
    description: str
    notes: list[str]

    def window(self, ys: slice, xs: slice) -> np.ndarray:
        """All z-planes of a rectangle, as a new float32 ``(Z, h, w)`` array."""
        out = np.empty((self.shape[0], ys.stop - ys.start, xs.stop - xs.start), np.float32)
        for z in range(self.shape[0]):
            out[z] = self.read(z, ys, xs)
        return out


@contextmanager
def open_source(path: str | Path, channel: str = "auto") -> Iterator[_Source]:
    """Open a TIFF stack or a folder of single-plane images for tiled reading.

    Args:
        path: Multi-page TIFF (any axis order containing Z, Y, X and optionally C), or
            a folder of images, one per z-plane.
        channel: Which channel, for multi-channel files: a name from the file or a
            number. ``"auto"`` takes the first channel (and says so in the notes).

    Yields:
        A :class:`_Source`.

    Raises:
        FileNotFoundError: If the path does not exist.
        ImageLoadError: If the layout is not a z-stack (time series, RGB samples, ...).
    """
    path = Path(path)
    if path.is_file():
        with _tiff_source(path, channel) as source:
            yield source
    elif path.is_dir():
        files = collect_image_paths(path)
        if not files:
            raise FileNotFoundError(f"No readable image slices in {path}")
        check_slice_files(files)
        first = load_image(files[0])
        notes: list[str] = []
        resolved = channel
        if first.ndim == 3:
            # "auto" is decided once, on the first slice, so every slice uses the same channel.
            first, resolved = select_channel(first, channel)  # type: ignore[arg-type]
            if str(channel).lower() == "auto":
                notes.append(f"Channel was set to auto: analysed the {resolved} channel (chosen on "
                             "the first slice and used for all slices).")

        def read(z: int, ys: slice, xs: slice) -> np.ndarray:
            plane = load_image(files[z])
            if plane.ndim == 3:
                plane = select_channel(plane, resolved)[0]  # type: ignore[arg-type]
            return plane[ys, xs]

        yield _Source((len(files), *first.shape), read, f"{len(files)} slices in {path.name}", notes)
    else:
        raise FileNotFoundError(f"Z-stack path does not exist: {path}")


@contextmanager
def _tiff_source(path: Path, channel: str) -> Iterator[_Source]:
    """Reading side of :func:`open_source` for one TIFF file."""
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes, shape = series.axes.upper(), tuple(series.shape)
        if "S" in axes and shape[axes.index("S")] > 1:
            raise ImageLoadError(
                f"{path.name} stores RGB colour (3 samples per pixel), which cannot be read in "
                "tiles. Save one channel as a single-channel TIFF, or an OME-TIFF with "
                "separate channels, and open that.")
        lead = [a for a in axes if a not in "YX"]
        lead_shape = tuple(shape[axes.index(a)] for a in lead)
        z_axis = next((a for a in ("Z", "I", "Q") if a in lead and a != "C"), None)
        notes: list[str] = []
        for a in lead:
            if a not in (z_axis, "C") and shape[axes.index(a)] != 1:
                raise ImageLoadError(
                    f"{path.name} has axis {a} of length {shape[axes.index(a)]}; tiled "
                    "analysis reads a single z-stack (no time series or scenes).")

        c_index = 0
        if "C" in lead and shape[axes.index("C")] > 1:
            if channel.lower() in _COLOUR_NAMES:
                notes.append(f"{shape[axes.index('C')]} channels in the file and channel="
                             f"'{channel}': used the first. Name a channel to choose another.")
            else:
                c_index = read_microscope(path, load_pixels=False).channel_index(channel)
        n_z = shape[axes.index(z_axis)] if z_axis else 1

        mapped = store = None
        try:
            mapped = tifffile.memmap(str(path), mode="r")   # free random access if uncompressed
            if tuple(mapped.shape) != shape:
                mapped = None
        except Exception:  # noqa: BLE001 - compressed or irregular: try the next way
            mapped = None
        if mapped is None:
            # Compressed or tiled files: zarr lets tifffile decode only the tiles or strips a
            # piece overlaps, instead of the whole plane.
            try:
                import zarr

                store = zarr.open(series.aszarr(), mode="r")
                if tuple(store.shape) != shape:
                    store = None
            except Exception:  # noqa: BLE001 - zarr missing or unsupported layout
                store = None
        if mapped is None and store is not None:
            notes.append("Compressed file: read in pieces (only the tiles each piece needs "
                         "are decoded).")
        elif mapped is None:
            plane_bytes = shape[axes.index("Y")] * shape[axes.index("X")] * 4 * 3
            free = _available_memory_bytes()
            if free and plane_bytes > free * 0.3:
                raise ImageLoadError(
                    f"{path.name} is compressed and one plane is about {plane_bytes / 1e9:.1f} GB "
                    "to decode, which cannot be read in pieces without the 'zarr' package. "
                    "Install it (pip install zarr) or save the file uncompressed.")
            notes.append("File is compressed or not contiguous, so every plane is decoded "
                         "again for each tile. Expect this to be slow. (pip install zarr "
                         "lets it be read in pieces.)")

        def read(z: int, ys: slice, xs: slice) -> np.ndarray:
            index = tuple(z if a == z_axis else (c_index if a == "C" else 0) for a in lead)
            if mapped is not None:
                return np.asarray(mapped[index + (ys, xs)], dtype=np.float32)
            if store is not None:
                return np.asarray(store[index + (ys, xs)], dtype=np.float32)
            page = int(np.ravel_multi_index(index, lead_shape)) if lead else 0
            return series.pages[page].asarray()[ys, xs].astype(np.float32)

        yield _Source((n_z, shape[axes.index("Y")], shape[axes.index("X")]), read,
                      f"{path.name} {shape} {axes}", notes)
        del mapped, store


def stack_float32_bytes(path: str | Path) -> int | None:
    """Size of a stack as float32, without loading it; ``None`` if it cannot be told."""
    path = Path(path)
    try:
        if path.is_file():
            with tifffile.TiffFile(str(path)) as tif:
                return int(np.prod(tif.series[0].shape)) * 4
        if path.is_dir():
            files = collect_image_paths(path)
            if files:
                probe = tifffile.TiffFile(str(files[0])) if files[0].suffix.lower() in (
                    ".tif", ".tiff") else None
                if probe is not None:
                    with probe:
                        return int(np.prod(probe.series[0].shape)) * 4 * len(files)
    except Exception:  # noqa: BLE001 - unreadable: let the normal loader report it
        return None
    return None


def needs_tiling(cfg: HiveConfig) -> bool:
    """Whether this run should use tiles.

    ``mode="force"`` always does; ``"off"`` never does; ``"auto"`` does when the
    whole-volume pipeline would not fit in RAM (measured peak: ten times the stack).
    """
    mode = cfg.large_image.mode
    if mode != "auto":
        return mode == "force"
    size, ram = stack_float32_bytes(cfg.input_path), _available_memory_bytes()
    return bool(size and ram and size * PIPELINE_MEMORY_FACTOR > ram * 0.85)


def filter_halo(pre: PreprocessConfig) -> int:
    """Pixels a tile needs around it for the filters to match a whole-volume run."""

    def xy(value) -> float:
        return float(value[-1] if isinstance(value, (tuple, list)) else value)

    reach = 0.0
    if pre.baseline_method == "tophat":
        reach = xy(pre.tophat_radius)
    elif pre.baseline_method == "smooth":
        reach = 4.0 * xy(pre.baseline_sigma)         # scipy's default truncation
    reach += 4.0 * pre.gaussian_sigma
    return max(_MIN_HALO, int(math.ceil(reach)) + 2)


def choose_tile_size(shape: tuple[int, int, int], halo: int, workers: int = 1) -> int:
    """Largest square tile core such that ``workers`` tiles at once fit in about half of RAM."""
    z, y, x = shape
    ram = _available_memory_bytes()
    if not ram:
        return min(1024, max(y, x))
    voxels = ram * 0.5 / (4 * PIPELINE_MEMORY_FACTOR * max(1, workers))
    side = int(math.sqrt(voxels / max(z, 1))) - 2 * halo
    return int(min(max(64, side // 64 * 64), max(y, x)))


def _grid(y: int, x: int, tile: int) -> list[tuple[int, int, int, int]]:
    """Tile cores ``(y0, y1, x0, x1)`` covering the image without overlap."""
    return [(y0, min(y0 + tile, y), x0, min(x0 + tile, x))
            for y0 in range(0, y, tile) for x0 in range(0, x, tile)]


def _smoothed(volume: np.ndarray, pre: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Background-corrected volume and its smoothed copy (the thresholded image)."""
    corrected, _ = apply_baseline_subtraction(
        volume, method=pre.baseline_method, percentile=pre.baseline_percentile,
        sigma=pre.baseline_sigma, tophat_radius=pre.tophat_radius, normalize=False,
        return_baseline=False, copy=False,
    )
    return corrected, build_segmentation_image(corrected, pre, rescale=False)


def _xy(value) -> float:
    """The in-plane part of a scalar or per-axis (z, y, x) size."""
    return float(value[-1] if isinstance(value, (tuple, list)) else value)


def _tile_preprocess(cfg: HiveConfig, plane: bool) -> PreprocessConfig:
    """Preprocessing as used inside a tile: CLAHE off, and 2D radii for a single plane."""
    pre = replace(cfg.preprocess, use_clahe=False)
    if plane:
        pre = replace(pre, tophat_radius=_xy(pre.tophat_radius), baseline_sigma=_xy(pre.baseline_sigma))
    return pre


def _read_window(cfg: HiveConfig, path: str, plane: bool, box: tuple[int, int, int, int]) -> np.ndarray:
    """One rectangle of the source as float32: ``(h, w)`` for a plane, ``(Z, h, w)`` for a stack."""
    y0, y1, x0, x1 = box
    with open_source(path, cfg.channel) as src:
        data = src.window(slice(y0, y1), slice(x0, x1))
    return data[0] if plane else data


def _window_box(core, halo, y_size, x_size):
    y0, y1, x0, x1 = core
    return (max(0, y0 - halo), min(y_size, y1 + halo), max(0, x0 - halo), min(x_size, x1 + halo))


def _threshold_sample(job) -> np.ndarray:
    """Pass 1 for one tile: its smoothed, corrected intensities (a lattice subsample if huge).

    The subsample is every ``stride``-th pixel of the whole image in y and x, counted from
    the image origin, so the same pixels are used however the image is cut into tiles. That
    keeps the threshold (and so the result) independent of the tile size, which depends on
    the memory that happens to be free.
    """
    cfg, path, plane, core, halo, size, stride = job
    box = _window_box(core, halo, size[1], size[2])
    _, smooth = _smoothed(_read_window(cfg, path, plane, box), _tile_preprocess(cfg, plane))
    y0 = -core[0] % stride
    x0 = -core[2] % stride
    piece = smooth[..., core[0] - box[0] + y0:core[1] - box[0]:stride,
                   core[2] - box[2] + x0:core[3] - box[2]:stride]
    return piece.ravel().copy()


def _segment_tile(job) -> dict:
    """Pass 2 for one tile: segment, keep the objects this tile owns, and measure them.

    Returns only small things: the table of owned objects (tile-local labels, global
    coordinates) and the positions and labels of their voxels, never the tile itself.
    """
    cfg, path, plane, core, halo, size, threshold, name = job
    z_size, y_size, x_size = size
    seg_cfg = replace(cfg.segmentation, clear_border=False)   # applied to the whole image below
    box = _window_box(core, halo, y_size, x_size)
    wy0, wy1, wx0, wx1 = box
    corrected, smooth = _smoothed(_read_window(cfg, path, plane, box), _tile_preprocess(cfg, plane))
    mask = threshold_segmentation(smooth, seg_cfg, threshold=threshold)
    del smooth
    labels = (watershed_split(mask) if cfg.segmentation.watershed_split and mask.any()
              else label_mask(mask))
    labels, _ = clean_labels(labels, seg_cfg)
    table = measure_objects(
        labels, corrected, image_name=name, voxel_size_xy=cfg.voxel_size_xy,
        voxel_size_z=cfg.voxel_size_z, unit=cfg.voxel_unit)
    del corrected, mask
    empty = {"table": pd.DataFrame(), "index": None, "local": None, "box": box,
             "shape": labels.shape, "clipped": 0}
    if table.empty:
        return empty

    # Column names differ by dimensionality: (y, x) for a plane, (z, y, x) for a stack.
    k = 0 if plane else 1
    cy_col, cx_col = f"centroid-{k}", f"centroid-{k + 1}"
    d = 2 if plane else 3
    ymin, xmin, ymax, xmax = (f"bbox-{k}", f"bbox-{k + 1}", f"bbox-{d + k}", f"bbox-{d + k + 1}")
    h, w = labels.shape[-2:]
    cy, cx = table[cy_col] + wy0, table[cx_col] + wx0
    owned = (cy >= core[0]) & (cy < core[1]) & (cx >= core[2]) & (cx < core[3])
    table["tile_clipped"] = (((table[ymin] == 0) & (wy0 > 0)) | ((table[ymax] == h) & (wy1 < y_size))
                             | ((table[xmin] == 0) & (wx0 > 0)) | ((table[xmax] == w) & (wx1 < x_size)))
    for column, offset in ((cy_col, wy0), (cx_col, wx0), (ymin, wy0), (ymax, wy0),
                           (xmin, wx0), (xmax, wx0)):
        table[column] += offset
    if cfg.segmentation.clear_border:
        touches = ((table[ymin] == 0) | (table[ymax] == y_size)
                   | (table[xmin] == 0) | (table[xmax] == x_size))
        if not plane:
            touches |= (table["bbox-0"] == 0) | (table["bbox-3"] == z_size)
        owned &= ~touches
    table = table[owned].copy()
    if table.empty:
        return empty

    keep_labels = np.zeros(int(labels.max()) + 1, dtype=np.uint32)
    keep_labels[table["label"].to_numpy()] = table["label"].to_numpy()
    kept = keep_labels[labels]
    index = np.flatnonzero(kept)
    return {"table": table, "index": index, "local": kept.ravel()[index], "box": box,
            "shape": labels.shape, "clipped": int(table["tile_clipped"].sum())}


def _repeat_note(run: dict) -> list[str]:
    """A note saying how many objects two overlapping tiles both found (each is counted once)."""
    n = run.get("repeats", 0)
    return [f"{n} object(s) lay where two tiles overlap and were found by both; each is counted once "
            "(the first tile's outline is kept)."] if n else []


def _tile_workers(cfg: HiveConfig, n_tiles: int) -> int:
    """Tiles processed at once: as set in ``cfg.large_image.workers``, else one per core."""
    wanted = cfg.large_image.workers or min(os.cpu_count() or 1, 4)
    return max(1, min(wanted, n_tiles))


_REPEAT_FRACTION = 0.5     # an object is the same one again when at least this share of it overlaps another


class _Painter:
    """Gives objects their final numbers and paints them into the label image, tile by tile.

    An object that lies where two tiles overlap is segmented by both, slightly differently, and its
    centre can fall on either side of the seam, so both tiles may "own" it. Two objects cannot really
    share voxels, so an overlap means one physical object found twice:

    * a new object with at least :data:`_REPEAT_FRACTION` of its voxels on voxels an earlier tile already
      gave to another object is dropped (the earlier one stays);
    * an earlier object that the new one covers by at least that share is dropped, and its leftover
      voxels are removed (the larger, newer one stays).

    Either way the object is counted once, and every row of the final table has voxels in the label image.
    Tiles are handled in a fixed order, so the result never depends on timing.
    """

    def __init__(self, mosaic, plane: bool):
        self.mosaic, self.plane = mosaic, plane
        self.tables: list[pd.DataFrame] = []
        self.next_id = 1
        self.repeats = 0
        self._left: dict[int, int] = {}          # voxels still carrying each id
        self._first: dict[int, int] = {}         # voxels each id was painted with
        self._box: dict[int, tuple] = {}         # its bounding box, to find its voxels again
        self._gone: set[int] = set()

    def add(self, tile_result: dict) -> None:
        table = tile_result["table"]
        if table.empty:
            return
        wy0, wy1, wx0, wx1 = tile_result["box"]
        region = self.mosaic[..., wy0:wy1, wx0:wx1]
        where = np.unravel_index(tile_result["index"], tile_result["shape"])
        local = tile_result["local"]
        span = int(table["label"].max()) + 1
        before = region[where]
        taken = np.bincount(local[before != 0], minlength=span)
        repeated = taken > _REPEAT_FRACTION * np.maximum(np.bincount(local, minlength=span), 1)
        if repeated.any():
            dropped = repeated[table["label"].to_numpy()]
            table = table[~dropped].copy()
            self.repeats += int(dropped.sum())
            keep = ~repeated[local]
            where, local, before = tuple(w[keep] for w in where), local[keep], before[keep]
        if table.empty:
            return
        new_ids = np.arange(self.next_id, self.next_id + len(table), dtype=np.uint32)
        lut = np.zeros(int(table["label"].max()) + 1, dtype=np.uint32)
        lut[table["label"].to_numpy()] = new_ids
        # earlier objects this one covers: how much of each is gone, and is it mostly gone?
        covered = before[before != 0]
        for old, count in zip(*np.unique(covered, return_counts=True)):
            self._left[int(old)] -= int(count)
        painted = np.bincount(local, minlength=span)[table["label"].to_numpy()]
        d = 2 if self.plane else 3
        boxes = table[[f"bbox-{i}" for i in range(2 * d)]].to_numpy()
        for new, count, box in zip(new_ids.tolist(), painted.tolist(), boxes):
            self._first[new], self._left[new], self._box[new] = count, count, tuple(int(v) for v in box)
        region[where] = lut[local]
        for old in np.unique(covered):
            old = int(old)
            if old not in self._gone and self._left[old] < (1.0 - _REPEAT_FRACTION) * self._first[old]:
                self._gone.add(old)
                box = self._box[old]
                block = self.mosaic[tuple(slice(box[i], box[i + d]) for i in range(d))]
                block[block == old] = 0                      # its leftover voxels go with it
        table["label"] = new_ids
        self.next_id += len(table)
        self.tables.append(table)

    def finish(self) -> pd.DataFrame:
        """The final table: every object once, in the order the tiles gave them."""
        if not self.tables:
            return pd.DataFrame()
        table = pd.concat(self.tables, ignore_index=True)
        self.repeats += len(self._gone)
        return table[~table["label"].isin(self._gone)].reset_index(drop=True)


def _run_tiled(cfg: HiveConfig, path: str, plane: bool, out: Path, labels_path: Path, name: str,
               progress: Callable[[float, str], None] | None) -> dict:
    """The tiled analysis shared by 3D stacks and single huge planes.

    Returns:
        Dict with ``measurements``, ``labels_path``, ``threshold``, ``tiles``, ``tile``,
        ``halo``, ``clipped`` and ``source_notes``.
    """
    if cfg.segmentation.method != "threshold":
        raise ValueError("Tiled analysis supports threshold segmentation only.")
    report = progress or (lambda fraction, message: None)
    pre = _tile_preprocess(cfg, plane)

    with open_source(path, cfg.channel) as src:
        size, notes = src.shape, list(src.notes)
    z_size, y_size, x_size = size
    halo = cfg.large_image.halo or filter_halo(pre)
    workers = cfg.large_image.workers or min(os.cpu_count() or 1, 4)
    tile = cfg.large_image.tile_size or choose_tile_size(size, halo, workers)
    tiles = _grid(y_size, x_size, tile)
    workers = _tile_workers(cfg, len(tiles))
    logger.info("Tiled analysis of %s: %d tiles of %d px, halo %d, %d worker(s)", name,
                len(tiles), tile, halo, workers)

    def run(function, jobs):
        """Results in job order, from one process or several."""
        if workers == 1:
            for job in jobs:
                yield function(job)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                yield from pool.map(function, jobs, chunksize=1)

    # Pass 1: one threshold for the whole image.
    stride = max(1, int(math.sqrt(z_size * y_size * x_size / _THRESHOLD_SAMPLE)))
    samples = []
    jobs = ((cfg, path, plane, core, halo, size, stride) for core in tiles)
    for i, piece in enumerate(run(_threshold_sample, jobs)):
        samples.append(piece)
        report(0.4 * (i + 1) / len(tiles), f"threshold pass, tile {i + 1}/{len(tiles)}")
    pooled = np.concatenate(samples)
    del samples
    if float(pooled.max()) <= float(pooled.min()):
        # As in the whole-image run, a blank image is a result (0 objects), not a crash.
        warnings.warn(
            "The image is uniform after background subtraction (an empty channel?). "
            "Reporting 0 objects.", stacklevel=2)
        threshold = float("inf")
    else:
        threshold = float(filters.threshold_otsu(pooled)) * cfg.segmentation.threshold_scale
    del pooled

    # Pass 2: segment tiles (in parallel), then give objects their final numbers and paint
    # the label file here, in tile order, so the numbering never depends on timing.
    mosaic = _create_label_file(labels_path, (y_size, x_size) if plane else size, cfg)
    painter = _Painter(mosaic, plane)
    jobs = ((cfg, path, plane, core, halo, size, threshold, name) for core in tiles)
    for i, tile_result in enumerate(run(_segment_tile, jobs)):
        painter.add(tile_result)
        report(0.4 + 0.6 * (i + 1) / len(tiles), f"tile {i + 1}/{len(tiles)}")
    mosaic.flush()
    del mosaic
    final = painter.finish()
    clipped_total = int(final["tile_clipped"].sum()) if len(final) else 0
    if clipped_total:
        logger.warning("%d object(s) reach beyond the %d px halo and may be cut at a tile "
                       "edge (column tile_clipped). Use a larger halo or exclude them.",
                       clipped_total, halo)
    return {
        "measurements": final,
        "labels_path": labels_path, "threshold": threshold, "tiles": len(tiles), "tile": tile,
        "halo": halo, "workers": workers, "clipped": clipped_total, "source_notes": notes,
        "repeats": painter.repeats,
    }


def analyze_zstack_tiled(
    cfg: HiveConfig, progress: Callable[[float, str], None] | None = None, save: bool = True,
):
    """Analyse a 3D stack tile by tile, writing tables and an OME-TIFF label image.

    Args:
        cfg: Run configuration; ``cfg.large_image`` sets tile size, halo and workers.
        progress: Optional ``progress(fraction, message)`` callback.
        save: ``False`` keeps the results out of ``cfg.output_folder``. The label image still
            has to be built on disk while the tiles are done, so it goes in a scratch folder
            that is removed before this returns; the result then has an empty ``labels`` and
            ``saved`` set to ``False``.

    Returns:
        An :class:`~hive.pipeline.AnalysisResult` whose ``labels`` is a read-only
        memory map of the label image and whose ``prepared`` is ``None`` (the corrected
        volume is never held whole).

    Raises:
        FileNotFoundError: If the stack does not exist.
        ImageLoadError: If it is not a plain z-stack.
        ValueError: If segmentation is not the threshold method.
    """
    if not save:
        with tempfile.TemporaryDirectory(prefix="hive_nosave_") as scratch:
            return _analyze_zstack_tiled(replace(cfg, output_folder=scratch), progress, False)
    return _analyze_zstack_tiled(cfg, progress, True)


def _analyze_zstack_tiled(cfg: HiveConfig, progress, save: bool):
    """The work of :func:`analyze_zstack_tiled`; with ``save`` false the folder is scratch."""
    from . import provenance
    from .pipeline import AnalysisResult

    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(cfg.input_path).name
    labels_path = out / "HIVE_3D_labels.ome.tif"
    run = _run_tiled(cfg, cfg.input_path, False, out, labels_path, name, progress)

    measurements = run["measurements"]
    summary = summarize_measurements(measurements, name, cfg.voxel_unit)
    if not save:
        cleanup = CleanupReport(initial_count=len(measurements), final_count=len(measurements))
        cleanup.steps.append(f"tiled: {run['tiles']} tiles, global threshold {run['threshold']:.4g}")
        result = AnalysisResult(
            measurements=measurements, summary=summary, labels=np.empty((0, 0, 0), np.uint32),
            prepared=None, cleanup=cleanup, name=name,  # type: ignore[arg-type]
            notes=list(run["source_notes"]) + _repeat_note(run), saved=False)
        import json

        result.blobs["settings_json"] = json.dumps(provenance.build_run_record(cfg, notes={
            "tiled": {"tiles": run["tiles"], "tile_size": run["tile"], "halo": run["halo"],
                      "workers": run["workers"], "global_threshold": run["threshold"],
                      "clahe": "disabled in tiled mode", "tile_clipped_objects": run["clipped"]},
        }), indent=2, default=str).encode()
        result.notes.append(
            "This stack is too large for memory, so it was analysed in tiles, and nothing was saved: "
            "the region and 3D views need the label image, which was only a scratch file and has "
            "been removed. Untick \"Don't save results to this computer\" to keep it and see them.")
        return result
    reporting.export_csv(out / "HIVE_3D_measurements.csv", measurements)
    reporting.export_csv(out / "HIVE_3D_summary.csv", summary)
    extra_sheets = _group_sheets(out, summary, measurements, cfg)
    reporting.export_excel_report(out / "HIVE_3D_report.xlsx", summary, measurements,
                                  settings=cfg.to_flat_dict(), extra_sheets=extra_sheets)
    run_notes = {
        "tiled": {"tiles": run["tiles"], "tile_size": run["tile"], "halo": run["halo"],
                  "workers": run["workers"], "global_threshold": run["threshold"],
                  "clahe": "disabled in tiled mode", "tile_clipped_objects": run["clipped"],
                  "repeats_removed": run["repeats"], "source_notes": run["source_notes"]},
        "not_produced": ["arborization", "extra channels", "3D mesh", "GIF", "ImageJ ROIs"],
    }
    record = provenance.write_run_record(out, cfg, notes=run_notes)

    cleanup = CleanupReport(initial_count=len(measurements), final_count=len(measurements))
    cleanup.steps.append(f"tiled: {run['tiles']} tiles, global threshold {run['threshold']:.4g}")
    result = AnalysisResult(
        measurements=measurements, summary=summary,
        labels=tifffile.memmap(str(labels_path), mode="r"), prepared=None,  # type: ignore[arg-type]
        cleanup=cleanup, name=name, notes=list(run["source_notes"]) + _repeat_note(run),
    )
    result.outputs.update({"labels_ome_tiff": labels_path, "excel": out / "HIVE_3D_report.xlsx",
                           "run_record": record})
    return result


def analyze_image_tiled(
    path: str | Path, cfg: HiveConfig, progress: Callable[[float, str], None] | None = None,
):
    """Analyse one single-plane image too large for memory, tile by tile.

    The 2D counterpart of :func:`analyze_zstack_tiled`: the same global threshold, halo and
    ownership rules, with the 2D measurement table (area, perimeter, shape, intensity).
    Writes ``<name>_labels.ome.tif`` and ``<name>_measurements.csv`` into the output folder;
    arborization, the overlays and ImageJ ROIs need the whole image and are not produced.

    Args:
        path: A TIFF (any size; uncompressed, or compressed with ``zarr`` installed) or a
            single-plane file the normal reader can open.
        cfg: Run configuration.
        progress: Optional ``progress(fraction, message)`` callback.

    Returns:
        An :class:`~hive.pipeline.AnalysisResult` with ``prepared=None`` and a read-only
        memory map of the labels. Its ``notes`` say that tiles were used.
    """
    from .pipeline import AnalysisResult

    path = Path(path)
    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)
    labels_path = out / f"{path.stem}_labels.ome.tif"
    run = _run_tiled(cfg, str(path), True, out, labels_path, path.name, progress)
    measurements = run["measurements"]
    summary = summarize_measurements(measurements, path.name, cfg.voxel_unit)
    reporting.export_csv(out / f"{path.stem}_measurements.csv", measurements)
    cleanup = CleanupReport(initial_count=len(measurements), final_count=len(measurements))
    result = AnalysisResult(
        measurements=measurements, summary=summary,
        labels=tifffile.memmap(str(labels_path), mode="r"), prepared=None,  # type: ignore[arg-type]
        cleanup=cleanup, name=path.name,
    )
    result.outputs.update({"labels_ome_tiff": labels_path,
                           "measurements_csv": out / f"{path.stem}_measurements.csv"})
    result.notes.append(
        f"{path.name} is larger than the memory free now, so it was analysed in {run['tiles']} "
        f"tiles ({run['workers']} at a time): one threshold for the whole image, CLAHE off, and "
        "no branching/Sholl analysis, overlays or ImageJ ROIs. The label image is saved as an "
        "OME-TIFF."
        + (f" {run['clipped']} object(s) are cut at a tile edge (column tile_clipped)." if run["clipped"] else "")
    )
    result.notes.extend(_repeat_note(run))
    return result


def _create_label_file(path: Path, shape: tuple[int, ...], cfg: HiveConfig) -> np.ndarray:
    """Create the on-disk label image that tiles are painted into (``YX`` or ``ZYX``).

    The file is created at its full size but only the pages that hold objects are written, so
    on most file systems it takes far less disk space than its apparent size.
    """
    from .exports import _OME_UNITS

    metadata: dict = {"axes": "YX" if len(shape) == 2 else "ZYX"}
    unit = _OME_UNITS.get(cfg.voxel_unit.strip().lower())
    if unit:
        metadata.update(PhysicalSizeX=cfg.voxel_size_xy, PhysicalSizeY=cfg.voxel_size_xy,
                        PhysicalSizeXUnit=unit, PhysicalSizeYUnit=unit)
        if len(shape) == 3:
            metadata.update(PhysicalSizeZ=cfg.voxel_size_z, PhysicalSizeZUnit=unit)
    path.unlink(missing_ok=True)
    return tifffile.memmap(str(path), shape=shape, dtype=np.uint32, mode="w+", ome=True,
                           photometric="minisblack", metadata=metadata, bigtiff=True)


def _group_sheets(out: Path, summary: pd.DataFrame, measurements: pd.DataFrame,
                  cfg: HiveConfig) -> dict[str, pd.DataFrame]:
    """Group comparison sheets, when conditions were supplied."""
    if not cfg.conditions:
        return {}
    from .stats import write_group_outputs

    return {k: v for k, v in write_group_outputs(
        out, summary, measurements, cfg.conditions, cfg.seed).items() if not v.empty}
