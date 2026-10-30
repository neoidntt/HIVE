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
import warnings
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
    PIPELINE_MEMORY_FACTOR, ImageLoadError, _physical_memory_bytes, collect_image_paths,
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
        first = load_image(files[0])
        first = select_channel(first, channel)[0] if first.ndim == 3 else first  # type: ignore[arg-type]

        def read(z: int, ys: slice, xs: slice) -> np.ndarray:
            plane = load_image(files[z])
            if plane.ndim == 3:
                plane = select_channel(plane, channel)[0]  # type: ignore[arg-type]
            return plane[ys, xs]

        yield _Source((len(files), *first.shape), read, f"{len(files)} slices in {path.name}", [])
    else:
        raise FileNotFoundError(f"Z-stack path does not exist: {path}")


@contextmanager
def _tiff_source(path: Path, channel: str) -> Iterator[_Source]:
    """Reading side of :func:`open_source` for one TIFF file."""
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes, shape = series.axes.upper(), tuple(series.shape)
        if "S" in axes and shape[axes.index("S")] > 1:
            raise ImageLoadError(f"{path.name}: RGB stacks are not supported in tiled mode")
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

        try:
            mapped = tifffile.memmap(str(path), mode="r")   # free random access if uncompressed
            if tuple(mapped.shape) != shape:
                mapped = None
        except Exception:  # noqa: BLE001 - compressed or irregular: fall back to page decoding
            mapped = None
        if mapped is None:
            notes.append("File is compressed or not contiguous, so every plane is decoded "
                         "again for each tile. Expect this to be slow.")

        def read(z: int, ys: slice, xs: slice) -> np.ndarray:
            index = tuple(z if a == z_axis else (c_index if a == "C" else 0) for a in lead)
            if mapped is not None:
                return np.asarray(mapped[index + (ys, xs)], dtype=np.float32)
            page = int(np.ravel_multi_index(index, lead_shape)) if lead else 0
            return series.pages[page].asarray()[ys, xs].astype(np.float32)

        yield _Source((n_z, shape[axes.index("Y")], shape[axes.index("X")]), read,
                      f"{path.name} {shape} {axes}", notes)
        del mapped


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
    size, ram = stack_float32_bytes(cfg.input_path), _physical_memory_bytes()
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


def choose_tile_size(shape: tuple[int, int, int], halo: int) -> int:
    """Largest square tile core whose working set fits in about half of RAM."""
    z, y, x = shape
    ram = _physical_memory_bytes()
    if not ram:
        return min(1024, max(y, x))
    voxels = ram * 0.5 / (4 * PIPELINE_MEMORY_FACTOR)
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


def analyze_zstack_tiled(
    cfg: HiveConfig, progress: Callable[[float, str], None] | None = None,
):
    """Analyse a 3D stack tile by tile, writing tables and an OME-TIFF label image.

    Args:
        cfg: Run configuration; ``cfg.large_image`` sets tile size and halo.
        progress: Optional ``progress(fraction, message)`` callback.

    Returns:
        An :class:`~hive.pipeline.AnalysisResult` whose ``labels`` is a read-only
        memory map of the label image and whose ``prepared`` is ``None`` (the corrected
        volume is never held whole).

    Raises:
        FileNotFoundError: If the stack does not exist.
        ImageLoadError: If it is not a plain z-stack.
        ValueError: If segmentation is not the threshold method.
    """
    from . import provenance
    from .pipeline import AnalysisResult

    if cfg.segmentation.method != "threshold":
        raise ValueError("Tiled analysis supports threshold segmentation only.")
    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)
    pre = replace(cfg.preprocess, use_clahe=False)
    seg_cfg = replace(cfg.segmentation, clear_border=False)   # applied to the whole volume below
    report = progress or (lambda fraction, message: None)
    name = Path(cfg.input_path).name

    with open_source(cfg.input_path, cfg.channel) as src:
        z_size, y_size, x_size = src.shape
        halo = cfg.large_image.halo or filter_halo(pre)
        tile = cfg.large_image.tile_size or choose_tile_size(src.shape, halo)
        tiles = _grid(y_size, x_size, tile)
        logger.info("Tiled analysis of %s: %d tiles of %d px, halo %d", src.description,
                    len(tiles), tile, halo)

        def window(core):
            y0, y1, x0, x1 = core
            return (max(0, y0 - halo), min(y_size, y1 + halo),
                    max(0, x0 - halo), min(x_size, x1 + halo))

        # Pass 1: one threshold for the whole volume.
        rng = np.random.default_rng(cfg.seed)
        keep = min(1.0, _THRESHOLD_SAMPLE / (z_size * y_size * x_size))
        samples = []
        for i, core in enumerate(tiles):
            wy0, wy1, wx0, wx1 = window(core)
            _, smooth = _smoothed(src.window(slice(wy0, wy1), slice(wx0, wx1)), pre)
            piece = smooth[:, core[0] - wy0:core[1] - wy0, core[2] - wx0:core[3] - wx0].ravel()
            if keep < 1.0:
                piece = piece[rng.random(piece.size) < keep]
            samples.append(piece.copy())
            report(0.4 * (i + 1) / len(tiles), f"threshold pass, tile {i + 1}/{len(tiles)}")
        pooled = np.concatenate(samples)
        del samples
        if float(pooled.max()) <= float(pooled.min()):
            # As in the whole-volume run, a blank volume is a result (0 objects), not a crash.
            warnings.warn(
                "The volume is uniform after background subtraction (an empty channel?). "
                "Reporting 0 objects.", stacklevel=2)
            threshold = float("inf")
        else:
            threshold = float(filters.threshold_otsu(pooled)) * cfg.segmentation.threshold_scale
        del pooled

        # Pass 2: segment, own, measure, and paint the label mosaic.
        labels_path = out / "HIVE_3D_labels.ome.tif"
        mosaic = _create_label_file(labels_path, src.shape, cfg)
        tables, next_id, clipped_total = [], 1, 0
        for i, core in enumerate(tiles):
            wy0, wy1, wx0, wx1 = window(core)
            corrected, smooth = _smoothed(src.window(slice(wy0, wy1), slice(wx0, wx1)), pre)
            mask = threshold_segmentation(smooth, seg_cfg, threshold=threshold)
            del smooth
            labels = (watershed_split(mask) if cfg.segmentation.watershed_split and mask.any()
                      else label_mask(mask))
            labels, _ = clean_labels(labels, seg_cfg)
            table = measure_objects(
                labels, corrected, image_name=name, voxel_size_xy=cfg.voxel_size_xy,
                voxel_size_z=cfg.voxel_size_z, unit=cfg.voxel_unit)
            del corrected, mask
            if table.empty:
                report(0.4 + 0.6 * (i + 1) / len(tiles), f"tile {i + 1}/{len(tiles)}")
                continue

            cy, cx = table["centroid-1"] + wy0, table["centroid-2"] + wx0
            owned = (cy >= core[0]) & (cy < core[1]) & (cx >= core[2]) & (cx < core[3])
            h, w = labels.shape[1:]
            edge = (((table["bbox-1"] == 0) & (wy0 > 0)) | ((table["bbox-4"] == h) & (wy1 < y_size))
                    | ((table["bbox-2"] == 0) & (wx0 > 0)) | ((table["bbox-5"] == w) & (wx1 < x_size)))
            table["tile_clipped"] = edge
            for column, offset in (("centroid-1", wy0), ("centroid-2", wx0),
                                   ("bbox-1", wy0), ("bbox-4", wy0),
                                   ("bbox-2", wx0), ("bbox-5", wx0)):
                table[column] += offset
            if cfg.segmentation.clear_border:
                touches = ((table["bbox-0"] == 0) | (table["bbox-3"] == z_size)
                           | (table["bbox-1"] == 0) | (table["bbox-4"] == y_size)
                           | (table["bbox-2"] == 0) | (table["bbox-5"] == x_size))
                owned &= ~touches

            table = table[owned].copy()
            if not table.empty:
                new_ids = np.arange(next_id, next_id + len(table), dtype=np.uint32)
                lut = np.zeros(int(labels.max()) + 1, dtype=np.uint32)
                lut[table["label"].to_numpy()] = new_ids
                painted = lut[labels]
                region = mosaic[:, wy0:wy1, wx0:wx1]
                hit = painted > 0
                region[hit] = painted[hit]
                table["label"] = new_ids
                next_id += len(table)
                clipped_total += int(table["tile_clipped"].sum())
                tables.append(table)
            report(0.4 + 0.6 * (i + 1) / len(tiles), f"tile {i + 1}/{len(tiles)}")
        mosaic.flush()
        del mosaic
        notes = src.notes

    measurements = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    summary = summarize_measurements(measurements, name, cfg.voxel_unit)
    if clipped_total:
        logger.warning("%d object(s) reach beyond the %d px halo and may be cut at a tile "
                       "edge (column tile_clipped). Use a larger halo or exclude them.",
                       clipped_total, halo)

    reporting.export_csv(out / "HIVE_3D_measurements.csv", measurements)
    reporting.export_csv(out / "HIVE_3D_summary.csv", summary)
    extra_sheets = _group_sheets(out, summary, measurements, cfg)
    reporting.export_excel_report(out / "HIVE_3D_report.xlsx", summary, measurements,
                                  settings=cfg.to_flat_dict(), extra_sheets=extra_sheets)
    run_notes = {
        "tiled": {"tiles": len(tiles), "tile_size": tile, "halo": halo,
                  "global_threshold": threshold, "clahe": "disabled in tiled mode",
                  "tile_clipped_objects": clipped_total, "source_notes": notes},
        "not_produced": ["arborization", "extra channels", "3D mesh", "GIF", "ImageJ ROIs"],
    }
    record = provenance.write_run_record(out, cfg, notes=run_notes)

    cleanup = CleanupReport(initial_count=len(measurements), final_count=len(measurements))
    cleanup.steps.append(f"tiled: {len(tiles)} tiles, global threshold {threshold:.4g}")
    result = AnalysisResult(
        measurements=measurements, summary=summary,
        labels=tifffile.memmap(str(labels_path), mode="r"), prepared=None,  # type: ignore[arg-type]
        cleanup=cleanup, name=name,
    )
    result.outputs.update({"labels_ome_tiff": labels_path, "excel": out / "HIVE_3D_report.xlsx",
                           "run_record": record})
    return result


def _create_label_file(path: Path, shape: tuple[int, int, int], cfg: HiveConfig) -> np.ndarray:
    """Create the on-disk label image that tiles are painted into."""
    from .exports import _OME_UNITS

    metadata: dict = {"axes": "ZYX"}
    unit = _OME_UNITS.get(cfg.voxel_unit.strip().lower())
    if unit:
        metadata.update(PhysicalSizeX=cfg.voxel_size_xy, PhysicalSizeY=cfg.voxel_size_xy,
                        PhysicalSizeZ=cfg.voxel_size_z, PhysicalSizeXUnit=unit,
                        PhysicalSizeYUnit=unit, PhysicalSizeZUnit=unit)
    path.unlink(missing_ok=True)
    return tifffile.memmap(str(path), shape=shape, dtype=np.uint32, mode="w+", ome=True,
                           photometric="minisblack", metadata=metadata)


def _group_sheets(out: Path, summary: pd.DataFrame, measurements: pd.DataFrame,
                  cfg: HiveConfig) -> dict[str, pd.DataFrame]:
    """Group comparison sheets, when conditions were supplied."""
    if not cfg.conditions:
        return {}
    from .stats import write_group_outputs

    return {k: v for k, v in write_group_outputs(
        out, summary, measurements, cfg.conditions, cfg.seed).items() if not v.empty}
