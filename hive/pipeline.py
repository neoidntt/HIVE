"""End-to-end 2D and 3D analysis pipelines.

Both pipelines return an :class:`AnalysisResult` rather than a bare tuple. The
legacy ``analyze_zstack`` returned ``(measurements, summary)``, and the Streamlit
app bound that tuple to a single name and handed it to ``st.dataframe`` -- an
error a named result object cannot express.
"""

from __future__ import annotations

import copy
import gc
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import channels, exports, formats, io_utils, provenance, reporting, stats
from .arborization import analyze_arbor, skeleton_metrics
from .cleanup import CleanupReport, clean_labels
from .config import HiveConfig
from .io_utils import (
    ImageLoadError, _available_memory_bytes, _fmt_bytes, collect_image_paths, load_image_native, load_zstack,
    save_uint8, save_volume_tiff, to_uint8, write_png,
)
from .measurements import measure_objects, summarize_measurements
from .preprocessing import PreparedImage, prepare_image, prepare_volume
from .segmentation import segment
from .tiled import analyze_image_tiled, analyze_zstack_tiled, needs_tiling
from .visualization import (
    create_3d_mesh_html, create_boundary_overlay, create_skeleton_overlay, export_obj,
    label_overlay_f32,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Everything one analysis produced.

    Attributes:
        measurements: Per-object table (morphology + arborisation).
        summary: One-row per-image summary.
        labels: Final cleaned label array.
        prepared: The measurement/segmentation image pair (``None`` for tiled runs,
            which never hold the whole corrected volume).
        sholl_curves: Mapping of label -> ``ShollResult``.
        cleanup: What mask cleanup removed.
        skeleton: Combined skeleton of all objects, for overlays.
        name: Source image name.
        outputs: Artifact paths written, keyed by kind.
        notes: Things the user should know about this analysis (for example that CLAHE was
            switched off to fit in memory).
    """

    measurements: pd.DataFrame
    summary: pd.DataFrame
    labels: np.ndarray = field(repr=False)
    prepared: PreparedImage | None = field(repr=False)
    sholl_curves: dict = field(default_factory=dict, repr=False)
    cleanup: CleanupReport = field(default_factory=CleanupReport)
    skeleton: np.ndarray | None = field(default=None, repr=False)
    name: str = ""
    outputs: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def object_count(self) -> int:
        """int: Number of objects measured."""
        return 0 if self.measurements is None or self.measurements.empty else len(self.measurements)


def auto_channel_note(cfg: HiveConfig, name: str, chosen: str) -> str | None:
    """The note that says which channel ``"auto"`` picked (``None`` if a channel was named)."""
    if str(cfg.channel).lower() != "auto":
        return None
    return (f"{name}: the channel was set to auto, so HIVE analysed the {chosen} channel (the one "
            "with the most contrast). To use a different one, choose it by name in the Channel "
            "box (or --channel on the command line).")


def _spacing(cfg: HiveConfig, is_3d: bool) -> tuple[float, ...] | float:
    """Build a per-axis spacing tuple from the config.

    Args:
        cfg: Run configuration.
        is_3d: Whether the data is a volume.

    Returns:
        ``(z, y, x)`` spacing for 3D, or the scalar xy size for 2D.
    """
    if is_3d:
        return (cfg.voxel_size_z, cfg.voxel_size_xy, cfg.voxel_size_xy)
    return cfg.voxel_size_xy


def _new_skeleton_buffer(labels: np.ndarray, cfg: HiveConfig) -> np.ndarray | None:
    """An empty array for the combined skeleton, or ``None`` to skeletonise the mask instead.

    Every label is one connected object, so the union of the objects' own skeletons is the
    skeleton of the union of the masks, and the per-object pass already computes them.
    With ``watershed_split`` touching objects share one mask, so the whole mask is
    skeletonised as before to keep that overlay unchanged.
    """
    return None if cfg.segmentation.watershed_split else np.zeros(labels.shape, dtype=bool)


def analyze_array(
    image: np.ndarray, name: str, cfg: HiveConfig,
    extras: dict[str, np.ndarray] | None = None, channel_name: str | None = None,
) -> AnalysisResult:
    """Run the full 2D pipeline on an in-memory image.

    Steps: channel select -> baseline subtract (measurement image) -> enhance
    (segmentation image) -> segment -> clean -> measure -> arborise.

    Exists so a caller holding a plane already in memory -- e.g. one page of a
    huge multi-page stack -- can analyse it without first writing it to disk.
    :func:`analyze_image` is a thin wrapper that loads a file and calls this.

    Args:
        image: Raw 2D or channel-last image, in its original intensity scale.
        name: Name recorded in the ``image`` column.
        cfg: Run configuration.
        extras: Further channels to measure, by name. ``None`` takes
            ``cfg.extra_channels`` from ``image`` when it is a colour image.
        channel_name: Name to record for the analysed channel when ``image`` is
            already a single plane taken from a named channel.

    Returns:
        An :class:`AnalysisResult`.

    Raises:
        SegmentationError: If the selected channel is empty.
        ValueError: If the image shape is unsupported.
    """
    prepared = prepare_image(image, cfg.preprocess, cfg.channel)
    if channel_name:
        prepared.channel_name = channel_name

    labels = segment(prepared.segmentation, cfg.segmentation)
    labels, cleanup = clean_labels(labels, cfg.segmentation)

    measurements = measure_objects(
        labels, prepared.measurement, image_name=name,
        voxel_size_xy=cfg.voxel_size_xy, unit=cfg.voxel_unit,
    )

    sholl_curves: dict = {}
    skeleton = None
    if cfg.arbor.enabled and labels.max() > 0:
        skeleton = _new_skeleton_buffer(labels, cfg)
        arbor_df, sholl_curves = analyze_arbor(
            labels, cfg.arbor, intensity=prepared.measurement,
            spacing=_spacing(cfg, is_3d=False), unit=cfg.voxel_unit, skeleton_out=skeleton,
        )
        measurements = reporting.consolidate(measurements, arbor_df, name)
        if skeleton is None:
            skeleton = skeleton_metrics(
                labels > 0, cfg.arbor.prune_length, _spacing(cfg, is_3d=False)
            ).skeleton

    summary = summarize_measurements(measurements, name, cfg.voxel_unit)
    if extras is None and cfg.extra_channels:
        extras = channels.rgb_extras(image, cfg.extra_channels)
    measurements, summary = channels.add_channel_measurements(
        measurements, summary, labels, prepared.measurement, extras or {}, cfg,
    )
    logger.info("%s: %s", name, cleanup.summary())

    notes = []
    if channel_name is None and np.ndim(image) == 3 and np.shape(image)[-1] >= 3:
        notes = [n for n in [auto_channel_note(cfg, name, prepared.channel_name)] if n]
    return AnalysisResult(
        measurements=measurements, summary=summary, labels=labels, prepared=prepared,
        sholl_curves=sholl_curves, cleanup=cleanup, skeleton=skeleton, name=name, notes=notes,
    )


def _channel_choice(meta: formats.MicroscopeImage, cfg: HiveConfig) -> int:
    """Index of the channel to analyse in a file with named channels.

    ``"auto"`` takes the channel with the highest standard deviation of its maximum
    projection -- a contrast heuristic, as for colour images; the resolved name is
    recorded in the results.
    """
    if meta.n_channels == 1:
        return 0
    if cfg.channel.lower() == "auto":
        spread = [float(np.std(meta.volume(i).max(axis=0))) for i in range(meta.n_channels)]
        return int(np.argmax(spread))
    return meta.channel_index(cfg.channel)


def _scaled(cfg: HiveConfig, meta: formats.MicroscopeImage) -> HiveConfig:
    """A copy of ``cfg`` using the file's voxel size where the user set none."""
    from dataclasses import replace

    scaled = replace(cfg)
    formats.apply_metadata(scaled, meta)
    return scaled


def _needs_tiled_2d(path: Path, cfg: HiveConfig) -> bool:
    """Whether this single image should be analysed in tiles because it will not fit in memory.

    Only TIFF files can be read in pieces. ``cfg.large_image.mode`` of ``"off"`` never tiles
    (the memory check then refuses an image that is too large) and ``"force"`` always does.
    """
    mode = cfg.large_image.mode
    if mode == "off" or path.suffix.lower() not in (".tif", ".tiff"):
        return False
    if mode == "force":
        return True
    free = _available_memory_bytes()
    need = _peak_bytes_per_image(path, False)
    return bool(free and need and need > free * 0.85)


def _check_vendor_file_fits(meta_path: Path) -> None:
    """Refuse a CZI / ND2 / LIF (or OME-TIFF) file that cannot be loaded whole.

    These readers decode the whole file, so a file bigger than memory would be killed by the
    operating system. The header gives the shape without decoding; two bytes per value is
    assumed (16-bit data), so this errs on the side of letting a file through.
    """
    shape = formats.read_microscope(meta_path, load_pixels=False).shape
    need = int(np.prod(shape)) * 2
    free = _available_memory_bytes()
    if free and need > free * 0.5:
        raise ImageTooLargeError(
            f"{meta_path.name} holds about {_fmt_bytes(need)} of pixel data but only "
            f"{_fmt_bytes(free)} of memory is free, and this format can only be read whole. "
            "Export it as an (OME-)TIFF (for example from Fiji or the microscope software); "
            "HIVE reads those in pieces, so there is no size limit.")


def analyze_image(image_path: str | Path, cfg: HiveConfig,
                  progress=None) -> AnalysisResult:
    """Run the full 2D pipeline on a single image file.

    OME-TIFF, CZI, ND2 and LIF files are read through :mod:`hive.formats`, so channels
    can be chosen by name and the pixel size is taken from the file (unless
    ``cfg`` already sets one).

    A TIFF too large for memory is analysed in tiles instead (see
    :func:`hive.tiled.analyze_image_tiled`), so there is no limit on its size.

    Args:
        image_path: Image file to analyse.
        cfg: Run configuration.
        progress: Optional ``progress(fraction, message)`` callback, used for tiled runs.

    Returns:
        An :class:`AnalysisResult`.

    Raises:
        FileNotFoundError: If the image does not exist.
        ImageLoadError: If the image cannot be decoded.
        ImageTooLargeError: If it cannot be read in pieces and does not fit in memory.
        SegmentationError: If the selected channel is empty.
    """
    image_path = Path(image_path)
    axis = _tiff_volume_axis(image_path)
    if axis:
        raise ImageLoadError(
            f"{image_path.name} is a {'z-stack' if axis == 'Z' else 'time series'} (many planes), "
            "not a single 2D image, so the 2D analysis would give meaningless numbers. "
            "Switch Mode to 3D and open it there.")
    if _needs_tiled_2d(image_path, cfg):
        return analyze_image_tiled(image_path, cfg, progress)
    cfg, note = _fit_memory(image_path, cfg)
    if formats.is_microscope_file(image_path):
        _check_vendor_file_fits(image_path)
        meta = formats.read_microscope(image_path)
        index = _channel_choice(meta, cfg)
        extras = {name: meta.plane(name) for name in cfg.extra_channels}
        result = analyze_array(
            meta.plane(index), image_path.name, _scaled(cfg, meta),
            extras=extras, channel_name=meta.channel_names[index],
        )
        if meta.n_channels > 1:
            chosen = auto_channel_note(cfg, image_path.name, meta.channel_names[index])
            if chosen:
                result.notes.append(chosen)
    else:
        # Native dtype: a 66-megapixel RGB JPEG is 0.2 GB as stored and 0.8 GB as float32;
        # channel selection converts only the plane it needs.
        result = analyze_array(load_image_native(image_path), image_path.name, cfg)
    if note:
        result.notes.append(note)
    return result


def _analyze_plane(job: tuple[str, int, HiveConfig]) -> tuple:
    """Analyse one page of a multi-page stack, reading only that page.

    Module-level so it can be pickled to a worker. Each worker opens its own file
    handle -- a TiffFile cannot be shared across processes.

    Args:
        job: Tuple of (stack path, page index, config).

    Returns:
        Tuple of (measurements, summary, sholl, error), as for :func:`_analyze_one`.
    """
    import tifffile

    path, z, cfg = job
    name = f"z{z:03d}"
    try:
        with tifffile.TiffFile(path) as tif:
            plane = tif.series[0].pages[z].asarray()
        result = analyze_array(plane.astype(np.float32), name, cfg)
        del plane
        sholl = (
            reporting.sholl_curves_to_frame(result.sholl_curves, name)
            if result.sholl_curves else None
        )
        return result.measurements, result.summary, sholl, None
    except Exception as exc:  # noqa: BLE001 - one bad plane must not kill the run
        return None, None, None, (name, str(exc))


def analyze_stack_planes(stack_path: str | Path, cfg: HiveConfig, workers: int = 1,
                         planes: list[int] | None = None) -> dict[str, object]:
    """Run the 2D pipeline on each plane of a multi-page stack, page by page.

    Analyses a huge stack **without extracting it to disk first**. A 38 GB
    109-plane stack would otherwise need 38 GB of intermediate files; here each
    worker decodes one page (a few hundred MB), analyses it, and returns only its
    tables.

    This is 2D analysis repeated per plane -- it does NOT do 3D. Objects spanning
    planes are counted once per plane they appear in. For true 3D use
    :func:`analyze_zstack` on a volume that fits in memory.

    Args:
        stack_path: Multi-page TIFF.
        cfg: Run configuration. ``cfg.input_path`` is ignored.
        workers: Parallel worker processes. Each needs its own per-plane working
            set; see :func:`hive.io_utils.memory_report`.
        planes: Specific page indices, e.g. ``[50, 51, 52]`` for a quick check.
            ``None`` does all of them.

    Returns:
        Dict with ``measurements``, ``summary``, ``sholl`` and ``errors``.

    Raises:
        ValueError: If ``workers`` < 1 or a plane index is out of range.
        FileNotFoundError: If the stack does not exist.
    """
    import tifffile

    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    stack_path = Path(stack_path)
    if not stack_path.exists():
        raise FileNotFoundError(f"Stack not found: {stack_path}")

    with tifffile.TiffFile(str(stack_path)) as tif:
        n_pages = len(tif.series[0].pages)
    todo = list(range(n_pages)) if planes is None else list(planes)
    bad = [z for z in todo if not 0 <= z < n_pages]
    if bad:
        raise ValueError(f"plane index out of range {bad}; the stack has {n_pages} pages")

    Path(cfg.output_folder).mkdir(parents=True, exist_ok=True)
    jobs = [(str(stack_path), z, cfg) for z in todo]
    workers = min(workers, len(jobs))

    if workers > 1:
        logger.info("Analysing %d planes on %d workers", len(jobs), workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_analyze_plane, jobs))
    else:
        results = [_analyze_plane(job) for job in jobs]

    measurements, summaries, sholls, errors = [], [], [], []
    for m, s, sh, err in results:
        if err is not None:
            logger.error("Skipping %s: %s", *err)
            errors.append(err)
            continue
        if m is not None and not m.empty:
            measurements.append(m)
        if s is not None:
            summaries.append(s)
        if sh is not None and not sh.empty:
            sholls.append(sh)

    out = Path(cfg.output_folder)
    df = pd.concat(measurements, ignore_index=True) if measurements else pd.DataFrame()
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    sholl = pd.concat(sholls, ignore_index=True) if sholls else pd.DataFrame()

    reporting.export_csv(out / "combined_measurements.csv", df)
    reporting.export_csv(out / "combined_summary.csv", summary)
    if not sholl.empty:
        reporting.export_csv(out / "combined_sholl_curves.csv", sholl)
    reporting.export_excel_report(out / "HIVE_2D_report.xlsx", summary, df,
                                  settings=cfg.to_flat_dict(), sholl_curves=sholl)
    provenance.write_run_record(out, cfg, notes={"planes": todo})
    return {"measurements": df, "summary": summary, "sholl": sholl, "errors": errors}


def _write_2d_outputs(result: AnalysisResult, cfg: HiveConfig, stem: str) -> None:
    """Write per-image QC images and tables.

    Args:
        result: The analysis result; its ``outputs`` map is populated in place.
        cfg: Run configuration.
        stem: Filename stem for the artifacts.
    """
    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)

    # Pictures are made a few at a time, never all first: building every float RGB overlay
    # up front held several at once, which for a 66-megapixel image is more than 8 GB and got
    # the process killed.
    measurement = result.prepared.measurement
    waiting: deque = deque()

    # Each picture is built, turned into 8 bit and encoded inside one worker thread (numpy and
    # zlib both release the interpreter lock), so several are made at once. A picture in flight
    # holds a float RGB overlay (12 bytes a pixel) and its 8-bit copy, so the number in flight
    # is limited by the memory that is free, down to one at a time for a huge image.
    pixels = int(np.prod(result.labels.shape[-2:]))
    free = io_utils._available_memory_bytes()
    in_flight = 3 if not free else max(1, min(3, int(free * 0.3 // (pixels * 16))))

    def add(name: str, build, rescale: bool = True) -> None:
        def make() -> None:
            write_png(out / name, to_uint8(build(), rescale, what=name))

        waiting.append(writers.submit(make))
        while len(waiting) >= in_flight:
            waiting.popleft().result()

    with ThreadPoolExecutor(max_workers=in_flight) as writers, ThreadPoolExecutor(max_workers=1) as background:
        # The OME-TIFF / ROI export runs while the pictures are being made and encoded.
        label_files = background.submit(exports.export_label_files, out, stem, result.labels, cfg)
        add(f"{stem}_mask.png", lambda: (result.labels > 0).astype(np.float32))
        add(f"{stem}_boundary_overlay.png", lambda: create_boundary_overlay(measurement, result.labels))
        add(f"{stem}_label_overlay.png", lambda: label_overlay_f32(measurement, result.labels))
        result.outputs["mask"] = out / f"{stem}_mask.png"
        result.outputs["boundary_overlay"] = out / f"{stem}_boundary_overlay.png"

        if result.skeleton is not None and result.skeleton.any():
            add(f"{stem}_skeleton.png", lambda: result.skeleton.astype(np.float32))
            add(f"{stem}_skeleton_overlay.png",
                lambda: create_skeleton_overlay(measurement, result.skeleton, result.labels > 0))
            result.outputs["skeleton_overlay"] = out / f"{stem}_skeleton_overlay.png"

        if cfg.save_intermediate:
            add(f"{stem}_measurement.png", lambda: measurement)
            add(f"{stem}_segmentation.png", lambda: result.prepared.segmentation, rescale=False)
        for pending in waiting:
            pending.result()

        reporting.export_csv(out / f"{stem}_measurements.csv", result.measurements)
        result.outputs["measurements_csv"] = out / f"{stem}_measurements.csv"
        result.outputs.update(label_files.result())


def _analyze_one(job: tuple) -> tuple:
    """Analyse one image and return only its tables.

    Module-level (not a closure) so it can be pickled to a worker process.

    Crucially this returns **DataFrames, not arrays**. An
    :class:`AnalysisResult` carries the measurement volume, segmentation volume
    and label array; pickling those back to the parent would send gigabytes
    through a pipe per image and defeat the point of parallelising.

    Args:
        job: Tuple of (image path, config, whether to write QC artifacts[, progress callback]).
            The callback is only passed for images analysed in tiles, which run in the main
            process.

    Returns:
        Tuple of (measurements, summary, sholl, error, notes) where ``error`` is
        ``None`` on success and ``(name, message)`` on failure, and ``notes`` lists anything
        the user should know about this image.
    """
    path, cfg, write_per_image, *rest = job
    try:
        result = analyze_image(path, cfg, rest[0] if rest else None)
        if write_per_image and result.prepared is not None:     # tiled runs wrote their own
            _write_2d_outputs(result, cfg, path.stem)
        sholl = (
            reporting.sholl_curves_to_frame(result.sholl_curves, path.name)
            if result.sholl_curves else None
        )
        return result.measurements, result.summary, sholl, None, result.notes
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
        return None, None, None, (path.name, str(exc)), []


class ImageTooLargeError(MemoryError):
    """An image needs more memory than this computer has free right now.

    Raised *before* any work starts. Without it the analysis ran until the operating system
    killed the whole process, which in the web app shows up only as "Connection error".
    """


def _tiff_series_info(path: Path):
    """``(axes, itemsize)`` of a TIFF's first series from its header, or ``None``."""
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            return series.axes.upper(), np.dtype(series.dtype).itemsize, tuple(series.shape)
    except Exception:  # noqa: BLE001 - unreadable here: the real loader will say why
        return None


def _tiff_itemsize(path: Path) -> int:
    info = _tiff_series_info(path)
    return info[1] if info else 2


def _tiff_volume_axis(path: Path) -> str | None:
    """``"Z"`` or ``"T"`` when the TIFF is a z-stack or time series (more than one plane)."""
    info = _tiff_series_info(path)
    if not info:
        return None
    axes, _, shape = info
    for axis in ("Z", "T"):
        if axis in axes and shape[axes.index(axis)] > 1:
            return axis
    return None


def _peak_bytes_per_image(path: Path, use_clahe: bool = True) -> int | None:
    """Estimated peak memory of analysing one image, from its header (nothing is decoded).

    The analysis works on one channel at a time, so the estimate is per pixel, not per colour
    channel. Measured on a 66-megapixel RGB photo: about 4.0 GB with CLAHE (CLAHE alone is
    ~14x the one-channel float32 image, taken here as 18x to stay safe) and about 2.5 GB
    without it (taken as 10x, the figure used for volumes).

    Returns:
        Bytes, or ``None`` when the size cannot be read cheaply (CZI, ND2, LIF).
    """
    from .io_utils import PIPELINE_MEMORY_FACTOR, _peek_stack_bytes

    loaded = 0
    peek = _peek_stack_bytes(path)
    if peek is not None:
        shape = peek[1]
        if _tiff_volume_axis(path):
            return None  # a z-stack / time series: refused in 2D mode before anything is read
        pixels = int(np.prod(shape[-2:])) if len(shape) >= 2 else int(np.prod(shape))
        # The whole file is read before one channel is picked, at the dtype it is stored in.
        loaded = int(np.prod(shape)) * _tiff_itemsize(path)
    else:
        try:
            from PIL import Image

            with Image.open(path) as image:
                pixels = image.width * image.height
                shape = (image.height, image.width)
        except Exception:  # noqa: BLE001 - unreadable here: the real loader will say why
            return None
    big_2d = pixels > 50_000_000
    factor = 18.0 if (use_clahe and big_2d) else PIPELINE_MEMORY_FACTOR
    return int(loaded + pixels * 4 * factor)


def _fit_memory(path: Path, cfg: HiveConfig) -> tuple[HiveConfig, str | None]:
    """Make sure one image fits in the memory that is free now.

    * Fits: nothing changes.
    * Fits only without CLAHE (the biggest single cost for a large image): CLAHE is switched
      off for this image and a note says so. CLAHE only shapes the image used to *find*
      objects; measurements are never taken from it.
    * Does not fit either way: :class:`ImageTooLargeError` with what to do about it.

    Returns:
        The config to use for this image and an optional note for the user.
    """
    free = _available_memory_bytes()
    need = _peak_bytes_per_image(path, cfg.preprocess.use_clahe)
    if not free or need is None or need <= free * 0.85:
        return cfg, None
    if cfg.preprocess.use_clahe:
        lean = _peak_bytes_per_image(path, False)
        if lean is not None and lean <= free * 0.85:
            note = (f"CLAHE was switched off for {path.name}: with it the analysis needs about "
                    f"{_fmt_bytes(need)} and only {_fmt_bytes(free)} is free. Objects are found "
                    "on a slightly different enhanced image; measurements are not affected. "
                    "Close other programs to keep CLAHE on.")
            logger.warning(note)
            return replace(cfg, preprocess=replace(cfg.preprocess, use_clahe=False)), note
    raise ImageTooLargeError(
        f"{path.name} needs about {_fmt_bytes(need)} of memory to analyse but only "
        f"{_fmt_bytes(free)} is free. Close other programs (a web browser can use several "
        "GB), or crop / downsample the image, then try again."
    )


def auto_workers(paths: list[Path], use_clahe: bool = True) -> int:
    """How many images to analyse at once on this computer.

    As many as there are CPU cores, but never more than there are images and never more
    than fit in memory (each worker holds its own working set, and swapping is far
    slower than running fewer workers).

    Args:
        paths: The images of the batch.
        use_clahe: Whether CLAHE will run (it raises the memory estimate for huge images).

    Returns:
        A worker count of at least 1.
    """
    workers = min(os.cpu_count() or 1, len(paths))
    ram = _available_memory_bytes()
    peaks = [p for p in (_peak_bytes_per_image(path, use_clahe) for path in paths[:50]) if p]
    if ram and peaks:
        workers = min(workers, int(ram * 0.85 // max(peaks)))
    return max(1, workers)


def _warn_on_worker_memory(workers: int, paths: list[Path], use_clahe: bool = True) -> None:
    """Warn when the requested worker count cannot fit in RAM.

    Each worker holds its own full copy of the pipeline's working set, so peak
    memory is roughly ``workers x per-image peak``. Oversubscribing does not
    fail fast -- it swaps, and on macOS that means the machine crawls for hours.

    Args:
        workers: Requested worker count.
        paths: Images to be processed (the largest sets the per-worker cost).
        use_clahe: Whether CLAHE will run.
    """
    total = _available_memory_bytes()
    if not total or workers <= 1 or not paths:
        return
    peaks = [p for p in (_peak_bytes_per_image(path, use_clahe) for path in paths[:50]) if p]
    if not peaks:
        return
    peak = max(peaks)
    estimated = peak * workers
    if estimated > total * 0.85:
        safe = max(1, int(total * 0.85 / peak))
        warnings.warn(
            f"{workers} workers x ~{peak / 1e9:.1f} GB each "
            f"= ~{estimated / 1e9:.0f} GB, but only {total / 1e9:.0f} GB is free. "
            f"This will swap. Use workers={safe} or fewer. "
            "(Setting use_clahe=False typically cuts per-worker memory ~3x.)",
            stacklevel=3,
        )


def analyze_folder(cfg: HiveConfig, write_per_image: bool = True,
                   workers: int | None = 1, progress=None) -> dict[str, object]:
    """Run the 2D pipeline over every image in a folder.

    Images are processed one at a time and only their tables are retained, so
    peak memory tracks the largest single image rather than the whole batch.

    A failure on one image is logged and skipped rather than aborting the batch:
    losing 200 good results to one corrupt file is the worse outcome. Failures
    are returned in ``errors`` so they cannot pass unnoticed.

    Args:
        cfg: Run configuration.
        write_per_image: Write per-image QC artifacts.
        workers: Parallel worker processes. ``1`` runs in-process. Images are
            independent, so this scales nearly linearly -- but each worker needs
            its OWN copy of the working set, so ``workers x per-image peak`` must
            fit in RAM. A warning fires if it will not.
        progress: Optional ``progress(fraction, message)`` callback, called while an image too
            large for memory is analysed in tiles.

    Returns:
        Dict with ``measurements``, ``summary``, ``sholl``, ``errors`` and
        ``outputs``.

    Raises:
        FileNotFoundError: If the input folder holds no readable images.
        ValueError: If ``workers`` is below 1.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    input_folder = Path(cfg.input_path)
    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)

    # One image file is accepted as well as a folder, so a huge file can be analysed where it
    # is, with no copy into a folder first.
    if input_folder.is_file() and input_folder.suffix.lower() in io_utils.VALID_EXTENSIONS:
        paths = [input_folder]
    else:
        paths = collect_image_paths(input_folder)
    if not paths:
        raise FileNotFoundError(f"No readable images found in: {input_folder}")

    all_measurements: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_sholl: list[pd.DataFrame] = []
    errors: list[tuple[str, str]] = []

    # Images too big for memory are analysed in tiles, one at a time in this process (the tile
    # engine uses all cores itself); the others go to the worker pool as before.
    big = [p for p in paths if _needs_tiled_2d(p, cfg)]
    regular = [p for p in paths if p not in big]
    if workers is None:
        workers = auto_workers(regular or paths, cfg.preprocess.use_clahe)
        logger.info("Using %d worker(s) for %d image(s)", workers, len(regular))
    workers = max(1, min(workers, len(regular))) if regular else 1
    _warn_on_worker_memory(workers, regular, cfg.preprocess.use_clahe)
    jobs = [(p, cfg, write_per_image) for p in regular]

    if workers > 1:
        logger.info("Analysing %d images on %d workers", len(regular), workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = dict(zip(regular, pool.map(_analyze_one, jobs)))
    else:
        done = {p: _analyze_one(job) for p, job in zip(regular, jobs)}
    for p in big:
        done[p] = _analyze_one((p, cfg, write_per_image, progress))
    results = [done[p] for p in paths]

    notes: list[str] = []
    for measurements, summary, sholl, error, image_notes in results:
        notes.extend(image_notes)
        if error is not None:
            logger.error("Skipping %s: %s", *error)
            errors.append(error)
            continue
        if measurements is not None and not measurements.empty:
            all_measurements.append(measurements)
        if summary is not None:
            all_summaries.append(summary)
        if sholl is not None and not sholl.empty:
            all_sholl.append(sholl)

    measurements = (
        pd.concat(all_measurements, ignore_index=True) if all_measurements else pd.DataFrame()
    )
    summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    sholl = pd.concat(all_sholl, ignore_index=True) if all_sholl else pd.DataFrame()

    reporting.export_csv(out / "combined_measurements.csv", measurements)
    reporting.export_csv(out / "combined_summary.csv", summary)
    if not sholl.empty:
        reporting.export_csv(out / "combined_sholl_curves.csv", sholl)
    group_sheets = (
        stats.write_group_outputs(out, summary, measurements, cfg.conditions, cfg.seed)
        if cfg.conditions else {}
    )
    if notes:
        group_sheets = {**group_sheets, "Notes": pd.DataFrame({"note": notes})}
    reporting.export_excel_report(
        out / "HIVE_2D_report.xlsx", summary, measurements,
        settings=cfg.to_flat_dict(), sholl_curves=sholl, extra_sheets=group_sheets,
    )
    run_record = provenance.write_run_record(out, cfg, notes={"notes": notes} if notes else None)

    if errors:
        logger.warning("%d of %d images failed", len(errors), len(paths))

    return {
        "measurements": measurements,
        "summary": summary,
        "sholl": sholl,
        "errors": errors,
        "notes": notes,
        "group_stats": group_sheets.get("Group_Stats"),
        "outputs": {"excel": out / "HIVE_2D_report.xlsx",
                    "measurements_csv": out / "combined_measurements.csv",
                    "run_record": run_record},
    }


def analyze_stacks(cfg: HiveConfig, progress=None) -> dict:
    """Analyse every file in a folder as its own 3D stack, one after another.

    Use this when each file is a whole stack (a folder of slices, one plane per file, is
    what :func:`analyze_zstack` takes). The stacks run one at a time on purpose: each one
    already uses every core, and running two at once would double the memory. A stack that
    fails is reported and the rest carry on.

    Each stack's results go to ``<output_folder>/<file name without extension>/``, exactly as
    if it had been analysed alone; the tables of all stacks are also combined in
    ``combined_3d_measurements.csv`` and ``combined_3d_summary.csv`` in the output folder.

    Args:
        cfg: Run configuration; ``cfg.input_path`` is the folder and ``cfg.output_folder``
            the base results folder.
        progress: Optional ``progress(fraction, message)`` callback.

    Returns:
        Dict with ``measurements``, ``summary``, ``errors`` (``(file, message)`` pairs),
        ``stacks`` (one dict per finished stack: ``name``, ``folder``, ``objects``,
        ``labels_path``, ``tiled``) and ``outputs``.

    Raises:
        FileNotFoundError: If the folder holds no readable files.
    """
    folder = Path(cfg.input_path)
    files = collect_image_paths(folder) if folder.is_dir() else [folder]
    if not files:
        raise FileNotFoundError(f"No readable image files found in: {folder}")
    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda fraction, message: None)

    tables, summaries, stacks, errors, notes = [], [], [], [], []
    for index, file in enumerate(files):
        sub = copy.deepcopy(cfg)
        sub.input_path = str(file)
        sub.output_folder = str(out / file.stem)

        def inner(fraction, message, index=index, name=file.name):
            report((index + float(fraction)) / len(files), f"{name} ({index + 1}/{len(files)}): {message}")

        report(index / len(files), f"{file.name} ({index + 1}/{len(files)})")
        try:
            result = analyze_zstack(sub, progress=inner)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the others
            logger.error("Skipping %s: %s", file.name, exc)
            errors.append((file.name, str(exc)))
            gc.collect()
            continue
        tables.append(result.measurements)
        summaries.append(result.summary)
        notes.extend(result.notes)
        stacks.append({
            "name": file.name, "folder": str(sub.output_folder), "objects": result.object_count,
            "labels_path": str(result.outputs["labels_ome_tiff"]) if "labels_ome_tiff" in result.outputs else None,
            "tiled": result.prepared is None,
        })
        del result
        gc.collect()          # release this stack's volumes before the next one is read
    report(1.0, "done")

    measurements = pd.concat([t for t in tables if t is not None and not t.empty],
                             ignore_index=True) if any(not t.empty for t in tables) else pd.DataFrame()
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    reporting.export_csv(out / "combined_3d_measurements.csv", measurements)
    reporting.export_csv(out / "combined_3d_summary.csv", summary)
    return {"measurements": measurements, "summary": summary, "errors": errors, "stacks": stacks,
            "notes": notes, "outputs": {"measurements_csv": out / "combined_3d_measurements.csv",
                        "summary_csv": out / "combined_3d_summary.csv"}}


def analyze_zstack(cfg: HiveConfig, render_gif: bool = False, progress=None) -> AnalysisResult:
    """Run the full 3D pipeline on a z-stack.

    The channel axis is reduced **before** baseline subtraction, which is the
    fix for the legacy crash: the old order handed a 4D ``(Z, Y, X, C)`` array
    to a function accepting only 2D/3D, so every RGB z-stack raised
    ``ValueError`` on the README's documented ``--mode 3d --channel magenta``
    workflow.

    Args:
        cfg: Run configuration. ``cfg.input_path`` is the stack file or folder.
        render_gif: Also render the rotating projection GIF.
        progress: Optional ``progress(fraction, message)`` callback for tiled runs, which
            are the ones that take long.

    Returns:
        An :class:`AnalysisResult` whose arrays are volumes.

    Raises:
        FileNotFoundError: If the stack path does not exist.
        ValueError: If the stack shape is unsupported or slices mismatch.
        SegmentationError: If the selected channel is empty.
    """
    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)

    if needs_tiling(cfg):
        logger.warning("Analysing in tiles (large_image.mode=%s): CLAHE is off, and "
                       "arborization and extra channels are skipped; see hive.tiled.",
                       cfg.large_image.mode)
        return analyze_zstack_tiled(cfg, progress)

    user_cfg = cfg
    extras: dict[str, np.ndarray] = {}
    channel_label = None
    multichannel = False
    source = Path(cfg.input_path)
    if source.is_file() and formats.is_microscope_file(source):
        _check_vendor_file_fits(source)
        meta = formats.read_microscope(source)
        index = _channel_choice(meta, cfg)
        raw = meta.volume(index)
        extras = {name: meta.volume(name) for name in cfg.extra_channels}
        channel_label = meta.channel_names[index]
        multichannel = meta.n_channels > 1
        cfg = _scaled(cfg, meta)
        del meta
    else:
        raw = load_zstack(cfg.input_path)
        multichannel = raw.ndim == 4
        if cfg.extra_channels:
            extras = channels.rgb_extras_volume(raw, cfg.extra_channels)
    prepared = prepare_volume(raw, cfg.preprocess, cfg.channel)
    if channel_label:
        prepared.channel_name = channel_label
    del raw  # a z-stack is the largest object in play; drop it as soon as possible

    labels = segment(prepared.segmentation, cfg.segmentation)
    labels, cleanup = clean_labels(labels, cfg.segmentation)

    name = Path(cfg.input_path).name
    measurements = measure_objects(
        labels, prepared.measurement, image_name=name,
        voxel_size_xy=cfg.voxel_size_xy, voxel_size_z=cfg.voxel_size_z, unit=cfg.voxel_unit,
    )

    sholl_curves: dict = {}
    skeleton = None
    if cfg.arbor.enabled and labels.max() > 0:
        skeleton = _new_skeleton_buffer(labels, cfg)
        arbor_df, sholl_curves = analyze_arbor(
            labels, cfg.arbor, intensity=prepared.measurement,
            spacing=_spacing(cfg, is_3d=True), unit=cfg.voxel_unit, skeleton_out=skeleton,
        )
        measurements = reporting.consolidate(measurements, arbor_df, name)
        if skeleton is None:
            skeleton = skeleton_metrics(
                labels > 0, cfg.arbor.prune_length, _spacing(cfg, is_3d=True)
            ).skeleton

    summary = summarize_measurements(measurements, name, cfg.voxel_unit)
    measurements, summary = channels.add_channel_measurements(
        measurements, summary, labels, prepared.measurement, extras, cfg,
    )
    del extras
    result = AnalysisResult(
        measurements=measurements, summary=summary, labels=labels, prepared=prepared,
        sholl_curves=sholl_curves, cleanup=cleanup, skeleton=skeleton, name=name,
    )
    if multichannel:
        note = auto_channel_note(cfg, name, prepared.channel_name)
        if note:
            result.notes.append(note)

    mask = labels > 0
    save_volume_tiff(out / "HIVE_3D_volume.tif", prepared.measurement,
                     cfg.voxel_size_xy, cfg.voxel_size_z, cfg.voxel_unit)
    save_volume_tiff(out / "HIVE_3D_mask.tif", mask.astype(np.float32),
                     cfg.voxel_size_xy, cfg.voxel_size_z, cfg.voxel_unit)
    save_uint8(out / "HIVE_3D_max_projection.png", prepared.measurement.max(axis=0))
    save_uint8(out / "HIVE_3D_mask_projection.png", mask.max(axis=0).astype(np.float32))

    reporting.export_csv(out / "HIVE_3D_measurements.csv", measurements)
    reporting.export_csv(out / "HIVE_3D_summary.csv", summary)
    sholl_df = reporting.sholl_curves_to_frame(sholl_curves, name)
    if not sholl_df.empty:
        reporting.export_csv(out / "HIVE_3D_sholl_curves.csv", sholl_df)
    reporting.export_excel_report(
        out / "HIVE_3D_report.xlsx", summary, measurements,
        settings=cfg.to_flat_dict(), sholl_curves=sholl_df,
    )

    viewer_path = out / "HIVE_3D_viewer.html"
    verts, faces = create_3d_mesh_html(
        viewer_path, mask, cfg.voxel_size_xy, cfg.voxel_size_z, cfg.mesh_step_size
    )
    result.outputs["viewer_html"] = viewer_path
    try:        # the interactive volume view; a failure here must not lose the analysis
        from .volume_view import volume_viewer_html

        view_path = out / "HIVE_3D_volume_viewer.html"
        view_path.write_text(volume_viewer_html(
            prepared.measurement, mask, (cfg.voxel_size_z, cfg.voxel_size_xy, cfg.voxel_size_xy),
            title=name), encoding="utf-8")
        result.outputs["volume_viewer"] = view_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build the 3D volume view: %s", exc)
        result.notes.append(f"The interactive 3D volume view could not be built: {exc}")
    if verts is not None:
        result.outputs["mesh_obj"] = export_obj(out / "HIVE_3D_mesh.obj", verts, faces)

    if render_gif:
        from .render3d import render_rotation_gif
        result.outputs["gif"] = render_rotation_gif(
            prepared.measurement, out / "HIVE_3D_rotation.gif", cfg.render,
            cfg.voxel_size_xy, cfg.voxel_size_z,
        )

    result.outputs.update(exports.export_label_files(out, "HIVE_3D", labels, cfg))
    result.outputs["excel"] = out / "HIVE_3D_report.xlsx"
    result.outputs["run_record"] = provenance.write_run_record(
        out, user_cfg, notes={"notes": result.notes} if result.notes else None)
    logger.info("3D: %s", cleanup.summary())
    return result
