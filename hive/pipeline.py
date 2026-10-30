"""End-to-end 2D and 3D analysis pipelines.

Both pipelines return an :class:`AnalysisResult` rather than a bare tuple. The
legacy ``analyze_zstack`` returned ``(measurements, summary)``, and the Streamlit
app bound that tuple to a single name and handed it to ``st.dataframe`` -- an
error a named result object cannot express.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import channels, exports, formats, provenance, reporting, stats
from .arborization import analyze_arbor, skeleton_metrics
from .cleanup import CleanupReport, clean_labels
from .config import HiveConfig
from .io_utils import collect_image_paths, load_image, load_zstack, save_uint8, save_volume_tiff
from .measurements import measure_objects, summarize_measurements
from .preprocessing import PreparedImage, prepare_image, prepare_volume
from .segmentation import segment
from .tiled import analyze_zstack_tiled, needs_tiling
from .visualization import (
    create_3d_mesh_html, create_boundary_overlay, create_label_overlay,
    create_skeleton_overlay, export_obj,
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

    @property
    def object_count(self) -> int:
        """int: Number of objects measured."""
        return 0 if self.measurements is None or self.measurements.empty else len(self.measurements)


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
        arbor_df, sholl_curves = analyze_arbor(
            labels, cfg.arbor, intensity=prepared.measurement,
            spacing=_spacing(cfg, is_3d=False), unit=cfg.voxel_unit,
        )
        measurements = reporting.consolidate(measurements, arbor_df, name)
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

    return AnalysisResult(
        measurements=measurements, summary=summary, labels=labels, prepared=prepared,
        sholl_curves=sholl_curves, cleanup=cleanup, skeleton=skeleton, name=name,
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


def analyze_image(image_path: str | Path, cfg: HiveConfig) -> AnalysisResult:
    """Run the full 2D pipeline on a single image file.

    OME-TIFF, CZI, ND2 and LIF files are read through :mod:`hive.formats`, so channels
    can be chosen by name and the pixel size is taken from the file (unless
    ``cfg`` already sets one).

    Args:
        image_path: Image file to analyse.
        cfg: Run configuration.

    Returns:
        An :class:`AnalysisResult`.

    Raises:
        FileNotFoundError: If the image does not exist.
        ImageLoadError: If the image cannot be decoded.
        SegmentationError: If the selected channel is empty.
    """
    image_path = Path(image_path)
    if formats.is_microscope_file(image_path):
        meta = formats.read_microscope(image_path)
        index = _channel_choice(meta, cfg)
        extras = {name: meta.plane(name) for name in cfg.extra_channels}
        return analyze_array(
            meta.plane(index), image_path.name, _scaled(cfg, meta),
            extras=extras, channel_name=meta.channel_names[index],
        )
    return analyze_array(load_image(image_path), image_path.name, cfg)


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

    save_uint8(out / f"{stem}_mask.png", (result.labels > 0).astype(np.float32))
    save_uint8(out / f"{stem}_boundary_overlay.png",
               create_boundary_overlay(result.prepared.measurement, result.labels))
    save_uint8(out / f"{stem}_label_overlay.png",
               create_label_overlay(result.prepared.measurement, result.labels))
    result.outputs["mask"] = out / f"{stem}_mask.png"
    result.outputs["boundary_overlay"] = out / f"{stem}_boundary_overlay.png"

    if result.skeleton is not None and result.skeleton.any():
        save_uint8(out / f"{stem}_skeleton.png", result.skeleton.astype(np.float32))
        save_uint8(out / f"{stem}_skeleton_overlay.png",
                   create_skeleton_overlay(result.prepared.measurement, result.skeleton,
                                           result.labels > 0))
        result.outputs["skeleton_overlay"] = out / f"{stem}_skeleton_overlay.png"

    if cfg.save_intermediate:
        save_uint8(out / f"{stem}_measurement.png", result.prepared.measurement)
        save_uint8(out / f"{stem}_segmentation.png", result.prepared.segmentation, rescale=False)

    reporting.export_csv(out / f"{stem}_measurements.csv", result.measurements)
    result.outputs["measurements_csv"] = out / f"{stem}_measurements.csv"
    result.outputs.update(exports.export_label_files(out, stem, result.labels, cfg))


def _analyze_one(job: tuple[Path, HiveConfig, bool]) -> tuple:
    """Analyse one image and return only its tables.

    Module-level (not a closure) so it can be pickled to a worker process.

    Crucially this returns **DataFrames, not arrays**. An
    :class:`AnalysisResult` carries the measurement volume, segmentation volume
    and label array; pickling those back to the parent would send gigabytes
    through a pipe per image and defeat the point of parallelising.

    Args:
        job: Tuple of (image path, config, whether to write QC artifacts).

    Returns:
        Tuple of (measurements, summary, sholl, error) where ``error`` is
        ``None`` on success and ``(name, message)`` on failure.
    """
    path, cfg, write_per_image = job
    try:
        result = analyze_image(path, cfg)
        if write_per_image:
            _write_2d_outputs(result, cfg, path.stem)
        sholl = (
            reporting.sholl_curves_to_frame(result.sholl_curves, path.name)
            if result.sholl_curves else None
        )
        return result.measurements, result.summary, sholl, None
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
        return None, None, None, (path.name, str(exc))


def _warn_on_worker_memory(workers: int, paths: list[Path]) -> None:
    """Warn when the requested worker count cannot fit in RAM.

    Each worker holds its own full copy of the pipeline's working set, so peak
    memory is roughly ``workers x per-image peak``. Oversubscribing does not
    fail fast -- it swaps, and on macOS that means the machine crawls for hours.

    Args:
        workers: Requested worker count.
        paths: Images to be processed (the largest sets the per-worker cost).
    """
    from .io_utils import PIPELINE_MEMORY_FACTOR, _peek_stack_bytes, _physical_memory_bytes

    total = _physical_memory_bytes()
    if not total or workers <= 1 or not paths:
        return
    peek = _peek_stack_bytes(paths[0])
    if peek is None:
        return
    estimated = peek[0] * PIPELINE_MEMORY_FACTOR * workers
    if estimated > total * 0.85:
        safe = max(1, int(total * 0.85 / (peek[0] * PIPELINE_MEMORY_FACTOR)))
        warnings.warn(
            f"{workers} workers x ~{peek[0] * PIPELINE_MEMORY_FACTOR / 1e9:.1f} GB each "
            f"= ~{estimated / 1e9:.0f} GB, but this machine has {total / 1e9:.0f} GB. "
            f"This will swap. Use workers={safe} or fewer. "
            "(Setting use_clahe=False typically cuts per-worker memory ~3x.)",
            stacklevel=3,
        )


def analyze_folder(cfg: HiveConfig, write_per_image: bool = True,
                   workers: int = 1) -> dict[str, object]:
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

    Returns:
        Dict with ``measurements``, ``summary``, ``sholl``, ``errors`` and
        ``outputs``.

    Raises:
        FileNotFoundError: If the input folder holds no readable images.
        ValueError: If ``workers`` is below 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    input_folder = Path(cfg.input_path)
    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)

    paths = collect_image_paths(input_folder)
    if not paths:
        raise FileNotFoundError(f"No readable images found in: {input_folder}")

    all_measurements: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_sholl: list[pd.DataFrame] = []
    errors: list[tuple[str, str]] = []

    workers = min(workers, len(paths))
    _warn_on_worker_memory(workers, paths)
    jobs = [(p, cfg, write_per_image) for p in paths]

    if workers > 1:
        logger.info("Analysing %d images on %d workers", len(paths), workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_analyze_one, jobs))
    else:
        results = [_analyze_one(job) for job in jobs]

    for measurements, summary, sholl, error in results:
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
    reporting.export_excel_report(
        out / "HIVE_2D_report.xlsx", summary, measurements,
        settings=cfg.to_flat_dict(), sholl_curves=sholl, extra_sheets=group_sheets,
    )
    run_record = provenance.write_run_record(out, cfg)

    if errors:
        logger.warning("%d of %d images failed", len(errors), len(paths))

    return {
        "measurements": measurements,
        "summary": summary,
        "sholl": sholl,
        "errors": errors,
        "group_stats": group_sheets.get("Group_Stats"),
        "outputs": {"excel": out / "HIVE_2D_report.xlsx",
                    "measurements_csv": out / "combined_measurements.csv",
                    "run_record": run_record},
    }


def analyze_zstack(cfg: HiveConfig, render_gif: bool = False) -> AnalysisResult:
    """Run the full 3D pipeline on a z-stack.

    The channel axis is reduced **before** baseline subtraction, which is the
    fix for the legacy crash: the old order handed a 4D ``(Z, Y, X, C)`` array
    to a function accepting only 2D/3D, so every RGB z-stack raised
    ``ValueError`` on the README's documented ``--mode 3d --channel magenta``
    workflow.

    Args:
        cfg: Run configuration. ``cfg.input_path`` is the stack file or folder.
        render_gif: Also render the rotating projection GIF.

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
        return analyze_zstack_tiled(cfg)

    user_cfg = cfg
    extras: dict[str, np.ndarray] = {}
    channel_label = None
    source = Path(cfg.input_path)
    if source.is_file() and formats.is_microscope_file(source):
        meta = formats.read_microscope(source)
        index = _channel_choice(meta, cfg)
        raw = meta.volume(index)
        extras = {name: meta.volume(name) for name in cfg.extra_channels}
        channel_label = meta.channel_names[index]
        cfg = _scaled(cfg, meta)
        del meta
    else:
        raw = load_zstack(cfg.input_path)
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
        arbor_df, sholl_curves = analyze_arbor(
            labels, cfg.arbor, intensity=prepared.measurement,
            spacing=_spacing(cfg, is_3d=True), unit=cfg.voxel_unit,
        )
        measurements = reporting.consolidate(measurements, arbor_df, name)
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
    result.outputs["run_record"] = provenance.write_run_record(out, user_cfg)
    logger.info("3D: %s", cleanup.summary())
    return result
