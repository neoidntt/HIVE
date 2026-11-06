"""HIVE Streamlit application (Phase 4, Format B).

Run with::

    streamlit run app.py

Design notes
------------
Results live in ``st.session_state``. Streamlit re-executes the whole script on
every widget interaction, so a result held in a local variable is destroyed the
moment the user touches a slider -- the legacy app recomputed the entire
analysis (or lost the results) on every click.

Uploads are written to a session-scoped temporary directory rather than a fresh
``mkdtemp`` per rerun, which otherwise leaks a new copy of every uploaded stack
each time a widget changes.
"""

from __future__ import annotations

import atexit
import base64
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tifffile

if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        import subprocess
        import sys

        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
            )
        )
from hive.config import (
    ArborConfig, HiveConfig, PreprocessConfig, RenderConfig, SegmentationConfig,
)
from hive import filedialog, provenance, reporting, stats
from hive.config import ExportConfig, LargeImageConfig
from hive.formats import is_microscope_file, read_microscope
from hive.hardware import get_backend
from hive.io_utils import VALID_EXTENSIONS
from hive.io_utils import collect_image_paths
from hive.isolate import run_isolated
from hive.pipeline import analyze_folder, analyze_stacks, analyze_zstack, auto_workers
from hive.synthetic import make_synthetic_2d, make_synthetic_3d
from hive.visualization import create_boundary_overlay, create_label_overlay

UPLOAD_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "czi", "nd2", "lif"]
COLOUR_CHANNELS = ["magenta", "green", "red", "blue", "gray"]
LOCAL_SOURCE = "File or folder on this computer"
# Above this much disk space the results are left in their folder instead of being offered
# as one download (Streamlit would have to hold the whole thing in memory to send it).
# A browser cannot show millions of rows or points; beyond these the page shows a sample and
# says so, and the complete table stays in the CSV in the results folder.
MAX_TABLE_ROWS = 20_000
MAX_PLOT_POINTS = 20_000
DOWNLOAD_LIMIT_BYTES = int(os.environ.get("HIVE_DOWNLOAD_LIMIT_BYTES", 1_000_000_000))

# Branding only. Deliberately does NOT restyle the sidebar background or force text
# colours -- Streamlit's native theming already makes every control readable in both
# light and dark, and the old forced-black sidebar left input values white-on-white
# in light mode. Everything below is either a self-contained coloured card (readable
# on its own background in any theme) or an accent that inherits the theme's text.
BRAND_CSS = """
<style>
:root {
    --hive-yellow: #f9b410;
    --hive-blue: #0d2942;
    --hive-red: #e80c54;
}
/* Accent the sidebar with a thin brand stripe -- no background/text override. */
[data-testid="stSidebar"] {
    border-right: 3px solid var(--hive-yellow);
}
.hero-card {
    background: linear-gradient(135deg, var(--hive-blue) 0%, #1a4066 100%);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    border-left: 5px solid var(--hive-yellow);
}
/* The card has a fixed navy background, so white text is correct in any theme. */
.hero-card h1 { color: #ffffff; margin: 0 0 .25rem 0; font-weight: 700; }
.hero-card p  { color: #d9e2ec; margin: 0; font-size: .95rem; }
.stButton > button {
    background: linear-gradient(90deg, var(--hive-yellow) 0%, #ffd25c 100%);
    color: #000000; border: 1px solid #c98a00; border-radius: 10px; font-weight: 700;
}
.stDownloadButton > button {
    background: linear-gradient(90deg, var(--hive-red) 0%, #ff4f84 100%);
    color: #ffffff; border: 1px solid #b6003f; border-radius: 10px; font-weight: 700;
}
/* Translucent yellow works over both light and dark; text stays the theme colour. */
.caveat {
    background: rgba(249,180,16,.12); border-left: 4px solid var(--hive-yellow);
    padding: .6rem .9rem; border-radius: 6px; font-size: .86rem;
}
/* White card so the white-background logo reads cleanly in light AND dark mode. */
.logo-card {
    background: #ffffff;
    border-radius: 16px;
    border-left: 6px solid var(--hive-yellow);
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,.12);
    text-align: center;
}
.logo-card img { max-width: 460px; width: 60%; height: auto; }
</style>
"""


@st.cache_data
def _load_logo() -> Path | None:
    """Locate the HIVE logo bundled in ``assets/``.

    Returns:
        Path to the logo, or ``None`` if it is not present (the app then falls
        back to a text header, so a missing file is not fatal).
    """
    candidate = Path(__file__).parent / "assets" / "hive_logo.jpg"
    return candidate if candidate.exists() else None


@st.cache_data
def _logo_data_uri() -> str:
    """Return the logo as a base64 ``data:`` URI for inline HTML embedding.

    Inlining avoids Streamlit's static-file serving and lets the logo sit inside
    a styled card. Cached so the encode happens once per session.

    Returns:
        A ``data:image/jpeg;base64,...`` string, or ``""`` if the logo is absent.
    """
    logo = _load_logo()
    if logo is None:
        return ""
    encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@st.cache_resource
def session_workspace() -> Path:
    """Create one temporary workspace for the whole session.

    ``st.cache_resource`` makes this run once per session rather than once per
    rerun, so uploads are not re-copied on every widget interaction. Registered
    for cleanup at interpreter exit.

    Returns:
        Path to the session's scratch directory.
    """
    workspace = Path(tempfile.mkdtemp(prefix="hive_session_"))
    atexit.register(shutil.rmtree, workspace, True)
    return workspace


@st.cache_resource
def detect_backend():
    """Detect compute hardware once per session.

    Returns:
        The resolved :class:`~hive.hardware.Backend`.
    """
    return get_backend()


def save_uploads(files, folder: Path) -> list[Path]:
    """Persist uploaded files to disk in chunks.

    Args:
        files: Streamlit ``UploadedFile`` objects.
        folder: Destination directory, created if absent.

    Returns:
        Paths written.
    """
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for uploaded in files:
        filename = Path(uploaded.name).name
        path = folder / filename
        suffix = Path(filename).suffix
        stem = Path(filename).stem
        duplicate = 2
        while path.exists():
            path = folder / f"{stem}_{duplicate}{suffix}"
            duplicate += 1
        with path.open("wb") as handle:
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, handle, length=4 * 1024 * 1024)
        written.append(path)
    return written


def zip_folder(folder: Path) -> Path:
    """Zip a results folder for download.

    Pictures, OME-TIFFs, Excel files and ROI zips are already compressed, so they are stored
    as they are; compressing them again took seconds and saved almost nothing. Text files
    are deflated at the fastest level.

    Args:
        folder: Directory to archive.

    Returns:
        Path to the written ``.zip``.
    """
    import zipfile

    archive = Path(tempfile.gettempdir()) / f"{folder.name}_results.zip"
    already_compressed = {".png", ".jpg", ".tif", ".tiff", ".zip", ".xlsx", ".gif"}
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                packed = path.suffix.lower() in already_compressed
                bundle.write(path, path.relative_to(folder),
                             compress_type=zipfile.ZIP_STORED if packed else zipfile.ZIP_DEFLATED,
                             compresslevel=None if packed else 1)
    return archive


def _zip_bytes(folder: Path) -> bytes:
    """The results folder as zip bytes (builds the archive, reads it, removes the file)."""
    archive = zip_folder(folder)
    try:
        return archive.read_bytes()
    finally:
        archive.unlink(missing_ok=True)


def _results_zip(folder: Path):
    """What to hand ``st.download_button`` for the all-results zip.

    Streamlit reruns this whole script on every click, and zipping the results used to
    happen on every rerun even if nobody downloaded anything. Newer Streamlit builds the
    data only when the button is pressed (a callable); older versions get the zip built
    once per set of results and remembered.
    """
    if "deferred data generation" in (st.download_button.__doc__ or ""):
        return lambda: _zip_bytes(folder)
    files = [p for p in folder.rglob("*") if p.is_file()]
    key = (str(folder), len(files), sum(p.stat().st_size for p in files))
    if st.session_state.get("results_zip_key") != key:
        st.session_state["results_zip_key"] = key
        st.session_state["results_zip"] = _zip_bytes(folder)
    return st.session_state["results_zip"]


def _index(options: list, value, default: int = 0) -> int:
    """Position of ``value`` in ``options``, or ``default`` if absent."""
    return options.index(value) if value in options else default


def _probe_metadata(uploads, workspace: Path):
    """Read the channel names and pixel size of the first uploaded microscope file.

    Cached in the session by file identity, so the file is written to disk and read
    once rather than on every widget interaction. Only the header is read.

    Returns:
        A :class:`~hive.formats.MicroscopeImage` without pixels, or ``None`` when
        the first upload is an ordinary image or cannot be read.
    """
    if not uploads:
        return None
    first = uploads[0]
    key = (first.name, first.size, getattr(first, "file_id", None))
    cache = st.session_state.setdefault("probe_cache", {})
    if key not in cache:
        folder = workspace / "probe"
        shutil.rmtree(folder, ignore_errors=True)
        try:
            path = save_uploads([first], folder)[0]
            cache[key] = read_microscope(path, load_pixels=False) if is_microscope_file(path) else None
        except Exception as exc:  # noqa: BLE001 - a bad header should not stop the app
            st.sidebar.warning(f"Could not read the metadata of {first.name}: {exc}")
            cache[key] = None
    return cache[key]


def _choose_path(kind: str) -> None:
    """Button callback: open the system file dialog and put the answer in the path box."""
    picked = filedialog.pick(kind)
    if picked:
        st.session_state["local_path"] = picked


def _local_path_inputs() -> str:
    """Sidebar controls for opening data in place; returns the path typed or chosen."""
    st.sidebar.caption(
        "Opens the file where it is: nothing is uploaded or copied, so there is no size "
        "limit. Use this for anything over about 1 GB."
    )
    if filedialog.available():
        left, right = st.sidebar.columns(2)
        left.button("Choose a file...", on_click=_choose_path, args=("file",), width="stretch")
        right.button("Choose a folder...", on_click=_choose_path, args=("folder",), width="stretch")
    path = st.sidebar.text_input(
        "Path to a file or folder", key="local_path",
        help="A folder of images, one image, or one stack. TIFF files of any size are read in "
             "pieces; OME-TIFF is best for large multi-channel data.",
    )
    st.sidebar.text_input(
        "Save results in", value=str(Path.home() / "HIVE_results"), key="results_folder",
        help="A new sub-folder is made for each run; nothing already there is touched or deleted.",
    )
    return path.strip().strip('"').strip("'")


def _probe_local(target: Path):
    """Channel names and pixel size of a local file, from its header only (or ``None``)."""
    try:
        if target.is_file() and is_microscope_file(target):
            return read_microscope(target, load_pixels=False)
    except Exception:  # noqa: BLE001 - a bad header should not stop the app
        return None
    return None


def _disk_bytes(path: Path) -> int:
    """Space a file or folder really occupies on disk (sparse files count only what is written)."""
    total = 0
    for item in ([path] if path.is_file() else path.rglob("*")):
        try:
            if item.is_file():
                info = item.stat()
                total += getattr(info, "st_blocks", 0) * 512 or info.st_size
        except OSError:
            continue
    return total


def _describe_local(target: Path, is_3d: bool) -> None:
    """Say what was found at the chosen path and how it will be read."""
    from hive.io_utils import _available_memory_bytes, _fmt_bytes

    if target.is_dir():
        files = collect_image_paths(target)
        size = sum(f.stat().st_size for f in files)
        st.success(f"{len(files)} image file(s), {_fmt_bytes(size)}, in {target}")
        return
    size = target.stat().st_size
    detail = ""
    try:
        if target.suffix.lower() in (".tif", ".tiff"):
            with tifffile.TiffFile(str(target)) as tif:
                series = tif.series[0]
                detail = f" - {series.axes} {tuple(series.shape)}, {np.dtype(series.dtype).name}"
    except Exception:  # noqa: BLE001 - just a description
        pass
    free = _available_memory_bytes()
    st.success(f"{target.name}: {_fmt_bytes(size)}{detail}")
    if free and size * 4 > free * 0.4:
        st.info(
            f"This is larger than the memory free right now ({_fmt_bytes(free)}), so HIVE will "
            "analyse it in tiles, a few at a time, reading only what it needs from disk. A big "
            "file can take a long while; the progress bar shows how far it has got."
        )


def _local_preview_plane(target: Path, cfg: HiveConfig):
    """A small central crop of one plane of a local file, read without loading the file."""
    from hive.tiled import open_source

    file = target
    if target.is_dir():
        found = collect_image_paths(target)
        if not found:
            return None
        file = found[0]
    try:
        if file.suffix.lower() in (".tif", ".tiff"):
            with open_source(str(file), cfg.channel) as src:
                z, h, w = src.shape
                y0, x0 = max(0, h // 2 - 512), max(0, w // 2 - 512)
                crop = src.read(z // 2, slice(y0, min(h, y0 + 1024)), slice(x0, min(w, x0 + 1024)))
            return crop, f"{file.name}, central 1024 px of plane {z // 2 + 1}"
        from hive.io_utils import load_image_native
        from hive.preprocessing import select_channel

        image = load_image_native(file)
        if image.size > 120_000_000:
            return None
        return select_channel(image, cfg.channel)[0], file.name  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not prepare a preview of {file.name}: {exc}")
        return None


def _saved_settings_base() -> HiveConfig:
    """The configuration to start from: a reloaded saved run, or the defaults.

    The uploader itself sits at the bottom of the sidebar (:func:`_saved_settings_input`);
    Streamlit keeps its value in ``session_state`` under the widget key, so it can be
    read here, before the widgets that take their defaults from it are drawn.
    """
    saved = st.session_state.get("saved_settings")
    if saved is None:
        return HiveConfig()
    try:
        return provenance.parse_run_record(saved.getvalue().decode("utf-8"), saved.name).config
    except (ValueError, UnicodeDecodeError):
        return HiveConfig()     # the uploader below explains what was wrong


def _saved_settings_input() -> None:
    """Sidebar control to reload a saved run, with a note on what it did."""
    saved = st.sidebar.file_uploader(
        "Reload saved settings", type=["json"], key="saved_settings",
        help="Pick the hive_run.json from an earlier run to restore every setting, "
             "including the random seed, so the result can be reproduced.",
    )
    if saved is None:
        return
    try:
        record = provenance.parse_run_record(saved.getvalue().decode("utf-8"), saved.name)
    except (ValueError, UnicodeDecodeError) as exc:
        st.sidebar.error(f"Could not use {saved.name}: {exc}")
        return
    st.sidebar.caption(f"Settings restored from {saved.name}.")
    for line in provenance.environment_differences(record.record):
        st.sidebar.caption(f"Library differs from the saved run: {line}")


def sidebar_config(workspace: Path) -> tuple[HiveConfig, str, object, str, int, object, str]:
    """Render sidebar controls and assemble a :class:`HiveConfig`.

    Returns:
        Tuple of (config, mode, uploads, data source, synthetic seed, metadata of the
        first uploaded or local microscope file or ``None``, path typed for a local source).
    """
    st.sidebar.markdown("## Analysis")
    base = _saved_settings_base()
    modes = ["2D images", "3D z-stack"]
    mode = st.sidebar.radio("Mode", modes, index=1 if base.mode == "3d" else 0, horizontal=True)
    is_3d = mode == "3D z-stack"

    st.sidebar.markdown("### Input data")
    data_source = st.sidebar.radio(
        "Data source", ["Synthetic demo", "Upload files", LOCAL_SOURCE],
        help="Browser uploads are held in memory, so they suit files up to about 1 GB. "
             "For bigger files use 'File or folder on this computer': there is no size limit.",
    )
    uploads = (
        st.sidebar.file_uploader(
            "Upload image(s)" if not is_3d else "Upload stack or slices",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help=f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}. Held in memory while "
                 "you work, so keep uploads under about 1 GB; larger files: use "
                 "'File or folder on this computer'.",
        )
        if data_source == "Upload files" else []
    )
    local_path = _local_path_inputs() if data_source == LOCAL_SOURCE else ""
    seed = int(st.sidebar.number_input(
        "Synthetic data seed", min_value=0, max_value=2_147_483_647, value=0,
        disabled=data_source != "Synthetic demo",
        help="The same seed recreates the same sample images for repeatable tests.",
    ))
    probe = (_probe_local(Path(local_path).expanduser()) if local_path
             else _probe_metadata(uploads, workspace))
    if probe is not None:
        st.sidebar.caption(
            f"{probe.source_format}: {probe.n_channels} channel(s) "
            f"({', '.join(probe.channel_names)}); pixel size "
            + (f"{probe.voxel_size_xy:.4g} um" if probe.voxel_size_xy else "not in the file")
        )

    channel_options = ["auto"] + (probe.channel_names if probe else COLOUR_CHANNELS)
    if base.channel not in channel_options:
        channel_options.append(base.channel)
    channel = st.sidebar.selectbox(
        "Channel", channel_options, index=_index(channel_options, base.channel),
        help="'auto' picks the highest-variance channel -- a contrast heuristic, "
             "not a measurement. The resolved choice is recorded in the report.",
    )
    extra_options = [c for c in channel_options if c not in ("auto", channel)]
    extra_channels = st.sidebar.multiselect(
        "Also measure (colocalization)", extra_options,
        default=[c for c in base.extra_channels if c in extra_options],
        help="Objects are found on the channel above. These channels are only measured "
             "at those objects, with Pearson and Manders colocalization against it.",
    )

    st.sidebar.markdown("### Physical scale")
    file_xy = probe.voxel_size_xy if probe else None
    file_z = probe.voxel_size_z if probe else None
    voxel_xy = st.sidebar.number_input(
        "Pixel size (XY)", min_value=0.0001,
        value=float(base.voxel_size_xy if base.voxel_size_xy != 1.0 else (file_xy or 1.0)),
        format="%.4f",
        help="Leave at 1.0 to report pixels. Set the real size to get physical units. "
             "Filled in from the file when it stores one.",
    )
    voxel_z = st.sidebar.number_input(
        "Z step", min_value=0.0001,
        value=float(base.voxel_size_z if base.voxel_size_z != 1.0 else (
            file_z or (3.0 if data_source == "Synthetic demo" else 1.0))),
        format="%.4f", disabled=not is_3d,
    ) if is_3d else 1.0
    default_unit = base.voxel_unit if base.voxel_unit != "px" else ("um" if voxel_xy != 1.0 else "px")
    unit = st.sidebar.text_input("Unit label", default_unit)

    st.sidebar.markdown("### Background & contrast")
    pre = base.preprocess
    baselines = ["tophat", "percentile", "smooth", "none"]
    baseline = st.sidebar.selectbox(
        "Background estimator", baselines, index=_index(baselines, pre.baseline_method),
        help="tophat = rolling-ball family; handles uneven illumination without "
             "the dark halo that Gaussian smoothing leaves around bright objects. "
             "Use the preview in the main panel to compare them on your image.",
    )
    saved_radius = pre.tophat_radius[-1] if isinstance(pre.tophat_radius, tuple) else pre.tophat_radius
    tophat_radius = st.sidebar.slider(
        "Top-hat radius", 3, 100, int(min(100, max(3, saved_radius))),
        disabled=baseline != "tophat",
        help="Must exceed your largest real feature, or real signal is eaten as background.",
    )
    gaussian_sigma = st.sidebar.slider(
        "Gaussian sigma (segmentation only)", 0.0, 5.0,
        float(min(5.0, max(0.0, round(pre.gaussian_sigma * 2) / 2))), 0.5)
    use_clahe = st.sidebar.checkbox(
        "CLAHE contrast enhancement", value=pre.use_clahe,
        help="Applied ONLY to the segmentation image. Measurements are always taken "
             "from the background-corrected image in its original scale.",
    )

    st.sidebar.markdown("### Segmentation")
    seg = base.segmentation
    methods = ["threshold", "cellpose"]
    method = st.sidebar.selectbox("Method", methods, index=_index(methods, seg.method))
    threshold_scale = st.sidebar.slider(
        "Otsu threshold scale", 0.1, 3.0, float(min(3.0, max(0.1, seg.threshold_scale))), 0.05,
        help="1.0 is unmodified Otsu (1979). Values above 1 are more conservative. "
             "This is an empirical knob -- report the value you used.",
    )
    min_size = st.sidebar.number_input("Min object size (debris filter)", 0, value=seg.min_size)
    max_size = st.sidebar.number_input("Max object size (0 = no limit)", 0, value=seg.max_size)
    hole_size = st.sidebar.number_input("Fill holes up to", 0, value=seg.hole_size)
    clear_border = st.sidebar.checkbox(
        "Drop edge-touching objects", value=seg.clear_border,
        help="Objects clipped by the field of view have unmeasurable true size.",
    )
    watershed = st.sidebar.checkbox(
        "Split touching objects (watershed)", value=seg.watershed_split,
        help="Assumes convex objects. Leave OFF for branched cells -- it will "
             "shatter a neuron into fragments.",
    )

    st.sidebar.markdown("### Arborization")
    arb = base.arbor
    arbor_on = st.sidebar.checkbox("Analyse branching / Sholl", value=arb.enabled)
    sholl_step = st.sidebar.slider(
        "Sholl shell spacing", 1.0, 25.0, float(min(25.0, max(1.0, arb.sholl_step))), 1.0,
        disabled=not arbor_on,
    )
    centres = ["centroid", "max_intensity"]
    sholl_center = st.sidebar.selectbox(
        "Sholl centre", centres, index=_index(centres, arb.sholl_center), disabled=not arbor_on,
        help="'max_intensity' better approximates a soma when an arbor is lopsided.",
    )
    prune = st.sidebar.slider(
        "Prune spurs shorter than", 0, 20, int(min(20, arb.prune_length)), disabled=not arbor_on)

    render_cfg = base.render
    if is_3d:
        st.sidebar.markdown("### 3D render")
        ren = base.render
        projections, colormaps = ["mip", "alpha"], [
            "magma", "viridis", "inferno", "plasma", "hot", "gray", "cividis"]
        axes = ["y", "x"]
        render_cfg = RenderConfig(
            mode=st.sidebar.selectbox("Projection", projections, index=_index(projections, ren.mode)),
            n_frames=st.sidebar.slider("Frames", 8, 120, int(min(120, max(8, ren.n_frames))), 4),
            fps=st.sidebar.slider("FPS", 5, 30, int(min(30, max(5, ren.fps)))),
            colormap=st.sidebar.selectbox("Colormap", colormaps, index=_index(colormaps, ren.colormap)),
            downsample=st.sidebar.slider("Downsample", 1, 6, int(min(6, max(1, ren.downsample)))),
            rotation_axis=st.sidebar.selectbox(
                "Rotation axis", axes, index=_index(axes, ren.rotation_axis)),
        )

    st.sidebar.markdown("### Reproducibility & export")
    _saved_settings_input()
    run_seed = int(st.sidebar.number_input(
        "Random seed", min_value=0, max_value=2_147_483_647, value=int(base.seed),
        help="Saved with the results in hive_run.json and used by every step that is random.",
    ))
    export_ome = st.sidebar.checkbox(
        "Export labels as OME-TIFF", value=base.export.ome_tiff_labels,
        help="Opens in napari and Fiji with the correct pixel size.")
    export_rois = st.sidebar.checkbox(
        "Export objects as ImageJ ROIs", value=base.export.imagej_rois,
        help="A zip that Fiji's ROI Manager opens directly.")
    tiling_modes = ["auto", "off", "force"]
    tiling = st.sidebar.selectbox(
        "Images and stacks larger than memory", tiling_modes, index=_index(tiling_modes, base.large_image.mode),
        help="auto: analyse in tiles only when the image or stack would not fit in memory. "
             "off: refuse stacks that do not fit. force: always tile (for testing).")
    tile_size = int(st.sidebar.number_input(
        "Tile size (px, 0 = automatic)", 0, value=int(base.large_image.tile_size),
        disabled=tiling == "off"))

    tophat: float | tuple[float, ...] = (
        (max(1.0, tophat_radius / 10.0), tophat_radius, tophat_radius) if is_3d else tophat_radius
    )
    cfg = HiveConfig(
        input_path=str(workspace / "input"),
        output_folder=str(workspace / "output"),
        mode="3d" if is_3d else "2d",
        channel=channel,
        extra_channels=extra_channels,
        conditions=dict(base.conditions),
        seed=run_seed,
        voxel_size_xy=voxel_xy,
        voxel_size_z=voxel_z,
        voxel_unit=unit or "px",
        preprocess=PreprocessConfig(
            gaussian_sigma=gaussian_sigma, use_clahe=use_clahe,
            baseline_method=baseline, tophat_radius=tophat,
        ),
        segmentation=SegmentationConfig(
            method=method, threshold_scale=threshold_scale, min_size=min_size,
            max_size=max_size, hole_size=hole_size, clear_border=clear_border,
            watershed_split=watershed,
        ),
        arbor=ArborConfig(
            enabled=arbor_on, sholl_step=sholl_step, sholl_center=sholl_center,
            prune_length=prune,
        ),
        render=render_cfg,
        export=ExportConfig(ome_tiff_labels=export_ome, imagej_rois=export_rois),
        large_image=LargeImageConfig(mode=tiling, tile_size=tile_size, halo=base.large_image.halo),
    )
    input_folder = workspace / "input"
    if is_3d and data_source == "Synthetic demo":
        cfg.input_path = str(input_folder / "synthetic_stack.tif")
    elif data_source == LOCAL_SOURCE:
        cfg.input_path = local_path
    elif is_3d and uploads and len(uploads) == 1:
        cfg.input_path = str(input_folder / Path(uploads[0].name).name)
    else:
        cfg.input_path = str(input_folder)
    return cfg, mode, uploads, data_source, seed, probe, local_path


def show_hardware() -> None:
    """Render the hardware panel, honestly."""
    import os

    backend = detect_backend()
    icon = {"cuda": "GPU", "mps": "GPU", "cpu": "CPU"}[backend.device]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Compute:** `{icon}` {backend.name}  ·  {os.cpu_count()} CPU cores")
    if backend.is_gpu:
        st.sidebar.caption(
            "GPU auto-detected and used where it helps (smoothing). Note: this pipeline is "
            "mostly connected-component and morphology work, which is CPU-bound — the parallel "
            "workers below give a much bigger speedup than the GPU here."
        )
    elif backend.notes:
        st.sidebar.caption(backend.notes)


def render_side_by_side(result, cfg: HiveConfig) -> None:
    """Show raw against segmented views for one result.

    Args:
        result: An :class:`~hive.pipeline.AnalysisResult`.
        cfg: The config that produced it.
    """
    measurement = result.prepared.measurement
    labels = result.labels
    if measurement.ndim == 3:
        z = st.slider("Z-plane", 0, measurement.shape[0] - 1, measurement.shape[0] // 2)
        view = st.radio("View", ["Single plane", "Max projection"], horizontal=True)
        if view == "Max projection":
            brightest_z = np.argmax(measurement, axis=0)
            measurement = measurement.max(axis=0)
            labels = np.take_along_axis(labels, brightest_z[None, ...], axis=0)[0]
        else:
            measurement, labels = measurement[z], labels[z]

    _overlay_pair(measurement, labels)


def _overlay_pair(measurement: np.ndarray, labels: np.ndarray, key: str = "overlay_kind",
                  raw: bool = False) -> None:
    """The image next to its segmentation overlay (``raw``: the image is not background-corrected)."""
    overlay_kind = st.radio("Overlay", ["Boundaries", "Filled labels"], horizontal=True, key=key)
    overlay = (
        create_boundary_overlay(measurement, labels) if overlay_kind == "Boundaries"
        else create_label_overlay(measurement, labels)
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Raw image**" if raw else "**Measurement image** (background-corrected)")
        st.image(_display(measurement), width="stretch")
    with right:
        st.markdown(f"**Segmentation** ({len(np.unique(labels)) - int(labels.min() == 0)} objects in view)")
        st.image(np.clip(overlay, 0, 1), width="stretch")


def _display(image: np.ndarray) -> np.ndarray:
    """Contrast-stretch an image for on-screen display only.

    Args:
        image: 2D array in any scale.

    Returns:
        float array in ``[0, 1]``.
    """
    from hive.cleanup import normalize_for_display

    return normalize_for_display(image)


def _preview_plane(cfg: HiveConfig, data_source: str, demo_images, demo_volume, uploads,
                   workspace: Path) -> tuple[np.ndarray, str] | None:
    """One 2D plane of the first available image, for the background preview.

    Returns:
        The plane (original intensity scale) and a caption, or ``None`` if there is
        nothing to preview or it cannot be read.
    """
    from hive.io_utils import load_image
    from hive.pipeline import _channel_choice
    from hive.preprocessing import select_channel

    if data_source == "Synthetic demo":
        if demo_volume is not None:
            mid = demo_volume.shape[0] // 2
            return demo_volume[mid], f"synthetic stack, plane {mid + 1}"
        return demo_images[0][1], demo_images[0][0]
    if not uploads:
        return None
    folder = workspace / "preview"
    shutil.rmtree(folder, ignore_errors=True)
    path = save_uploads([uploads[0]], folder)[0]
    try:
        if is_microscope_file(path):
            meta = read_microscope(path)
            volume = meta.volume(_channel_choice(meta, cfg))
            return volume[volume.shape[0] // 2], f"{path.name}, channel {meta.channel_names[_channel_choice(meta, cfg)]}"
        image = load_image(path)
        if image.ndim == 3 and image.shape[-1] not in (3, 4):
            return image[image.shape[0] // 2], f"{path.name}, middle plane"
        return select_channel(image, cfg.channel)[0], path.name  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - show the reason instead of crashing the page
        st.warning(f"Could not prepare a preview of {path.name}: {exc}")
        return None


def _background_preview(cfg: HiveConfig, plane: np.ndarray, caption: str) -> None:
    """Show all four background methods on a crop of one plane."""
    from hive.baseline_preview import center_crop, preview_figure, preview_methods

    st.caption(
        f"{caption}. Uses your current radius and settings on the central "
        f"{min(512, min(plane.shape))}-px crop. 'Pixels at zero' near 100% means the "
        "method is removing real signal."
    )
    st.pyplot(preview_figure(center_crop(plane), preview_methods(plane, cfg.preprocess)))


def _local_identity(local_path: str) -> tuple:
    """What identifies a local source: its path, size and modification time."""
    if not local_path:
        return ()
    try:
        info = Path(local_path).expanduser().stat()
        return (local_path, info.st_size, info.st_mtime_ns)
    except OSError:
        return (local_path,)


def _analysis_signature(source: str, mode: str, seed: int, cfg: HiveConfig, uploads,
                        local: tuple = ()) -> str:
    settings = tuple(sorted(
        (key, repr(value)) for key, value in cfg.to_flat_dict().items()
        if key not in {"input_path", "output_folder"}
    ))
    files = tuple(
        (file.name, file.size, getattr(file, "file_id", None)) for file in uploads
    )
    return repr((source, mode, seed if source == "Synthetic demo" else None, settings, files, local))


def main() -> None:
    """Run the Streamlit app."""
    logo = _load_logo()
    # Favicon: the bee logo if present, else the honeybee emoji.
    st.set_page_config(page_title="HIVE Image Analysis", layout="wide",
                       page_icon=str(logo) if logo else "honeybee")
    st.markdown(BRAND_CSS, unsafe_allow_html=True)

    # Persistent brand mark at the top of the sidebar (st.logo handles sizing/theme).
    if logo:
        st.logo(str(logo))

    if logo:
        # The logo art is on white, so it would read as a white box in dark mode.
        # Sitting it on an explicit white card makes that intentional in both themes.
        st.markdown(
            f'<div class="logo-card"><img src="{_logo_data_uri()}" alt="HIVE logo"></div>',
            unsafe_allow_html=True,
        )
    else:
        # Fallback if the logo file is missing.
        st.markdown(
            '<div class="hero-card"><h1>HIVE Image Analysis</h1>'
            "<p>Segmentation, intensity quantification, arborization (Sholl) analysis "
            "and 3D reconstruction for fluorescence microscopy.</p></div>",
            unsafe_allow_html=True,
        )

    workspace = session_workspace()
    cfg, mode, uploads, data_source, seed, probe, local_path = sidebar_config(workspace)
    show_hardware()

    st.markdown(
        '<div class="caveat"><b>How HIVE measures.</b> Objects are <i>found</i> on a '
        "contrast-enhanced image, but <i>measured</i> on the background-corrected image "
        "in its original intensity scale. That separation is what makes "
        "<code>intensity_mean</code> comparable between images and z-planes. Displayed "
        "images are contrast-stretched for visibility -- read the CSV for numbers."
        "</div>",
        unsafe_allow_html=True,
    )

    is_3d = mode == "3D z-stack"
    demo_images = None
    demo_volume = None
    if data_source == "Synthetic demo":
        st.subheader("Built-in synthetic sample")
        if is_3d:
            demo_volume = make_synthetic_3d(seed)
            z_preview = st.slider(
                "Preview Z-plane", 0, demo_volume.shape[0] - 1,
                demo_volume.shape[0] // 2, key="synthetic_z_preview",
            )
            preview_cols = st.columns(2)
            preview_cols[0].image(
                _display(demo_volume[z_preview]), caption=f"Plane {z_preview + 1}",
                alt="Synthetic z-stack slice",
            )
            preview_cols[1].image(
                _display(demo_volume.max(axis=0)), caption="Maximum intensity projection",
                alt="Maximum intensity projection of the synthetic z-stack",
            )
            st.caption(
                f"{demo_volume.shape[0]} planes · {demo_volume.shape[1]} × "
                f"{demo_volume.shape[2]} px · anisotropic sample (XY 1, Z 3)"
            )
        else:
            demo_images = make_synthetic_2d(seed)
            preview_cols = st.columns(len(demo_images))
            for column, (name, image) in zip(preview_cols, demo_images):
                column.image(
                    _display(image), caption=name.removeprefix("synthetic_").removesuffix(".tif"),
                    alt=f"{name} synthetic microscopy sample",
                )
            st.caption("Three seeded fields with six known cell-like objects per image.")
    elif data_source == LOCAL_SOURCE:
        if not local_path:
            st.info("Type a path or use **Choose a file...** / **Choose a folder...** in the "
                    "sidebar. The data stays where it is: nothing is uploaded or copied.")
            st.stop()
        if not Path(local_path).expanduser().exists():
            st.error(f"There is nothing at {local_path}")
            st.stop()
        _describe_local(Path(local_path).expanduser(), is_3d)
    elif not uploads:
        st.info("Upload one or more image files in the sidebar to begin.")
        st.stop()
    else:
        st.success(f"{len(uploads)} file(s) selected for analysis.")

    with st.expander("Preview background subtraction (optional)"):
        if st.checkbox(
            "Show the effect of each background method", key="bg_preview_on",
            help="Applies none / percentile / smooth / top-hat to one image so you can "
                 "choose before running the whole analysis.",
        ):
            preview = (_local_preview_plane(Path(local_path).expanduser(), cfg)
                       if data_source == LOCAL_SOURCE else
                       _preview_plane(cfg, data_source, demo_images, demo_volume, uploads,
                                      workspace))
            if preview is not None:
                _background_preview(cfg, *preview)

    each_stack = False
    if is_3d and data_source == LOCAL_SOURCE and Path(local_path).expanduser().is_dir():
        from hive.io_utils import ImageLoadError, check_slice_files

        try:        # a folder of whole stacks (many planes per file) cannot be one stack of slices
            check_slice_files(collect_image_paths(Path(local_path).expanduser()))
            looks_like_stacks = False
        except ImageLoadError:
            looks_like_stacks = True
        each_stack = st.checkbox(
            "Each file in this folder is its own stack (analysed one after another)",
            value=looks_like_stacks, key=f"each_stack_{local_path}",
            help="Tick this when every file is a whole z-stack. They run one at a time, because "
                 "one stack already uses every core and running two at once doubles the memory. "
                 "Leave it unticked when the files are the slices (one plane each) of one stack.")
    local_id = (_local_identity(local_path), each_stack)
    signature = _analysis_signature(data_source, mode, seed, cfg, uploads, local_id)
    if st.session_state.get("analysis_signature") != signature:
        for key in ("result", "batch", "cfg", "analysis_signature"):
            st.session_state.pop(key, None)

    want_gif = st.checkbox("Also render rotating 3D GIF", value=False, disabled=not is_3d)

    import os

    # One worker per CPU core, but no more than there are images to analyse.
    n_images = len(demo_images) if demo_images else max(1, len(uploads) if uploads else 1)
    default_workers = max(1, min(os.cpu_count() or 1, n_images))
    workers = st.slider(
        "Parallel workers (2D batches)", 1, os.cpu_count() or 8, default_workers,
        disabled=is_3d,
        help="Each worker analyses one image in its own process. Speeds up multi-image "
             "2D batches almost linearly. Each holds its own working set, so keep "
             "workers x per-image memory within RAM.",
    )

    if st.button("Run analysis", type="primary"):
        for key in ("result", "batch", "cfg", "analysis_signature"):
            st.session_state.pop(key, None)
        bar = st.progress(0.0, text="Starting...")

        def show_progress(fraction: float, message: str) -> None:
            bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

        with st.spinner(f"Analysing on {workers} worker(s)..."):
            try:
                if data_source == LOCAL_SOURCE:
                    # Open the data where it is; results go to a new folder of their own, and
                    # nothing the user already has is deleted or overwritten.
                    target = Path(local_path).expanduser()
                    base = Path(st.session_state.get("results_folder") or Path.home() / "HIVE_results")
                    run_folder = base.expanduser() / f"{target.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
                    cfg.input_path = str(target)
                    cfg.output_folder = str(run_folder)
                    input_paths = [target] if target.is_file() else collect_image_paths(target)
                else:
                    input_folder = workspace / "input"
                    if input_folder.exists():
                        shutil.rmtree(input_folder)
                    input_folder.mkdir(parents=True)

                    if data_source == "Upload files":
                        paths = save_uploads(uploads, input_folder)
                        if is_3d:
                            cfg.input_path = str(paths[0] if len(paths) == 1 else input_folder)
                    elif is_3d:
                        from hive.io_utils import save_volume_tiff

                        save_volume_tiff(
                            cfg.input_path, demo_volume, cfg.voxel_size_xy,
                            cfg.voxel_size_z, cfg.voxel_unit,
                        )
                    else:
                        for name, image in demo_images:
                            tifffile.imwrite(input_folder / name, image.astype(np.float32))

                    output_folder = Path(cfg.output_folder)
                    if output_folder.exists():
                        shutil.rmtree(output_folder)
                    input_paths = collect_image_paths(input_folder)

                if is_3d and each_stack:
                    # In a process of its own: if the computer runs out of memory, the analysis is what
                    # gets killed, not this app.
                    st.session_state["batch"] = run_isolated(
                        analyze_stacks, cfg, progress=show_progress)
                    st.session_state["result"] = None
                elif is_3d:
                    st.session_state["result"] = analyze_zstack(
                        cfg, render_gif=want_gif, progress=show_progress)
                    st.session_state["batch"] = None
                else:
                    # Never more workers than memory allows: swapping is slower than fewer.
                    safe = auto_workers(input_paths, cfg.preprocess.use_clahe)
                    used = min(workers, safe)
                    batch = run_isolated(analyze_folder, cfg, workers=used, progress=show_progress)
                    batch["workers_used"] = used
                    batch["workers_limited"] = used < workers
                    st.session_state["batch"] = batch
                    st.session_state["result"] = None
                st.session_state["cfg"] = cfg
                st.session_state["analysis_signature"] = _analysis_signature(
                    data_source, mode, seed, cfg, uploads, local_id
                )
                bar.empty()
            except Exception as exc:
                bar.empty()
                st.error(f"Analysis failed: {exc}")
                st.stop()

    if st.session_state.get("analysis_signature") != signature:
        st.info("Choose the settings and data, then select **Run analysis**.")
        st.stop()

    result = st.session_state.get("result")
    batch = st.session_state.get("batch")
    if result is None and batch is None:
        st.stop()
    cfg = st.session_state.get("cfg", cfg)

    if batch is not None and "stacks" in batch:
        _show_stacks(batch, cfg)
    elif batch is not None:
        _show_batch(batch, cfg)
    else:
        _show_3d(result, cfg)


def _show_batch(batch: dict, cfg: HiveConfig) -> None:
    """Render 2D batch results."""
    measurements: pd.DataFrame = batch["measurements"]
    summary: pd.DataFrame = batch["summary"]

    if batch.get("workers_used"):
        note = f"Analysed with {batch['workers_used']} parallel worker(s)."
        if batch.get("workers_limited"):
            note += " Fewer than you chose, because the images need more memory than that many would leave free."
        st.caption(note)
    for note in batch.get("notes", []):
        st.info(note)
    if batch["errors"]:
        st.warning(f"{len(batch['errors'])} file(s) failed and were skipped:")
        for name, err in batch["errors"]:
            st.caption(f"- {name}: {err}")

    if not summary.empty and "image" in summary.columns:
        image_names = summary["image"].astype(str).tolist()
        selected_image = st.selectbox("Image to inspect", image_names)
        stem = Path(selected_image).stem
        output_folder = Path(cfg.output_folder)
        measurement_path = output_folder / f"{stem}_measurement.png"
        overlay_path = output_folder / f"{stem}_boundary_overlay.png"
        if measurement_path.exists() and overlay_path.exists():
            st.subheader("2D image inspection")
            image_cols = st.columns(2)
            image_cols[0].image(
                str(measurement_path), caption="Background-corrected measurement image",
                alt="Selected 2D measurement image",
            )
            image_cols[1].image(
                str(overlay_path), caption="Detected object boundaries",
                alt="Segmentation boundaries over the selected 2D image",
            )

    cols = st.columns(3)
    cols[0].metric("Objects", len(measurements))
    cols[1].metric("Images", len(summary))
    if not measurements.empty and "intensity_mean" in measurements:
        cols[2].metric("Mean intensity", f"{measurements['intensity_mean'].mean():.1f}")

    st.subheader("Per-image summary")
    st.dataframe(summary, width="stretch")

    st.subheader("Per-object measurements")
    _show_table(measurements, width="stretch", height=320)
    _measurement_charts(measurements)

    _conditions_section(batch, cfg)
    _download_buttons(measurements, summary, batch.get("sholl"), cfg, batch.get("outputs"))
    _sholl_plot(batch.get("sholl"), cfg)


def _conditions_section(batch: dict, cfg: HiveConfig) -> None:
    """Label images by condition and compare them, with effect sizes and plots."""
    summary: pd.DataFrame = batch["summary"]
    if summary.empty or len(summary) < 2:
        return
    st.subheader("Compare conditions")
    st.caption(
        "Give each image a condition label (for example control / treated). Images are the "
        "unit of replication by default, because cells in one image are not independent. "
        "A p-value needs at least 3 images per condition; the effect size is always shown."
    )
    names = summary["image"].astype(str).tolist()
    frame = pd.DataFrame({"image": names,
                          "condition": [cfg.conditions.get(n, "") for n in names]})
    edited = st.data_editor(frame, key="conditions_editor", hide_index=True,
                            disabled=["image"], width="stretch")
    conditions = {str(row.image): str(row.condition).strip() for row in edited.itertuples()
                  if str(row.condition).strip()}
    if len(set(conditions.values())) < 2:
        st.info("Enter at least two different condition labels to compare.")
        return
    unit = st.radio("Unit of replication", ["Image (recommended)",
                                            "Object (treats cells as independent)"],
                    horizontal=True)
    level = "image" if unit.startswith("Image") else "object"
    if level == "object":
        st.warning("Objects from the same image are not independent, so these p-values are "
                   "too small. Use them to explore, not to report.")

    measurements: pd.DataFrame = batch["measurements"]
    data = stats.replicate_table(summary, measurements, conditions, level)
    result = stats.compare_conditions(data, level=level)
    if result.empty:
        st.info("Nothing to compare yet for these labels.")
        return
    st.dataframe(result, width="stretch")
    metric = st.selectbox("Plot", result["metric"].unique().tolist(), key="group_metric")
    st.pyplot(stats.plot_groups(data, metric, level, cfg.seed))

    # Keep the results folder, Excel report and run record in step with what is shown.
    if st.session_state.get("written_conditions") != (conditions, level):
        cfg.conditions = conditions
        output = Path(cfg.output_folder)
        sheets = stats.write_group_outputs(output, summary, measurements, conditions,
                                           cfg.seed, level)
        reporting.export_excel_report(
            output / "HIVE_2D_report.xlsx", summary, measurements, settings=cfg.to_flat_dict(),
            sholl_curves=batch.get("sholl"), extra_sheets=sheets)
        provenance.write_run_record(output, cfg)
        st.session_state["written_conditions"] = (conditions, level)


def _embed_html(page: str, height: int = 700) -> None:
    """Show a HIVE-made HTML page inside the app (``st.iframe`` where this Streamlit has it)."""
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:
        import streamlit.components.v1 as components

        components.html(page, height=height, scrolling=False)


@st.cache_data(show_spinner=False, max_entries=2)
def _read_page(path: str, modified: int) -> str:
    return Path(path).read_text(encoding="utf-8")


def _volume_view_section(page: str, caption: str = "") -> None:
    """The interactive 3D view: built from every z-plane, drag to rotate, scroll to zoom."""
    st.subheader("3D volume (rotate, zoom, pan)")
    st.caption(
        "Drag to rotate, scroll to zoom, right-drag (or Shift+drag) to pan. Built from every "
        "z-plane. Thin samples are stretched in depth so the 3D shape can be seen: the slider "
        "\"Depth exaggeration\" in the view sets it (1 = true to scale). " + caption)
    _embed_html(page)


def _show_3d(result, cfg: HiveConfig) -> None:
    """Render 3D results, including the viewer and GIF."""
    cols = st.columns(3)
    cols[0].metric("Objects", result.object_count)
    cols[1].metric("Cleanup", f"{result.cleanup.final_count}/{result.cleanup.initial_count} kept")
    if not result.measurements.empty and "intensity_mean" in result.measurements:
        cols[2].metric("Mean intensity", f"{result.measurements['intensity_mean'].mean():.1f}")
    st.caption(result.cleanup.summary())
    for note in result.notes:
        st.info(note)

    if result.prepared is None:
        _show_3d_tiled(result, cfg)
    else:
        st.subheader("Inspect")
        render_side_by_side(result, cfg)
        page = result.outputs.get("volume_viewer")
        if page and Path(page).exists():
            _volume_view_section(_read_page(str(page), Path(page).stat().st_mtime_ns))
        _show_3d_point_cloud(result, cfg)

    gif_path = result.outputs.get("gif")
    if gif_path and Path(gif_path).exists():
        st.subheader("3D rotation")
        st.image(str(gif_path))
        with open(gif_path, "rb") as handle:
            st.download_button("Download GIF", handle, file_name=Path(gif_path).name,
                               mime="image/gif")

    viewer = result.outputs.get("viewer_html")
    if viewer and Path(viewer).exists():
        with open(viewer, "rb") as handle:
            st.download_button("Download interactive 3D viewer (HTML)", handle,
                               file_name=Path(viewer).name, mime="text/html")

    st.subheader("Per-object measurements")
    _show_table(result.measurements, width="stretch", height=320)

    from hive.reporting import sholl_curves_to_frame

    sholl = sholl_curves_to_frame(result.sholl_curves, result.name)
    _download_buttons(result.measurements, result.summary, sholl, cfg, result.outputs)
    _sholl_plot(sholl, cfg)
    _measurement_charts(result.measurements)


def _show_stacks(batch: dict, cfg: HiveConfig) -> None:
    """Results of a folder where every file was analysed as its own stack."""
    stacks, summary, measurements = batch["stacks"], batch["summary"], batch["measurements"]
    cols = st.columns(3)
    cols[0].metric("Stacks analysed", len(stacks))
    cols[1].metric("Failed", len(batch["errors"]))
    cols[2].metric("Objects", len(measurements))
    st.success(f"Each stack has its own folder inside {cfg.output_folder}")
    for note in batch.get("notes", []):
        st.info(note)
    for name, message in batch["errors"]:
        st.error(f"{name}: {message}")
    if not stacks:
        return

    st.subheader("Per-stack summary")
    st.dataframe(summary, width="stretch")

    st.subheader("Look at one stack")
    chosen = st.selectbox("Stack", [s["name"] for s in stacks], key="stacks_pick")
    stack = next(s for s in stacks if s["name"] == chosen)
    if not stack["labels_path"] or not Path(stack["labels_path"]).exists():
        st.caption("This stack's label image was not saved, so it cannot be viewed here.")
    else:
        import copy
        from types import SimpleNamespace

        import tifffile as _tifffile

        own = copy.copy(cfg)
        own.input_path, own.output_folder = str(Path(cfg.input_path) / stack["name"]), stack["folder"]
        shown = SimpleNamespace(
            labels=_tifffile.memmap(stack["labels_path"], mode="r"),
            outputs={"labels_ome_tiff": stack["labels_path"]},
            measurements=measurements[measurements["image"] == stack["name"]]
            if "image" in measurements else measurements)
        _show_3d_tiled(shown, own, prefix=f"{stack['name']}-", note=False)
        gif = Path(stack["folder"]) / "HIVE_3D_rotation.gif"
        if gif.exists():
            st.image(str(gif))

    st.subheader("Per-object measurements")
    _show_table(measurements, width="stretch", height=320)
    _measurement_charts(measurements)
    st.download_button("Download combined summary (CSV)", _csv_data(summary),
                       file_name="combined_3d_summary.csv", mime="text/csv")


def _show_3d_tiled(result, cfg: HiveConfig, prefix: str = "", note: bool = True) -> None:
    """Views of a stack that was analysed in tiles, read from disk instead of from memory."""
    from hive import preview

    if note:
        st.info(
            "This stack did not fit in memory, so it was analysed in tiles. The views below are "
            "read from the files on disk instead: a full-resolution look at any region, and a "
            "reduced copy of the whole stack for the projection, 3D view and GIF."
        )
    if "tile_clipped" in result.measurements and result.measurements["tile_clipped"].any():
        st.warning(
            f"{int(result.measurements['tile_clipped'].sum())} object(s) are larger than "
            "the tile overlap and may be cut at a tile edge (column tile_clipped)."
        )
    labels_path, source = result.outputs.get("labels_ome_tiff"), cfg.input_path
    if not labels_path or not source or not Path(source).exists():
        st.caption("The source or label file is no longer available, so these views cannot be built.")
        return

    st.subheader("Inspect")
    z_size, y_size, x_size = result.labels.shape
    tab_region, tab_whole, tab_3d = st.tabs(
        ["Region (full resolution)", "Whole stack (reduced)", "3D view and GIF"])

    with tab_region:
        z = st.slider("Z-plane", 0, z_size - 1, z_size // 2, key=f"{prefix}tiled_z")
        size = st.select_slider("Region size (pixels)", [256, 512, 1024, 2048], 1024,
                                key=f"{prefix}tiled_size")
        y0 = st.slider("Top edge (y)", 0, max(1, y_size - 1), max(0, y_size // 2 - size // 2),
                       step=64, key=f"{prefix}tiled_y0")
        x0 = st.slider("Left edge (x)", 0, max(1, x_size - 1), max(0, x_size // 2 - size // 2),
                       step=64, key=f"{prefix}tiled_x0")
        raw, labels = preview.read_region(source, labels_path, z, y0, x0, size, cfg.channel)
        _overlay_pair(raw, labels, key=f"{prefix}tiled_overlay_region", raw=True)

    # The reduced copy is read once (every plane, in turn) and kept for the session.
    stamp = (str(labels_path), Path(labels_path).stat().st_mtime_ns)

    def ensure_overview():
        cached = st.session_state.get("tiled_overview")
        if cached and cached[0] == stamp:
            return cached[1]
        if st.button("Load the whole-stack overview", key=f"{prefix}tiled_load",
                     help="Reads every plane once and keeps a reduced copy (about 30 million "
                          "voxels) in memory."):
            with st.spinner("Reading the stack plane by plane..."):
                view = preview.overview(source, labels_path, cfg.channel)
            st.session_state["tiled_overview"] = (stamp, view)
            return view
        return None

    with tab_whole:
        view = ensure_overview()
        if view is not None:
            st.caption(f"Every {view.stride}-th pixel of the {y_size} x {x_size} planes "
                       "(reduced view; measurements come from the full analysis).")
            kind = st.radio("View", ["Single plane", "Max projection"], horizontal=True,
                            key=f"{prefix}tiled_whole_kind")
            if kind == "Max projection":
                raw, labels = preview.max_projection(view)
            else:
                plane = st.slider("Z-plane", 0, z_size - 1, z_size // 2, key=f"{prefix}tiled_whole_z")
                raw, labels = view.raw[plane], view.labels[plane]
            _overlay_pair(raw, labels, key=f"{prefix}tiled_overlay_whole", raw=True)

    with tab_3d:
        view = st.session_state.get("tiled_overview")
        view = view[1] if view and view[0] == stamp else None
        if view is None:
            st.caption("Load the whole-stack overview in the previous tab first.")
        else:
            from hive.volume_view import volume_viewer_html

            key = (str(labels_path), view.stride, cfg.voxel_size_xy, cfg.voxel_size_z)
            cached = st.session_state.get(f"{prefix}tiled_volume_page")
            if not cached or cached[0] != key:
                with st.spinner("Building the 3D view from every z-plane..."):
                    page = volume_viewer_html(
                        view.raw, view.labels > 0,
                        (cfg.voxel_size_z, cfg.voxel_size_xy * view.stride, cfg.voxel_size_xy * view.stride),
                        title=Path(cfg.input_path).name)
                try:        # keep a copy next to the other results, to open without the app
                    (Path(cfg.output_folder) / "HIVE_3D_volume_viewer.html").write_text(page, encoding="utf-8")
                except OSError:
                    pass
                st.session_state[f"{prefix}tiled_volume_page"] = cached = (key, page)
            _volume_view_section(cached[1], f"Every {view.stride}th pixel of the full image is used in x and y.")
            _show_3d_point_cloud(result, cfg, labels=view.labels, step=view.stride,
                                 key=f"{prefix}tiled_point_cloud")
            if st.button("Make a rotation GIF from the reduced stack", key=f"{prefix}tiled_gif"):
                from hive.render3d import render_rotation_gif

                with st.spinner("Rendering..."):
                    path = render_rotation_gif(
                        view.raw, Path(cfg.output_folder) / "HIVE_3D_rotation.gif", cfg.render,
                        cfg.voxel_size_xy * view.stride, cfg.voxel_size_z)
                result.outputs["gif"] = path
                st.rerun()


def _show_table(table: pd.DataFrame, **kwargs) -> None:
    """``st.dataframe``, limited to the first rows when the table is huge."""
    if len(table) > MAX_TABLE_ROWS:
        st.caption(f"Showing the first {MAX_TABLE_ROWS:,} of {len(table):,} rows. "
                   "The complete table is in the CSV file in the results folder.")
        table = table.head(MAX_TABLE_ROWS)
    st.dataframe(table, **kwargs)


def _csv_data(table: pd.DataFrame):
    """CSV for a download button: built only when pressed if this Streamlit supports that."""
    if "deferred data generation" in (st.download_button.__doc__ or ""):
        return lambda: table.to_csv(index=False).encode()
    return table.to_csv(index=False).encode()


def _download_buttons(measurements, summary, sholl, cfg: HiveConfig, outputs=None) -> None:
    """Render CSV / ZIP download controls."""
    st.subheader("Export")
    output = Path(cfg.output_folder)
    on_disk = _disk_bytes(output) if output.exists() else 0
    if not str(output).startswith(str(session_workspace())):
        # A folder of the user's own choosing (not the temporary one): say where it is.
        st.success(f"Results are saved in {output}")
    cols = st.columns(4)
    if measurements is not None and not measurements.empty and len(measurements) <= 500_000:
        cols[0].download_button(
            "Measurements CSV", _csv_data(measurements),
            file_name="hive_measurements.csv", mime="text/csv",
        )
    elif measurements is not None and len(measurements) > 500_000:
        cols[0].caption(f"{len(measurements):,} objects: the CSV is in the results folder.")
    if summary is not None and not summary.empty:
        cols[1].download_button(
            "Summary CSV", summary.to_csv(index=False).encode(),
            file_name="hive_summary.csv", mime="text/csv",
        )
    offer_zip = output.exists() and any(output.iterdir()) and on_disk <= DOWNLOAD_LIMIT_BYTES
    if offer_zip:
        cols[2].download_button("All results (ZIP)", _results_zip(output),
                                file_name=f"{output.name}_results.zip", mime="application/zip")
    elif on_disk > DOWNLOAD_LIMIT_BYTES:
        cols[2].caption(f"Results use {on_disk / 1e9:.1f} GB, too much to send through the "
                        "browser: take them from the results folder.")
    excel_path = output / ("HIVE_3D_report.xlsx" if cfg.mode == "3d" else "HIVE_2D_report.xlsx")
    if excel_path.exists():
        with excel_path.open("rb") as handle:
            cols[3].download_button(
                "Excel report", handle, file_name=excel_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    extra = st.columns(3)
    record = output / provenance.RUN_RECORD_NAME
    if record.exists():
        extra[0].download_button(
            "Settings (hive_run.json)", record.read_bytes(), file_name=record.name,
            mime="application/json",
            help="Reload this in the sidebar (or with --settings) to reproduce the run.",
        )
    for column, kind, label, mime in (
        (extra[1], "labels_ome_tiff", "Labels (OME-TIFF)", "image/tiff"),
        (extra[2], "imagej_rois", "ImageJ ROIs (zip)", "application/zip"),
    ):
        path = (outputs or {}).get(kind)
        if path and Path(path).exists():
            if Path(path).stat().st_size > 300_000_000:
                column.caption(f"{Path(path).name} is large: open it from the results folder.")
            else:
                column.download_button(label, Path(path).read_bytes(), file_name=Path(path).name,
                                       mime=mime)
    if cfg.mode == "2d" and offer_zip:
        st.caption("Per-image labels (OME-TIFF) and ROI zips are inside the ZIP of all results.")


def _measurement_charts(measurements: pd.DataFrame) -> None:
    """Render interactive size/intensity plots for 2D or 3D measurements."""
    if measurements.empty or "intensity_mean" not in measurements.columns:
        st.info("No objects were detected for plotting with these settings.")
        return

    size_column = "volume_voxels" if "volume_voxels" in measurements.columns else "area_px"
    if size_column not in measurements.columns:
        return

    import plotly.express as px

    st.subheader("Interactive measurement plots")
    if len(measurements) > MAX_PLOT_POINTS:
        st.caption(f"Plotting a random {MAX_PLOT_POINTS:,} of {len(measurements):,} objects "
                   "(a browser cannot draw them all); the tables and CSV hold every one.")
        measurements = measurements.sample(MAX_PLOT_POINTS, random_state=0)
    axis_cols = st.columns(2)
    numeric_columns = [
        column for column in (size_column, "intensity_mean", "integrated_intensity")
        if column in measurements.columns
    ]
    x_column = axis_cols[0].selectbox("X-axis", numeric_columns, key="plot_x_axis")
    y_column = axis_cols[1].selectbox(
        "Y-axis", numeric_columns, index=min(1, len(numeric_columns) - 1), key="plot_y_axis",
    )
    color_column = "image" if "image" in measurements and measurements["image"].nunique() <= 8 else None
    scatter = px.scatter(
        measurements, x=x_column, y=y_column, color=color_column,
        hover_data=[column for column in ("image", "label") if column in measurements],
        labels={"area_px": "Area (pixels)", "volume_voxels": "Volume (voxels)"},
    )
    scatter.update_layout(height=420, margin=dict(l=8, r=8, t=24, b=8))
    st.plotly_chart(scatter, width="stretch")

    histogram = px.histogram(
        measurements, x="intensity_mean", color=color_column, marginal="box", nbins=32,
        labels={"intensity_mean": "Mean intensity"},
    )
    histogram.update_layout(height=360, margin=dict(l=8, r=8, t=24, b=8))
    st.plotly_chart(histogram, width="stretch")


def _show_3d_point_cloud(result, cfg: HiveConfig, labels=None, step: int = 1,
                         key: str = "hive_3d_point_cloud") -> None:
    """Render a browser-native, sampled 3D view of segmented voxels.

    ``labels`` and ``step`` let a reduced copy stand in for the full label image (``step`` is
    how many source pixels one of its pixels covers in y and x).
    """
    labels = result.labels if labels is None else labels
    coordinates = np.argwhere(labels > 0)
    if coordinates.size == 0:
        st.info("No segmented voxels are available for the 3D view.")
        return

    point_limit = st.slider(
        "3D display point limit", 1_000, 30_000, 12_000, 1_000,
        help="Large volumes are sampled for a responsive interactive view.",
    )
    if len(coordinates) > point_limit:
        chosen = np.random.default_rng(0).choice(len(coordinates), point_limit, replace=False)
        coordinates = coordinates[np.sort(chosen)]

    object_labels = labels[tuple(coordinates.T)]
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter3d(
        x=coordinates[:, 2] * step * cfg.voxel_size_xy,
        y=coordinates[:, 1] * step * cfg.voxel_size_xy,
        z=coordinates[:, 0] * cfg.voxel_size_z,
        mode="markers",
        marker=dict(
            size=2.5, color=object_labels, colorscale="Turbo", opacity=0.8,
            colorbar=dict(title="Object label"),
        ),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
    ))
    figure.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis_title=f"X ({cfg.voxel_unit})",
            yaxis_title=f"Y ({cfg.voxel_unit})",
            zaxis_title=f"Z ({cfg.voxel_unit})",
            aspectmode="data",
        ),
    )
    st.subheader("Interactive 3D segmentation")
    st.plotly_chart(figure, width="stretch", key=key)


MANY_CURVES = 12     # above this, curves are drawn as one trace instead of one coloured line each


@st.cache_data(show_spinner=False)
def _sholl_figure(sholl: pd.DataFrame, unit: str):
    """The Sholl chart, built once per table (the page redraws on every interaction).

    A handful of objects get one coloured line each, as before. With hundreds, a trace per
    object made plotly build and send hundreds of figure parts (over a second per redraw),
    so all curves go into a single thin trace -- same data, hover still names the object --
    with the average curve on top.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    labels = {"radius": f"Distance from centre ({unit})", "intersections": "Intersections N(r)"}
    groups = [g for _, g in sholl.groupby(["image", "label"], sort=False)]
    if len(groups) <= MANY_CURVES:
        figure = px.line(sholl, x="radius", y="intersections", color="label", markers=True,
                         labels=labels)
        figure.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        return figure

    x, y, hover = [], [], []
    for group in groups:
        x += group["radius"].tolist() + [None]          # None breaks the line between objects
        y += group["intersections"].tolist() + [None]
        name = f"{group['image'].iloc[0]} #{group['label'].iloc[0]}"
        hover += [name] * len(group) + [None]
    mean = sholl.groupby("radius")["intersections"].mean()
    figure = go.Figure([
        go.Scatter(x=x, y=y, mode="lines", line=dict(width=1, color="rgba(13,41,66,0.25)"),
                   text=hover, hovertemplate="%{text}<br>r=%{x}<br>N=%{y}<extra></extra>",
                   name=f"{len(groups)} objects"),
        go.Scatter(x=mean.index, y=mean.values, mode="lines", line=dict(width=3, color="#e80c54"),
                   name="average", hovertemplate="average<br>r=%{x}<br>N=%{y:.2f}<extra></extra>"),
    ])
    figure.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis_title=labels["radius"], yaxis_title=labels["intersections"])
    return figure


def _sholl_plot(sholl, cfg: HiveConfig) -> None:
    """Render an interactive Sholl profile chart."""
    if sholl is None or getattr(sholl, "empty", True):
        return
    st.subheader("Sholl profiles")
    st.caption(
        "Intersections N(r) against distance from each object's centre (Sholl, 1953). "
        "A peak marks the radius of maximum branching complexity."
    )
    try:
        st.plotly_chart(_sholl_figure(sholl, cfg.voxel_unit), width="stretch")
    except ImportError:
        st.line_chart(sholl.pivot(index="radius", columns="label", values="intersections"))


if __name__ == "__main__":
    main()
