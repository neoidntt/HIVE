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
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from hive.config import (
    ArborConfig, HiveConfig, PreprocessConfig, RenderConfig, SegmentationConfig,
)
from hive.hardware import get_backend
from hive.io_utils import VALID_EXTENSIONS
from hive.pipeline import analyze_folder, analyze_zstack
from hive.visualization import create_boundary_overlay, create_label_overlay

UPLOAD_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "czi"]

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
        path = folder / uploaded.name
        with path.open("wb") as handle:
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, handle, length=4 * 1024 * 1024)
        written.append(path)
    return written


def zip_folder(folder: Path) -> Path:
    """Zip a results folder for download.

    Args:
        folder: Directory to archive.

    Returns:
        Path to the written ``.zip``.
    """
    archive = Path(tempfile.gettempdir()) / f"{folder.name}_results"
    return Path(shutil.make_archive(str(archive), "zip", root_dir=folder))


def sidebar_config(workspace: Path) -> tuple[HiveConfig, str, object]:
    """Render sidebar controls and assemble a :class:`HiveConfig`.

    Returns:
        Tuple of (config, mode label, uploaded file handle(s)).
    """
    st.sidebar.markdown("## Analysis")
    mode = st.sidebar.radio("Mode", ["2D images", "3D z-stack"], horizontal=True)
    is_3d = mode == "3D z-stack"

    uploads = st.sidebar.file_uploader(
        "Upload image(s)" if not is_3d else "Upload stack or slices",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help=f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}",
    )

    channel = st.sidebar.selectbox(
        "Channel", ["auto", "magenta", "green", "red", "blue", "gray"],
        help="'auto' picks the highest-variance channel -- a contrast heuristic, "
             "not a measurement. The resolved choice is recorded in the report.",
    )

    st.sidebar.markdown("### Physical scale")
    voxel_xy = st.sidebar.number_input(
        "Pixel size (XY)", min_value=0.0001, value=1.0, format="%.4f",
        help="Leave at 1.0 to report pixels. Set the real size to get physical units.",
    )
    voxel_z = st.sidebar.number_input(
        "Z step", min_value=0.0001, value=1.0, format="%.4f", disabled=not is_3d,
    ) if is_3d else 1.0
    unit = st.sidebar.text_input("Unit label", "px" if voxel_xy == 1.0 else "um")

    st.sidebar.markdown("### Background & contrast")
    baseline = st.sidebar.selectbox(
        "Background estimator", ["tophat", "percentile", "smooth", "none"],
        help="tophat = rolling-ball family; handles uneven illumination without "
             "the dark halo that Gaussian smoothing leaves around bright objects.",
    )
    tophat_radius = st.sidebar.slider(
        "Top-hat radius", 3, 100, 25, disabled=baseline != "tophat",
        help="Must exceed your largest real feature, or real signal is eaten as background.",
    )
    gaussian_sigma = st.sidebar.slider("Gaussian sigma (segmentation only)", 0.0, 5.0, 1.0, 0.5)
    use_clahe = st.sidebar.checkbox(
        "CLAHE contrast enhancement", value=True,
        help="Applied ONLY to the segmentation image. Measurements are always taken "
             "from the background-corrected image in its original scale.",
    )

    st.sidebar.markdown("### Segmentation")
    method = st.sidebar.selectbox("Method", ["threshold", "cellpose"])
    threshold_scale = st.sidebar.slider(
        "Otsu threshold scale", 0.1, 3.0, 1.0, 0.05,
        help="1.0 is unmodified Otsu (1979). Values above 1 are more conservative. "
             "This is an empirical knob -- report the value you used.",
    )
    min_size = st.sidebar.number_input("Min object size (debris filter)", 0, value=80)
    max_size = st.sidebar.number_input("Max object size (0 = no limit)", 0, value=0)
    hole_size = st.sidebar.number_input("Fill holes up to", 0, value=50)
    clear_border = st.sidebar.checkbox(
        "Drop edge-touching objects", value=False,
        help="Objects clipped by the field of view have unmeasurable true size.",
    )
    watershed = st.sidebar.checkbox(
        "Split touching objects (watershed)", value=False,
        help="Assumes convex objects. Leave OFF for branched cells -- it will "
             "shatter a neuron into fragments.",
    )

    st.sidebar.markdown("### Arborization")
    arbor_on = st.sidebar.checkbox("Analyse branching / Sholl", value=True)
    sholl_step = st.sidebar.slider(
        "Sholl shell spacing", 1.0, 25.0, 5.0, 1.0, disabled=not arbor_on
    )
    sholl_center = st.sidebar.selectbox(
        "Sholl centre", ["centroid", "max_intensity"], disabled=not arbor_on,
        help="'max_intensity' better approximates a soma when an arbor is lopsided.",
    )
    prune = st.sidebar.slider("Prune spurs shorter than", 0, 20, 3, disabled=not arbor_on)

    render_cfg = RenderConfig()
    if is_3d:
        st.sidebar.markdown("### 3D render")
        render_cfg = RenderConfig(
            mode=st.sidebar.selectbox("Projection", ["mip", "alpha"]),
            n_frames=st.sidebar.slider("Frames", 8, 120, 36, 4),
            fps=st.sidebar.slider("FPS", 5, 30, 15),
            colormap=st.sidebar.selectbox(
                "Colormap", ["magma", "viridis", "inferno", "plasma", "hot", "gray", "cividis"]
            ),
            downsample=st.sidebar.slider("Downsample", 1, 6, 2),
            rotation_axis=st.sidebar.selectbox("Rotation axis", ["y", "x"]),
        )

    tophat: float | tuple[float, ...] = (
        (max(1.0, tophat_radius / 10.0), tophat_radius, tophat_radius) if is_3d else tophat_radius
    )
    cfg = HiveConfig(
        input_path=str(workspace / "input"),
        output_folder=str(workspace / "output"),
        mode="3d" if is_3d else "2d",
        channel=channel,
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
    )
    return cfg, mode, uploads


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
            measurement, labels = measurement.max(axis=0), labels.max(axis=0)
        else:
            measurement, labels = measurement[z], labels[z]

    overlay_kind = st.radio("Overlay", ["Boundaries", "Filled labels"], horizontal=True)
    overlay = (
        create_boundary_overlay(measurement, labels) if overlay_kind == "Boundaries"
        else create_label_overlay(measurement, labels)
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Measurement image** (background-corrected)")
        st.image(_display(measurement), width="stretch")
    with right:
        st.markdown(f"**Segmentation** ({int(labels.max())} objects in view)")
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
    cfg, mode, uploads = sidebar_config(workspace)
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

    if not uploads:
        st.info("Upload one or more images in the sidebar to begin.")
        st.stop()

    input_folder = workspace / "input"
    if input_folder.exists():
        shutil.rmtree(input_folder)
    paths = save_uploads(uploads, input_folder)
    st.success(f"{len(paths)} file(s) ready: {', '.join(p.name for p in paths[:4])}"
               + (" ..." if len(paths) > 4 else ""))

    is_3d = mode == "3D z-stack"
    if is_3d:
        # One multi-page file is a stack; several files are slices of one stack.
        cfg.input_path = str(paths[0]) if len(paths) == 1 else str(input_folder)

    want_gif = st.checkbox("Also render rotating 3D GIF", value=False, disabled=not is_3d)

    import os

    default_workers = min(4, max(1, (os.cpu_count() or 2) - 2))
    workers = st.slider(
        "Parallel workers (2D batches)", 1, os.cpu_count() or 8, default_workers,
        disabled=is_3d,
        help="Each worker analyses one image in its own process. Speeds up multi-image "
             "2D batches almost linearly. Each holds its own working set, so keep "
             "workers x per-image memory within RAM.",
    )

    if st.button("Run analysis", type="primary"):
        with st.spinner(f"Analysing on {workers} worker(s)..."):
            try:
                if is_3d:
                    st.session_state["result"] = analyze_zstack(cfg, render_gif=want_gif)
                    st.session_state["batch"] = None
                else:
                    st.session_state["batch"] = analyze_folder(cfg, workers=workers)
                    st.session_state["result"] = None
                st.session_state["cfg"] = cfg
            except (FileNotFoundError, ValueError, ImportError) as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

    result = st.session_state.get("result")
    batch = st.session_state.get("batch")
    if result is None and batch is None:
        st.stop()
    cfg = st.session_state.get("cfg", cfg)

    if batch is not None:
        _show_batch(batch, cfg)
    else:
        _show_3d(result, cfg)


def _show_batch(batch: dict, cfg: HiveConfig) -> None:
    """Render 2D batch results."""
    measurements: pd.DataFrame = batch["measurements"]
    summary: pd.DataFrame = batch["summary"]

    if batch["errors"]:
        st.warning(f"{len(batch['errors'])} file(s) failed and were skipped:")
        for name, err in batch["errors"]:
            st.caption(f"- {name}: {err}")

    cols = st.columns(3)
    cols[0].metric("Objects", len(measurements))
    cols[1].metric("Images", len(summary))
    if not measurements.empty and "intensity_mean" in measurements:
        cols[2].metric("Mean intensity", f"{measurements['intensity_mean'].mean():.1f}")

    st.subheader("Per-image summary")
    st.dataframe(summary, width="stretch")

    st.subheader("Per-object measurements")
    st.dataframe(measurements, width="stretch", height=320)

    _download_buttons(measurements, summary, batch.get("sholl"), cfg)
    _sholl_plot(batch.get("sholl"), cfg)


def _show_3d(result, cfg: HiveConfig) -> None:
    """Render 3D results, including the viewer and GIF."""
    cols = st.columns(3)
    cols[0].metric("Objects", result.object_count)
    cols[1].metric("Cleanup", f"{result.cleanup.final_count}/{result.cleanup.initial_count} kept")
    if not result.measurements.empty and "intensity_mean" in result.measurements:
        cols[2].metric("Mean intensity", f"{result.measurements['intensity_mean'].mean():.1f}")
    st.caption(result.cleanup.summary())

    st.subheader("Inspect")
    render_side_by_side(result, cfg)

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
    st.dataframe(result.measurements, width="stretch", height=320)

    from hive.reporting import sholl_curves_to_frame

    sholl = sholl_curves_to_frame(result.sholl_curves, result.name)
    _download_buttons(result.measurements, result.summary, sholl, cfg)
    _sholl_plot(sholl, cfg)


def _download_buttons(measurements, summary, sholl, cfg: HiveConfig) -> None:
    """Render CSV / ZIP download controls."""
    st.subheader("Export")
    cols = st.columns(3)
    if measurements is not None and not measurements.empty:
        cols[0].download_button(
            "Measurements CSV", measurements.to_csv(index=False).encode(),
            file_name="hive_measurements.csv", mime="text/csv",
        )
    if summary is not None and not summary.empty:
        cols[1].download_button(
            "Summary CSV", summary.to_csv(index=False).encode(),
            file_name="hive_summary.csv", mime="text/csv",
        )
    output = Path(cfg.output_folder)
    if output.exists() and any(output.iterdir()):
        archive = zip_folder(output)
        with open(archive, "rb") as handle:
            cols[2].download_button("All results (ZIP)", handle, file_name=archive.name,
                                    mime="application/zip")


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
        import plotly.express as px

        fig = px.line(
            sholl, x="radius", y="intersections", color="label", markers=True,
            labels={"radius": f"Distance from centre ({cfg.voxel_unit})",
                    "intersections": "Intersections N(r)"},
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    except ImportError:
        st.line_chart(sholl.pivot(index="radius", columns="label", values="intersections"))


if __name__ == "__main__":
    main()
