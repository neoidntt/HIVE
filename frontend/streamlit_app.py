"""LEGACY (HIVE 1.x) Streamlit front end.

.. deprecated:: 2.0
    Superseded by the top-level ``app.py``. Run that instead::

        streamlit run app.py

This module drives the legacy :mod:`src` pipeline, so every caveat about 1.x
applies: reported intensities are contrast-enhanced rather than quantitative,
and the 3D path rejects channel-last RGB stacks. Two further defects specific
to this file are documented on :func:`show_3d_results` and :func:`main`.
Retained only so 1.x outputs can be reproduced for comparison.
"""

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from skimage.io import imread

from src.config import HiveConfig
from src.pipeline import analyze_folder
from src.reconstruction_3d import analyze_zstack

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
# Advertised per-file upload ceiling (MB). Streamlit enforces its own limit via
# .streamlit/config.toml; this constant only drives the caption text.
MAX_UPLOAD_SIZE_MB = 8192


def save_uploaded_files(uploaded_files, target_folder: Path) -> list[Path]:
    """Stream uploaded files to disk, preserving their original names.

    Args:
        uploaded_files: Streamlit ``UploadedFile`` objects.
        target_folder: Destination directory, created if absent.

    Returns:
        Paths of the written files, in upload order -- which matters for
        z-stacks assembled from one file per slice.

    Note:
        Copies in 1 MB chunks rather than calling ``getbuffer()``, so a
        multi-gigabyte stack is never held in memory in full.
    """
    target_folder.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = target_folder / uploaded_file.name
        with file_path.open("wb") as output_file:
            while True:
                chunk = uploaded_file.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
        saved_paths.append(file_path)
    return saved_paths


def inject_custom_css():
    """Inject the HIVE colour scheme and light/dark-aware component styling.

    Overrides Streamlit's defaults with the HIVE palette (yellow, blue, red)
    and adds ``prefers-color-scheme`` rules so text stays legible in both
    themes. Purely cosmetic -- nothing here affects analysis.
    """
    st.markdown(
        """
        <style>
        :root {
            --hive-yellow: #f9b410;
            --hive-blue: #0d2942;
            --hive-red: #e80c54;
            --hive-bg-light: #ffffff;
            --hive-bg-dark: #060606;
            --hive-dark: #000000;
            --hive-light: #ffffff;
            --hive-text-light: #111111;
            --hive-text-dark: #ffffff;
            --hive-muted: #4b5563;
            --panel-bg: #ffffff;
            --panel-border: #dfe6ee;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp, .block-container {
            background-color: var(--hive-bg-light) !important;
            color: var(--hive-text-light) !important;
        }

        @media (prefers-color-scheme: dark) {
            html, body, [data-testid="stAppViewContainer"], .stApp, .block-container {
                background-color: var(--hive-bg-dark) !important;
                color: var(--hive-text-dark) !important;
            }
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--hive-dark) 0%, #1a1a1a 100%);
            border-right: 1px solid rgba(255,255,255,0.14);
            color: var(--hive-light);
        }

        [data-testid="stSidebar"] * {
            color: var(--hive-light) !important;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            background: linear-gradient(135deg, var(--panel-bg) 0%, #f4f8fc 100%);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
            overflow: visible;
        }

        @media (prefers-color-scheme: dark) {
            .hero-card {
                background: linear-gradient(135deg, #ffffff 0%, #f2f4f7 100%);
                border-color: #dfe6ee;
            }
        }

        .hero-card h1 {
            margin-bottom: 0.25rem;
            color: #000000;
            font-weight: 700;
            line-height: 1.2;
            white-space: normal;
        }

        .hero-card p {
            margin: 0;
            color: #111111;
            font-size: 1rem;
            line-height: 1.4;
            white-space: normal;
        }

        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, var(--hive-yellow) 0%, #ffd25c 100%);
            color: var(--hive-dark);
            border: 1px solid #c98a00;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.5rem 1rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stDownloadButton > button {
            background: linear-gradient(90deg, var(--hive-red) 0%, #ff4f84 100%);
            color: var(--hive-light);
            border-color: #b6003f;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(13, 41, 66, 0.16);
        }

        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input,
        .stTextArea > textarea,
        .stNumberInput input,
        .stTextInput input {
            border: 1px solid #b7b7b7;
            border-radius: 8px;
            background-color: var(--hive-light);
            color: var(--hive-text-light);
        }

        @media (prefers-color-scheme: dark) {
            .stTextInput > div > div > input,
            .stSelectbox > div > div > div,
            .stNumberInput > div > div > input,
            .stTextArea > textarea,
            .stNumberInput input,
            .stTextInput input {
                background-color: #111111;
                color: var(--hive-text-dark);
                border-color: #777777;
            }
        }

        label, .stMarkdown, .stAlert, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label, .stRadio label, .stCheckbox label, [data-testid="stMetric"], .stDataFrame, .stDataFrame * {
            color: #111111 !important;
        }

        @media (prefers-color-scheme: dark) {
            label, .stMarkdown, .stAlert, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label, .stRadio label, .stCheckbox label, [data-testid="stMetric"], .stDataFrame, .stDataFrame * {
                color: #ffffff !important;
            }
        }

        h1, h2, h3 {
            color: inherit;
        }

        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #d8d8d8;
            border-radius: 12px;
            color: #111111;
        }

        @media (prefers-color-scheme: dark) {
            [data-testid="stMetric"] {
                background-color: #111111;
                border-color: #444444;
                color: #ffffff;
            }
        }

        .stAlert {
            background-color: #f3f4f6;
            border: 1px solid #d1d5db;
            color: #111111;
        }

        @media (prefers-color-scheme: dark) {
            .stAlert {
                background-color: #111111;
                border-color: #444444;
                color: #ffffff;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_zip_for_folder(folder_path: Path) -> Path:
    """Archive a results folder into a ZIP in the system temp directory.

    Args:
        folder_path: Directory to archive.

    Returns:
        Path to the created ``.zip``.

    Note:
        The archive name is derived only from the folder name, so repeated
        runs writing to the same output folder overwrite the same temp
        archive. Nothing cleans these up; the OS temp sweeper does.
    """
    archive_root = Path(tempfile.gettempdir())
    archive_name = archive_root / f"{folder_path.name}_results"
    archive_path = shutil.make_archive(str(archive_name), "zip", root_dir=folder_path)
    return Path(archive_path)


def load_image_for_display(image_path: Path):
    """Read an image for preview, returning ``None`` instead of raising.

    Args:
        image_path: Image to read.

    Returns:
        The image array, or ``None`` if it could not be read. A preview that
        fails to load should not take down the results page, so the broad
        ``except`` here is deliberate.
    """
    try:
        return imread(str(image_path))
    except Exception:
        return None


def display_output_images(output_folder: Path, prefix: str, labels: list[str]):
    """Render the first output image matching each label suffix.

    Args:
        output_folder: Folder to search.
        prefix: Filename prefix; ``""`` matches any.
        labels: Filename suffixes to look for, in display order.

    Note:
        Takes only the *first* glob match per label, so in a batch run of many
        images this shows one representative preview rather than all of them.
        Missing files are skipped silently.
    """
    for label in labels:
        path = next(output_folder.glob(f"{prefix}*{label}"), None)
        if path and path.exists():
            image = load_image_for_display(path)
            if image is not None:
                st.image(image, caption=path.name, width=500)


def show_2d_results(output_folder: Path, measurements, summary):
    """Render 2D results: tables, preview images and a ZIP download button.

    Args:
        output_folder: Folder the pipeline wrote to.
        measurements: Combined per-object table, or ``None``.
        summary: Combined per-image summary, or ``None``.

    Note:
        The preview images are contrast-stretched per file by
        :func:`src.io_utils.save_uint8`. Read the CSV for real values.
    """
    st.success("2D analysis complete.")
    st.write(f"Results saved to: `{output_folder}`")

    if summary is not None:
        st.subheader("Summary")
        st.dataframe(summary)

    if measurements is not None:
        st.subheader("Combined measurements")
        st.dataframe(measurements)

    st.subheader("Example output images")
    cols = st.columns(2)
    with cols[0]:
        display_output_images(output_folder, "", ["boundary_overlay.png", "label_overlay.png", "mask.png", "skeleton.png"])
    with cols[1]:
        display_output_images(output_folder, "", ["preprocessed.png", "gray_channel.png", "max_projection.png", "mask_projection.png"])

    archive_path = build_zip_for_folder(output_folder)
    with open(archive_path, "rb") as f:
        st.download_button(
            label="Download results ZIP",
            data=f,
            file_name=archive_path.name,
            mime="application/zip",
        )


def show_3d_results(output_folder: Path, measurements):
    """Render 3D results: measurements, projections and a ZIP download button.

    Args:
        output_folder: Folder the pipeline wrote to.
        measurements: Value returned by :func:`src.reconstruction_3d.analyze_zstack`.

    Warning:
        Two known defects in this legacy view, left unfixed here so 1.x
        behaviour is reproducible:

        1. ``analyze_zstack`` returns a ``(measurements, summary)`` tuple, and
           :func:`main` passes that tuple through unpacked. ``st.dataframe``
           therefore receives a tuple rather than a DataFrame.
        2. The glob below looks for ``*_3d_view.html``, but the pipeline
           writes ``HIVE_Amira_Imaris_style_3D_viewer.html``. The names do not
           match, so the interactive-mesh link never appears. The file is
           still present in the results folder and in the ZIP.
    """
    st.success("3D analysis complete.")
    st.write(f"Results saved to: `{output_folder}`")

    if measurements is not None:
        st.subheader("Object measurements")
        st.dataframe(measurements)

    st.subheader("3D preview images")
    display_output_images(output_folder, "", ["max_projection.png", "mask_projection.png"])

    # Never matches the pipeline's actual filename -- see the warning above.
    html_path = next(output_folder.glob("*_3d_view.html"), None)
    if html_path and html_path.exists():
        st.markdown(f"[Open interactive 3D mesh view]({html_path})")

    archive_path = build_zip_for_folder(output_folder)
    with open(archive_path, "rb") as f:
        st.download_button(
            label="Download results ZIP",
            data=f,
            file_name=archive_path.name,
            mime="application/zip",
        )


def main():
    """Build the sidebar controls and dispatch a 2D or 3D run.

    Layout: shared controls (channel, sizes, threshold scale, output folder)
    in the sidebar, then a mode-specific block for 2D folders or 3D z-stacks.
    Input can come from a local path or an upload; uploads are staged into a
    fresh temp directory before analysis.

    Note:
        The 3D branch assigns the ``(measurements, summary)`` tuple returned
        by :func:`src.reconstruction_3d.analyze_zstack` to a single name --
        see the warning on :func:`show_3d_results`.

        The 3D branch also drops ``gaussian_kernel`` and ``mesh_step_size``,
        so those fall back to pipeline defaults regardless of the sidebar.
    """
    st.set_page_config(page_title="HIVE Streamlit UI", layout="wide")
    inject_custom_css()
    st.caption(f"Up to {MAX_UPLOAD_SIZE_MB // 1024} GB per file • PNG, JPG, TIF")

    st.markdown(
        """
        <div class="hero-card">
            <h1>HIVE Image Analysis</h1>
            <p>Run polished 2D and 3D segmentation, measurement, and reporting workflows from a professional browser interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.sidebar.selectbox("Analysis mode", ["2D image folder", "3D z-stack"])
    channel = st.sidebar.selectbox("Image channel", ["auto", "red", "green", "blue", "magenta", "gray"])
    min_size = st.sidebar.number_input("Minimum object size", min_value=1, value=50)
    hole_size = st.sidebar.number_input("Maximum hole size to fill", min_value=0, value=50)
    threshold_scale = st.sidebar.slider(
        "Threshold scale", min_value=0.1, max_value=3.0, value=1.0, step=0.1
    )

    output_folder = Path(st.sidebar.text_input("Output folder", "output_results")).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    if mode == "2D image folder":
        st.sidebar.markdown("### 2D settings")
        method = st.sidebar.selectbox("Segmentation method", ["threshold", "cellpose"])
        gaussian_kernel = st.sidebar.number_input(
            "Gaussian blur kernel size", min_value=0, value=5, step=2
        )

        input_mode = st.sidebar.radio("2D input source", ["Local folder", "Upload images"])
        config = HiveConfig(
            input_folder="input_images",
            output_folder=str(output_folder),
            channel=channel,
            method=method,
            min_size=min_size,
            hole_size=hole_size,
            threshold_scale=threshold_scale,
            gaussian_kernel=gaussian_kernel,
        )

        if input_mode == "Local folder":
            config.input_folder = st.text_input("Local input folder", "input_images")
        else:
            uploaded_images = st.file_uploader(
                "Upload images to analyze",
                type=SUPPORTED_IMAGE_TYPES,
                accept_multiple_files=True,
                help="8GB per file • PNG, JPG, TIF",
            )
            if uploaded_images:
                upload_folder = Path(tempfile.mkdtemp(prefix="hive_upload_"))
                save_uploaded_files(uploaded_images, upload_folder)
                config.input_folder = str(upload_folder)
                st.info(f"Saved {len(uploaded_images)} uploaded files to {upload_folder}")

        if st.button("Run 2D analysis"):
            if not Path(config.input_folder).exists():
                st.error(f"Input folder not found: {config.input_folder}")
            else:
                measurements, summary = analyze_folder(config)
                show_2d_results(output_folder, measurements, summary)

    else:
        st.sidebar.markdown("### 3D settings")
        voxel_size_xy = st.sidebar.number_input("Voxel size XY", min_value=0.001, value=1.0, format="%.3f")
        voxel_size_z = st.sidebar.number_input("Voxel size Z", min_value=0.001, value=1.0, format="%.3f")

        input_mode = st.sidebar.radio("3D input source", ["Local file/folder", "Upload multi-page TIFF", "Upload slice images"])
        zstack_path = st.text_input("Local z-stack path", "input_zstacks")
        uploaded_stack = None
        uploaded_slices = None

        if input_mode == "Upload multi-page TIFF":
            uploaded_stack = st.file_uploader(
                "Upload a multi-page TIFF z-stack",
                type=["tif", "tiff"],
                accept_multiple_files=False,
            )
        elif input_mode == "Upload slice images":
            uploaded_slices = st.file_uploader(
                "Upload individual slice images",
                type=SUPPORTED_IMAGE_TYPES,
                accept_multiple_files=True,
            )

        if st.button("Run 3D analysis"):
            if uploaded_stack is not None:
                temp_file = Path(tempfile.mkdtemp(prefix="hive_zstack_")) / uploaded_stack.name
                with temp_file.open("wb") as output_file:
                    while True:
                        chunk = uploaded_stack.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                zstack_path = str(temp_file)
                st.info(f"Saved uploaded z-stack to {temp_file}")
            elif uploaded_slices is not None and len(uploaded_slices) > 0:
                temp_folder = Path(tempfile.mkdtemp(prefix="hive_zstack_slices_"))
                save_uploaded_files(uploaded_slices, temp_folder)
                zstack_path = str(temp_folder)
                st.info(f"Saved {len(uploaded_slices)} uploaded slices to {temp_folder}")

            if not Path(zstack_path).exists():
                st.error(f"Z-stack path not found: {zstack_path}")
            else:
                measurements = analyze_zstack(
                    zstack_path=zstack_path,
                    output_folder=output_folder,
                    channel=channel,
                    threshold_scale=threshold_scale,
                    min_size=min_size,
                    hole_size=hole_size,
                    voxel_size_xy=voxel_size_xy,
                    voxel_size_z=voxel_size_z,
                )
                show_3d_results(output_folder, measurements)


if __name__ == "__main__":
    main()