"""LEGACY (HIVE 1.x) 2D analysis orchestration.

.. deprecated:: 2.0
    Superseded by :mod:`hive.pipeline`, whose ``analyze_folder`` returns a
    dict and whose ``analyze_zstack`` returns an ``AnalysisResult``.

Pipeline order::

    load -> select_channel -> preprocess (normalise + blur + CLAHE)
         -> threshold/cellpose -> label -> measure -> report

The defect: the enhanced image produced by ``preprocess`` is used both to
*find* objects and to *measure* their intensity. HIVE 2.0 splits these into a
segmentation image and a measurement image so enhancement can never reach a
reported number.
"""

from pathlib import Path
import pandas as pd
from .config import HiveConfig
from .preprocessing import load_image, select_channel, preprocess, ensure_2d
from .segmentation import threshold_segmentation, label_mask
from .cellpose_segmentation import segment_with_cellpose
from .measurements import measure_objects, summarize_image, skeletonize_mask
from .visualization import create_boundary_overlay, create_label_overlay
from .excel_report import export_excel_report
from .io_utils import save_uint8, collect_image_paths


def analyze_single_image(image_path, config: HiveConfig):
    """Analyse one image and write its CSV, Excel and preview outputs.

    Args:
        image_path: Path to the image to analyse.
        config: Legacy :class:`src.config.HiveConfig`.

    Returns:
        ``(measurements, summary)`` -- the per-object table and its one-row
        per-image summary.

    Raises:
        ValueError: If ``config.method`` is neither ``threshold`` nor
            ``cellpose``.

    Warning:
        ``intensity_2d`` below is the *preprocessed* (min-max normalised,
        CLAHE-equalised) image, and it is passed to :func:`measure_objects` as
        the intensity source. Every intensity column in the outputs of this
        function is therefore contrast-enhanced, per-image-rescaled data, not
        fluorescence. Areas, counts and centroids are unaffected.
    """
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    stem = image_path.stem

    original = load_image(str(image_path))
    channel_img = select_channel(original, config.channel)
    # Single enhanced image, reused below for BOTH detection and measurement.
    # This is the 1.x defect; hive.preprocessing keeps the two separate.
    preprocessed = preprocess(channel_img, config.gaussian_kernel)
    intensity_2d = ensure_2d(preprocessed)

    if config.method == "threshold":
        binary = threshold_segmentation(
            intensity_2d,
            min_size=config.min_size,
            hole_size=config.hole_size,
            threshold_scale=config.threshold_scale,
        )
    elif config.method == "cellpose":
        binary = segment_with_cellpose(
            intensity_2d,
            model_type=config.cellpose_model,
            diameter=config.cellpose_diameter,
            cellprob_threshold=config.cellpose_cellprob,
            flow_threshold=config.cellpose_flow,
            gpu=config.cellpose_gpu,
            min_size=config.min_size,
        )
    else:
        raise ValueError("Unknown method. Use threshold or cellpose.")

    labeled = label_mask(binary)
    measurements = measure_objects(labeled, intensity_2d, image_name=image_path.name)
    summary = summarize_image(measurements, image_name=image_path.name)
    skeleton = skeletonize_mask(binary)

    measurements.to_csv(output_folder / f"{stem}_measurements.csv", index=False)
    summary.to_csv(output_folder / f"{stem}_summary.csv", index=False)
    export_excel_report(
        output_folder / f"{stem}_report.xlsx",
        summary_df=summary,
        measurements_df=measurements,
        settings={
            "mode": "2d",
            "image": image_path.name,
            "channel": config.channel,
            "method": config.method,
            "min_size": config.min_size,
            "hole_size": config.hole_size,
            "threshold_scale": config.threshold_scale,
            "gaussian_kernel": config.gaussian_kernel,
            "cellpose_model": config.cellpose_model if config.method == "cellpose" else "not used",
        },
    )

    save_uint8(output_folder / f"{stem}_mask.png", binary.astype(float))
    save_uint8(output_folder / f"{stem}_skeleton.png", skeleton.astype(float))
    save_uint8(output_folder / f"{stem}_boundary_overlay.png", create_boundary_overlay(original, labeled))
    save_uint8(output_folder / f"{stem}_label_overlay.png", create_label_overlay(intensity_2d, labeled))

    if config.save_intermediate:
        save_uint8(output_folder / f"{stem}_channel.png", ensure_2d(channel_img))
        save_uint8(output_folder / f"{stem}_preprocessed.png", intensity_2d)

    print(f"Analyzed: {image_path.name}")
    print(f"Objects detected: {len(measurements)}")
    return measurements, summary


def analyze_folder(config: HiveConfig):
    """Analyse every supported image in ``config.input_folder``.

    Runs :func:`analyze_single_image` per image, then concatenates the results
    into ``combined_measurements.csv``, ``combined_summary.csv`` and a single
    ``HIVE_2D_report.xlsx``.

    Args:
        config: Legacy :class:`src.config.HiveConfig`.

    Returns:
        ``(combined_measurements, combined_summaries)``, or ``(None, None)``
        if the input folder held no supported images.

    Note:
        A failure on any single image propagates and aborts the whole batch --
        there is no per-image error isolation. The combined workbook's Settings
        sheet also records only five fields, omitting the thresholding and
        size parameters that actually determined the results.
    """
    input_folder = Path(config.input_folder)
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    paths = collect_image_paths(input_folder)
    if not paths:
        print(f"No images found in: {input_folder}")
        return None, None

    all_measurements = []
    all_summaries = []
    for path in paths:
        m, s = analyze_single_image(path, config)
        all_measurements.append(m)
        all_summaries.append(s)

    combined_measurements = pd.concat(all_measurements, ignore_index=True) if all_measurements else pd.DataFrame()
    combined_summaries = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    combined_measurements.to_csv(output_folder / "combined_measurements.csv", index=False)
    combined_summaries.to_csv(output_folder / "combined_summary.csv", index=False)
    export_excel_report(
        output_folder / "HIVE_2D_report.xlsx",
        summary_df=combined_summaries,
        measurements_df=combined_measurements,
        settings={
            "mode": "2d",
            "input_folder": str(input_folder),
            "output_folder": str(output_folder),
            "channel": config.channel,
            "method": config.method,
        },
    )
    print("Done.")
    print(f"Combined files saved in: {output_folder}")
    return combined_measurements, combined_summaries
