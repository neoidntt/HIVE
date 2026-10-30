#!/usr/bin/env python3
"""HIVE command-line interface.

Replaces the legacy ``hive_app.py``, which built a ``HiveConfig`` and then
ignored it in 3D mode, passing a hand-copied argument list to ``analyze_zstack``
instead -- so ``--no-intermediate`` and several other flags silently did nothing
for volumes. Here every mode consumes the same config object.

Examples:
    2D batch over a folder::

        python hive_cli.py --mode 2d --input input_images --channel magenta

    3D reconstruction with a rotating GIF and real voxel sizes::

        python hive_cli.py --mode 3d --input input_zstacks/my_stack \\
            --voxel-size-xy 0.2 --voxel-size-z 1.0 --unit um --gif

    Report physical units and drop edge-truncated objects::

        python hive_cli.py --mode 2d --input imgs --voxel-size-xy 0.325 \\
            --unit um --clear-border

    Repeat an earlier run exactly from the record it saved::

        python hive_cli.py --settings old_results/hive_run.json --output rerun

    Measure a second channel and compare two conditions::

        python hive_cli.py --input scans --channel DAPI --extra-channels GFP \\
            --conditions conditions.csv

    Look at what a file contains, or compare background methods, before a run::

        python hive_cli.py --inspect scan.nd2
        python hive_cli.py --preview-baseline scan.ome.tif preview.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hive.config import (
    ArborConfig, ExportConfig, HiveConfig, LargeImageConfig, PreprocessConfig, RenderConfig,
    SegmentationConfig,
)
from hive.formats import is_microscope_file, read_microscope
from hive.hardware import describe_hardware
from hive.io_utils import ImageLoadError, load_image
from hive.provenance import environment_differences, read_run_record
from hive.stats import load_conditions


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="HIVE: High-Processing Imaging Visualization Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="2d", choices=["2d", "3d"])
    parser.add_argument("--input", default=None,
                        help="Image folder (2D) or stack file/folder (3D) [input_images]")
    parser.add_argument("--output", default=None, help="Results folder [output_results]")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the steps that use one; recorded in hive_run.json")
    parser.add_argument("--channel", default="auto",
                        help="auto, a colour (red, green, blue, magenta, gray) for RGB images, "
                             "or a channel name/number for OME-TIFF, CZI, ND2 and LIF files")
    parser.add_argument("--extra-channels", nargs="+", default=[], metavar="CHANNEL",
                        help="Further channels to measure at the objects, with colocalization "
                             "against --channel")
    parser.add_argument("--settings", metavar="FILE",
                        help="Load every analysis setting from a hive_run.json (or its folder); "
                             "only --input, --output, --conditions, --workers and --gif may "
                             "accompany it")
    parser.add_argument("--conditions", metavar="CSV",
                        help="CSV with columns image,condition: compare conditions in the report")

    scale = parser.add_argument_group("physical scale")
    scale.add_argument("--voxel-size-xy", type=float, default=1.0,
                       help="Physical pixel size in XY (leave 1.0 to report pixels)")
    scale.add_argument("--voxel-size-z", type=float, default=1.0, help="Physical z-step (3D)")
    scale.add_argument("--unit", default="px", help="Unit label, e.g. um")

    pre = parser.add_argument_group("preprocessing")
    pre.add_argument("--gaussian-sigma", type=float, default=1.0,
                     help="Gaussian sigma for the segmentation image (0 disables)")
    pre.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    pre.add_argument("--baseline", default="tophat",
                     choices=["none", "percentile", "smooth", "tophat"],
                     help="Background estimator")
    pre.add_argument("--tophat-radius", type=float, default=25.0,
                     help="Top-hat radius; must exceed the largest real feature")
    pre.add_argument("--tophat-radius-z", type=float, default=None,
                     help="Separate z radius for anisotropic stacks (3D)")

    seg = parser.add_argument_group("segmentation")
    seg.add_argument("--method", default="threshold", choices=["threshold", "cellpose"])
    seg.add_argument("--threshold-scale", type=float, default=1.0,
                     help="Otsu multiplier; 1.0 is unmodified Otsu")
    seg.add_argument("--min-size", type=int, default=80, help="Debris threshold (px/voxels)")
    seg.add_argument("--max-size", type=int, default=0, help="Reject objects above this (0 = off)")
    seg.add_argument("--hole-size", type=int, default=50, help="Fill holes up to this size")
    seg.add_argument("--clear-border", action="store_true",
                     help="Drop objects touching the image edge")
    seg.add_argument("--watershed", action="store_true",
                     help="Split touching convex objects (leave off for branched cells)")
    seg.add_argument("--cellpose-model", default="cyto3")
    seg.add_argument("--cellpose-diameter", type=float, default=0.0)
    seg.add_argument("--cellpose-cellprob", type=float, default=0.0)
    seg.add_argument("--cellpose-flow", type=float, default=0.4)

    arb = parser.add_argument_group("arborization")
    arb.add_argument("--no-arbor", action="store_true", help="Skip arborisation/Sholl analysis")
    arb.add_argument("--sholl-step", type=float, default=5.0, help="Sholl shell spacing")
    arb.add_argument("--sholl-center", default="centroid", choices=["centroid", "max_intensity"])
    arb.add_argument("--prune-length", type=int, default=3, help="Remove spurs below this length")

    ren = parser.add_argument_group("3D rendering")
    ren.add_argument("--gif", action="store_true", help="Render a rotating projection GIF (3D)")
    ren.add_argument("--gif-mode", default="mip", choices=["mip", "alpha"])
    ren.add_argument("--gif-frames", type=int, default=36)
    ren.add_argument("--gif-fps", type=int, default=15)
    ren.add_argument("--gif-colormap", default="magma")
    ren.add_argument("--gif-downsample", type=int, default=2)
    ren.add_argument("--mesh-step-size", type=int, default=2)

    big = parser.add_argument_group("large stacks and exports")
    big.add_argument("--tiling", default="auto", choices=["auto", "off", "force"],
                     help="auto: process a 3D stack in tiles only if it will not fit in RAM")
    big.add_argument("--tile-size", type=int, default=0, help="Tile edge in px (0 = automatic)")
    big.add_argument("--halo", type=int, default=0,
                     help="Extra pixels read around each tile (0 = automatic)")
    big.add_argument("--no-ome-tiff", action="store_true", help="Do not write the OME-TIFF labels")
    big.add_argument("--no-rois", action="store_true", help="Do not write the ImageJ ROI zip")

    tools = parser.add_argument_group("inspection")
    tools.add_argument("--inspect", metavar="FILE",
                       help="Print a file's channels, size and pixel size, then exit")
    tools.add_argument("--preview-baseline", nargs=2, metavar=("IMAGE", "PNG"),
                       help="Save a picture comparing the four background methods, then exit")

    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for 2D batches. Each worker needs its "
                             "own copy of the working set, so workers x per-image peak must fit "
                             "in RAM; HIVE warns if it will not.")
    parser.add_argument("--no-intermediate", action="store_true",
                        help="Skip QC images of the measurement/segmentation pair")
    parser.add_argument("--hardware", action="store_true",
                        help="Print detected compute hardware and exit")
    parser.add_argument("--quiet", action="store_true", help="Warnings and errors only")
    return parser


def config_from_args(args: argparse.Namespace) -> HiveConfig:
    """Translate parsed arguments into a :class:`HiveConfig`.

    Args:
        args: Parsed arguments.

    Returns:
        The populated configuration.

    Raises:
        ValueError: If any parameter is outside its physical domain.
    """
    tophat: float | tuple[float, ...] = args.tophat_radius
    if args.mode == "3d":
        # A scalar radius applied to a stack blurs z as hard as xy, which is
        # almost never right; default z to a tenth of the xy radius.
        z_radius = args.tophat_radius_z if args.tophat_radius_z is not None else max(
            1.0, args.tophat_radius / 10.0
        )
        tophat = (z_radius, args.tophat_radius, args.tophat_radius)

    return HiveConfig(
        input_path=args.input or "input_images",
        output_folder=args.output or "output_results",
        mode=args.mode,
        channel=args.channel,
        extra_channels=list(args.extra_channels),
        conditions=load_conditions(args.conditions) if args.conditions else {},
        seed=args.seed,
        voxel_size_xy=args.voxel_size_xy,
        voxel_size_z=args.voxel_size_z,
        voxel_unit=args.unit,
        save_intermediate=not args.no_intermediate,
        mesh_step_size=args.mesh_step_size,
        preprocess=PreprocessConfig(
            gaussian_sigma=args.gaussian_sigma,
            use_clahe=not args.no_clahe,
            baseline_method=args.baseline,
            tophat_radius=tophat,
        ),
        segmentation=SegmentationConfig(
            method=args.method,
            threshold_scale=args.threshold_scale,
            min_size=args.min_size,
            max_size=args.max_size,
            hole_size=args.hole_size,
            clear_border=args.clear_border,
            watershed_split=args.watershed,
            cellpose_model=args.cellpose_model,
            cellpose_diameter=args.cellpose_diameter,
            cellpose_cellprob=args.cellpose_cellprob,
            cellpose_flow=args.cellpose_flow,
        ),
        arbor=ArborConfig(
            enabled=not args.no_arbor,
            sholl_step=args.sholl_step,
            sholl_center=args.sholl_center,
            prune_length=args.prune_length,
        ),
        render=RenderConfig(
            mode=args.gif_mode,
            n_frames=args.gif_frames,
            fps=args.gif_fps,
            colormap=args.gif_colormap,
            downsample=args.gif_downsample,
        ),
        export=ExportConfig(
            ome_tiff_labels=not args.no_ome_tiff, imagej_rois=not args.no_rois,
        ),
        large_image=LargeImageConfig(
            mode=args.tiling, tile_size=args.tile_size, halo=args.halo,
        ),
    )


# Flags that may be combined with --settings; everything else comes from the file.
_WITH_SETTINGS = {"settings", "input", "output", "conditions", "workers", "gif",
                  "quiet", "hardware", "inspect", "preview_baseline"}


def config_with_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> HiveConfig:
    """Load ``--settings`` and apply the few flags allowed alongside it.

    Args:
        args: Parsed arguments.
        parser: The parser, used to tell which flags the user actually set.

    Returns:
        The configuration from the saved run.

    Raises:
        ValueError: If the settings file holds parameters this version does not know.
        FileNotFoundError: If the file does not exist.
    """
    saved = read_run_record(args.settings)
    defaults = parser.parse_args([])
    ignored = sorted(k for k in vars(args)
                     if k not in _WITH_SETTINGS and getattr(args, k) != getattr(defaults, k))
    if ignored:
        print("Note: --settings supplies these, so the flags were ignored: "
              + ", ".join("--" + k.replace("_", "-") for k in ignored), file=sys.stderr)
    for line in environment_differences(saved.record):
        print(f"Note: library differs from the saved run: {line}", file=sys.stderr)
    cfg = saved.config
    if args.input:
        cfg.input_path = args.input
    if args.output:
        cfg.output_folder = args.output
    if args.conditions:
        cfg.conditions = load_conditions(args.conditions)
    return cfg


def inspect_file(path: str) -> int:
    """Print what a microscope file contains."""
    meta = read_microscope(path, load_pixels=False)
    c, z, y, x = meta.shape
    print(f"{path}\n  format        {meta.source_format}\n  size          "
          f"{c} channel(s) x {z} plane(s) x {y} x {x} px")
    for i, name in enumerate(meta.channel_names):
        print(f"  channel {i}     {name}")
    scale = (f"{meta.voxel_size_xy:.4g} um" if meta.voxel_size_xy else "not in the file")
    step = (f"{meta.voxel_size_z:.4g} um" if meta.voxel_size_z else "not in the file")
    print(f"  pixel size    {scale}\n  z step        {step}")
    for note in meta.notes:
        print(f"  note          {note}")
    return 0


def preview_baseline(image_path: str, png_path: str, args: argparse.Namespace) -> int:
    """Save a picture of the four background methods for one image."""
    from hive.baseline_preview import center_crop, preview_figure, preview_methods
    from hive.pipeline import _channel_choice
    from hive.preprocessing import select_channel

    path = Path(image_path)
    if is_microscope_file(path):
        meta = read_microscope(path)
        plane = meta.volume(_channel_choice(meta, HiveConfig(channel=args.channel)))
        plane = plane[plane.shape[0] // 2]
    else:
        plane = select_channel(load_image(path), args.channel)[0]  # type: ignore[arg-type]
    pre = PreprocessConfig(
        baseline_method=args.baseline, tophat_radius=args.tophat_radius,
    )
    figure = preview_figure(center_crop(plane), preview_methods(plane, pre))
    figure.savefig(png_path, dpi=130)
    print(f"Saved {png_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 1 on a handled user-facing error.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.hardware:
        print(describe_hardware())
        return 0

    try:
        if args.inspect:
            return inspect_file(args.inspect)
        if args.preview_baseline:
            return preview_baseline(*args.preview_baseline, args)
        cfg = (config_with_settings(args, build_parser()) if args.settings
               else config_from_args(args))
    except (ValueError, FileNotFoundError, ImageLoadError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        if cfg.mode == "3d":
            from hive.pipeline import analyze_zstack

            result = analyze_zstack(cfg, render_gif=args.gif)
            print(f"\n3D reconstruction complete: {result.object_count} objects")
            print(f"  {result.cleanup.summary()}")
            for kind, path in result.outputs.items():
                print(f"  {kind:18s} {path}")
        else:
            from hive.pipeline import analyze_folder

            out = analyze_folder(cfg, workers=args.workers)
            measurements = out["measurements"]
            print(f"\n2D analysis complete: {len(measurements)} objects "
                  f"across {len(out['summary'])} images")
            if out["errors"]:
                print(f"  {len(out['errors'])} image(s) FAILED:", file=sys.stderr)
                for name, err in out["errors"]:
                    print(f"    {name}: {err}", file=sys.stderr)
            print(f"  results in {Path(cfg.output_folder).resolve()}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
