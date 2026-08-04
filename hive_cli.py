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
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hive.config import ArborConfig, HiveConfig, PreprocessConfig, RenderConfig, SegmentationConfig
from hive.hardware import describe_hardware


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="HIVE: High-Precision Image Visualization Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="2d", choices=["2d", "3d"])
    parser.add_argument("--input", default="input_images",
                        help="Image folder (2D) or stack file/folder (3D)")
    parser.add_argument("--output", default="output_results")
    parser.add_argument("--channel", default="auto",
                        choices=["auto", "red", "green", "blue", "magenta", "gray"])

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
        input_path=args.input,
        output_folder=args.output,
        mode=args.mode,
        channel=args.channel,
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
    )


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
        cfg = config_from_args(args)
    except ValueError as exc:
        print(f"Invalid parameters: {exc}", file=sys.stderr)
        return 1

    try:
        if args.mode == "3d":
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
