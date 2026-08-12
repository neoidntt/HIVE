"""LEGACY (HIVE 1.x) command-line entry point.

.. deprecated:: 2.0
    Superseded by ``hive_cli.py``, which exposes physical voxel units,
    arborization/Sholl analysis, hardware detection and the full
    :class:`hive.config.HiveConfig`. Use::

        python hive_cli.py --mode 2d --input input_images

This script drives the legacy :mod:`src` pipeline and inherits its caveats:
intensities are contrast-enhanced rather than quantitative, ``threshold_scale``
defaults to ``1.2`` instead of true Otsu, and the 3D path rejects channel-last
RGB stacks. Retained only to regenerate 1.x outputs for comparison.
"""

import argparse
from src.config import HiveConfig
from src.pipeline import analyze_folder
from src.reconstruction_3d import analyze_zstack
from src.pyimagej_bridge import pyimagej_status


def parse_args():
    """Define and parse the legacy command-line interface.

    Returns:
        Populated :class:`argparse.Namespace`.

    Note:
        2D and 3D take their input from *different* flags -- ``--input`` for
        2D image folders, ``--zstack`` for 3D stacks -- so passing ``--input``
        with ``--mode 3d`` is silently ignored.
    """
    parser = argparse.ArgumentParser(description="HIVE: High-Processing Imaging Visualization Engine")
    parser.add_argument("--mode", default="2d", choices=["2d", "3d"], help="Use 2d for images or 3d for z-stacks.")
    parser.add_argument("--input", default="input_images", help="Input image folder for 2D mode")
    parser.add_argument("--zstack", default="input_zstacks", help="Z-stack folder or multi-page TIFF for 3D mode")
    parser.add_argument("--output", default="output_results", help="Output result folder")
    parser.add_argument("--channel", default="auto", choices=["auto", "red", "green", "blue", "magenta", "gray"], help="Image channel to analyze")
    parser.add_argument("--method", default="threshold", choices=["threshold", "cellpose"], help="2D segmentation method")
    parser.add_argument("--min-size", type=int, default=80, help="Minimum object size in pixels/voxels")
    parser.add_argument("--hole-size", type=int, default=50, help="Maximum hole size to fill")
    parser.add_argument("--threshold-scale", type=float, default=1.2, help="Scale Otsu threshold")
    parser.add_argument("--gaussian-kernel", type=int, default=3, help="Gaussian blur kernel size")
    parser.add_argument("--no-intermediate", action="store_true", help="Do not save preprocessed/channel intermediate files")

    parser.add_argument("--cellpose-model", default="cyto3", help="Cellpose model: cyto3, cyto2, cyto, nuclei")
    parser.add_argument("--cellpose-diameter", type=float, default=0.0, help="Cellpose diameter. Use 0 to estimate")
    parser.add_argument("--cellpose-cellprob", type=float, default=0.0, help="Cellpose cell probability threshold")
    parser.add_argument("--cellpose-flow", type=float, default=0.4, help="Cellpose flow threshold")
    parser.add_argument("--cellpose-gpu", action="store_true", help="Use GPU for Cellpose if available")

    parser.add_argument("--pyimagej-check", action="store_true", help="Check optional PyImageJ/Fiji availability")
    parser.add_argument("--fiji-path", default="/Applications/Fiji", help="Path to Fiji folder")

    parser.add_argument("--voxel-size-xy", type=float, default=1.0, help="Physical XY pixel size for 3D mode")
    parser.add_argument("--voxel-size-z", type=float, default=1.0, help="Physical Z step size for 3D mode")
    parser.add_argument("--mesh-step-size", type=int, default=2, help="Marching cubes step size for 3D mesh")
    return parser.parse_args()


def main():
    """Dispatch to the PyImageJ probe, the 3D reconstruction, or the 2D batch.

    Note:
        The ``config`` object below is built for both modes but is used only
        by the 2D branch -- ``analyze_zstack`` predates the config object and
        takes its parameters as explicit keyword arguments. Any 3D parameter
        added to :class:`src.config.HiveConfig` alone is therefore ignored.
        Eliminating exactly this duplication is why
        :class:`hive.config.HiveConfig` exists.
    """
    args = parse_args()

    if args.pyimagej_check:
        ok, message = pyimagej_status(args.fiji_path)
        print("PyImageJ available:" if ok else "PyImageJ not available:")
        print(message)
        return

    config = HiveConfig(
        input_folder=args.input,
        zstack_path=args.zstack,
        output_folder=args.output,
        mode=args.mode,
        channel=args.channel,
        method=args.method,
        min_size=args.min_size,
        hole_size=args.hole_size,
        threshold_scale=args.threshold_scale,
        gaussian_kernel=args.gaussian_kernel,
        save_intermediate=not args.no_intermediate,
        cellpose_model=args.cellpose_model,
        cellpose_diameter=args.cellpose_diameter,
        cellpose_cellprob=args.cellpose_cellprob,
        cellpose_flow=args.cellpose_flow,
        cellpose_gpu=args.cellpose_gpu,
        fiji_path=args.fiji_path,
        voxel_size_xy=args.voxel_size_xy,
        voxel_size_z=args.voxel_size_z,
        mesh_step_size=args.mesh_step_size,
    )

    if args.mode == "3d":
        analyze_zstack(
            zstack_path=args.zstack,
            output_folder=args.output,
            channel=args.channel,
            threshold_scale=args.threshold_scale,
            min_size=args.min_size,
            hole_size=args.hole_size,
            voxel_size_xy=args.voxel_size_xy,
            voxel_size_z=args.voxel_size_z,
            gaussian_kernel=args.gaussian_kernel,
            mesh_step_size=args.mesh_step_size,
        )
        return

    analyze_folder(config)


if __name__ == "__main__":
    main()
