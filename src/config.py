"""LEGACY (HIVE 1.x) run configuration.

.. deprecated:: 2.0
    Superseded by :class:`hive.config.HiveConfig`, which groups parameters
    into nested ``preprocess``/``segmentation``/``arbor``/``render`` sections
    and validates them on construction. This flat dataclass performs **no
    validation**: a negative ``min_size`` or a zero ``voxel_size_xy`` is
    accepted here and only fails (or silently corrupts results) further down.

    Retained solely so 1.x outputs can be regenerated and compared against
    2.0. See "Migrating from ``src/``" in the README.
"""

from dataclasses import dataclass


@dataclass
class HiveConfig:
    """Flat parameter bag shared by the legacy 2D and 3D entry points.

    Note:
        ``threshold_scale`` defaults to ``1.2`` here, **not** to ``1.0``. That
        multiplier is an empirical knob applied on top of Otsu's threshold, so
        the legacy default silently reports a more conservative segmentation
        than unmodified Otsu (1979). :class:`hive.config.SegmentationConfig`
        changed the default to ``1.0`` for exactly this reason -- object counts
        from 1.x and 2.0 are therefore not directly comparable unless this is
        set to match.
    """

    input_folder: str = "input_images"
    zstack_path: str = "input_zstacks"
    output_folder: str = "output_results"
    mode: str = "2d"
    channel: str = "auto"  # auto, red, green, blue, magenta, gray
    method: str = "threshold"  # threshold, cellpose
    min_size: int = 80
    hole_size: int = 50
    threshold_scale: float = 1.2
    gaussian_kernel: int = 3
    save_intermediate: bool = True

    # Cellpose options
    cellpose_model: str = "cyto3"
    cellpose_diameter: float = 0.0
    cellpose_cellprob: float = 0.0
    cellpose_flow: float = 0.4
    cellpose_gpu: bool = False

    # PyImageJ/Fiji options. Kept optional because Java/Maven can be unstable.
    use_pyimagej: bool = False
    fiji_path: str = "/Applications/Fiji"

    # 3D options
    voxel_size_xy: float = 1.0
    voxel_size_z: float = 1.0
    mesh_step_size: int = 2
    mesh_smoothing_sigma: float = 1.0
