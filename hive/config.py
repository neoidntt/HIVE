"""Single source of truth for all HIVE run parameters.

Every entry point (CLI, Streamlit, FastAPI) constructs one :class:`HiveConfig`
and passes it down. The legacy code duplicated the 3D argument list between
``hive_app.py`` and ``analyze_zstack``, so a parameter added in one place was
silently ignored in the other; this module exists to make that impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Literal

Channel = Literal["auto", "red", "green", "blue", "magenta", "gray"]
SegMethod = Literal["threshold", "cellpose"]
BaselineMethod = Literal["none", "percentile", "smooth", "tophat"]


@dataclass
class PreprocessConfig:
    """Controls how the *segmentation* image is built from the raw data.

    These operations are deliberately quarantined from measurement. Gaussian
    smoothing and CLAHE both alter pixel values non-linearly; applying them to
    an image and then calling ``regionprops(intensity_image=...)`` on it yields
    numbers that are not fluorescence.

    Attributes:
        gaussian_sigma: Standard deviation (px) of the Gaussian applied before
            thresholding. ``0`` disables. Suppresses shot noise so Otsu sees a
            cleaner bimodal histogram.
        use_clahe: Apply Contrast Limited Adaptive Histogram Equalisation
            (Zuiderveld, 1994) to the segmentation image. Strongly improves
            detection of dim objects under uneven illumination. Never applied
            to the measurement image.
        clahe_clip_limit: CLAHE clip limit. Higher = more aggressive contrast.
        baseline_method: Background estimator. See :mod:`hive.baseline`.
            ``tophat`` (rolling-ball-family) is the safe default for uneven
            illumination.
        baseline_percentile: Percentile used when ``baseline_method="percentile"``.
        baseline_sigma: Gaussian sigma (scalar or per-axis) for ``"smooth"``.
        tophat_radius: Structuring-element radius (scalar or per-axis) for
            ``"tophat"``. Must exceed the largest real feature, or real signal
            is eaten as background.
        keep_baseline: Retain the estimated background array for QC. It is a
            FULL extra copy of the volume held for the whole run, so this
            defaults to off; enable it only when inspecting what was subtracted.
    """

    gaussian_sigma: float = 1.0
    use_clahe: bool = True
    clahe_clip_limit: float = 0.01
    baseline_method: BaselineMethod = "tophat"
    baseline_percentile: float = 10.0
    baseline_sigma: float | tuple[float, ...] = 20.0
    tophat_radius: float | tuple[float, ...] = 25.0
    keep_baseline: bool = False


@dataclass
class SegmentationConfig:
    """Controls object detection and mask cleanup.

    Attributes:
        method: ``"threshold"`` (Otsu) or ``"cellpose"`` (deep learning).
        threshold_scale: Multiplier on the Otsu threshold. NOTE: this is an
            empirical knob, not a published method. ``1.0`` is true Otsu
            (1979); values >1 are more conservative. Record it when reporting.
        min_size: Objects smaller than this (px/voxels) are discarded as debris.
        max_size: Objects larger than this are discarded (0 = no limit). Useful
            for rejecting merged clumps or imaging artifacts.
        hole_size: Holes smaller than this are filled. Set 0 to keep holes.
        clear_border: Drop objects touching the image edge. These are truncated
            by the field of view, so their area/morphology is unmeasurable.
        watershed_split: Split touching convex objects via distance-transform
            watershed (Beucher & Lantuejoul, 1979).
        cellpose_*: Cellpose parameters, used only when ``method="cellpose"``.
    """

    method: SegMethod = "threshold"
    threshold_scale: float = 1.0
    min_size: int = 80
    max_size: int = 0
    hole_size: int = 50
    clear_border: bool = False
    watershed_split: bool = False

    cellpose_model: str = "cyto3"
    cellpose_diameter: float = 0.0
    cellpose_cellprob: float = 0.0
    cellpose_flow: float = 0.4


@dataclass
class ArborConfig:
    """Controls arborization / Sholl analysis.

    Attributes:
        enabled: Run arborization analysis. Adds skeleton, branch and Sholl
            metrics to the per-object table.
        sholl_step: Radial spacing (px) between Sholl shells (Sholl, 1953).
        sholl_max_radius: Outermost radius (px). ``0`` = auto (object's maximum
            extent from its centre).
        sholl_center: ``"centroid"`` uses the object centroid; ``"max_intensity"``
            uses the brightest point, a better soma proxy for neurons whose
            centroid falls off-soma in a lopsided arbor.
        prune_length: Skeleton spurs shorter than this (px) are removed before
            branch counting. Skeletonisation of a noisy mask creates short
            false twigs that otherwise inflate branch-point counts.
    """

    enabled: bool = True
    sholl_step: float = 5.0
    sholl_max_radius: float = 0.0
    sholl_center: Literal["centroid", "max_intensity"] = "centroid"
    prune_length: int = 3


@dataclass
class RenderConfig:
    """Controls the 3D rotating-projection GIF.

    Attributes:
        mode: ``"mip"`` (maximum intensity projection) or ``"alpha"``
            (front-to-back alpha-blended volume rendering).
        n_frames: Frames over the full rotation. 36 = 10 degrees per frame.
        total_degrees: Total rotation sweep.
        fps: Playback frames per second.
        colormap: Any matplotlib colormap name.
        downsample: Integer factor to shrink the volume before rendering.
            Rotation is the dominant cost and scales with voxel count.
        rotation_axis: ``"y"`` spins about the vertical screen axis (the usual
            turntable); ``"x"`` tumbles top-over-bottom.
        alpha_scale: Opacity multiplier for ``mode="alpha"``.
        loop: GIF loop count (0 = forever).
        gamma: Display gamma applied to the normalised volume. ``1.0`` = no
            change; values BELOW 1 brighten dim structure (e.g. 0.5), above 1
            darken. The usual knob for making faint neurites visible in a MIP.
        contrast_percentile: Normalise to this intensity percentile instead of
            the true maximum. ``100`` = old max-based behaviour; a value like
            99.5 stops a few hot voxels from darkening the whole render.
    """

    mode: Literal["mip", "alpha"] = "mip"
    n_frames: int = 36
    total_degrees: float = 360.0
    fps: int = 15
    colormap: str = "magma"
    downsample: int = 2
    rotation_axis: Literal["x", "y"] = "y"
    alpha_scale: float = 0.5
    loop: int = 0
    gamma: float = 1.0
    contrast_percentile: float = 100.0


@dataclass
class ExportConfig:
    """Controls interchange exports for Fiji/ImageJ and napari.

    Attributes:
        ome_tiff_labels: Write the label image as an OME-TIFF (opens in napari
            and Fiji with correct physical scale).
        imagej_rois: Write each object outline as an ImageJ ROI, zipped into one
            file that Fiji's ROI Manager opens directly.
    """

    ome_tiff_labels: bool = True
    imagej_rois: bool = True


@dataclass
class LargeImageConfig:
    """Controls tiled processing of stacks that do not fit in memory.

    Attributes:
        mode: ``"auto"`` tiles only when the stack would not fit in RAM,
            ``"off"`` keeps the old behaviour of refusing it, ``"force"`` always
            tiles (used for testing).
        tile_size: Edge length (px) of the square tile core. ``0`` = choose from
            available memory.
        halo: Extra pixels read around each tile so background subtraction and
            smoothing see the same neighbourhood as in a whole-image run. Objects
            reaching further than this from their centre are flagged as
            ``tile_clipped``. ``0`` = choose from the preprocessing radii.
        workers: Tiles analysed at the same time (separate processes). ``0`` = one per CPU
            core, at most 4. Tiles are made smaller as workers are added so that all of
            them together fit in memory.
    """

    mode: Literal["auto", "off", "force"] = "auto"
    tile_size: int = 0
    halo: int = 0
    workers: int = 0


@dataclass
class HiveConfig:
    """Top-level run configuration.

    Attributes:
        input_path: Folder of images (2D) or stack file/folder (3D).
        output_folder: Destination for all artifacts.
        mode: ``"2d"`` or ``"3d"``.
        channel: Which channel to analyse. ``"auto"`` picks the highest-variance
            channel, which is a heuristic, not a measurement.
        voxel_size_xy: Physical size of one pixel (e.g. um). Used to convert
            areas/lengths to physical units. Leave 1.0 to report in pixels.
        voxel_size_z: Physical z-step. Ignored in 2D.
        voxel_unit: Unit label for reports.
        save_intermediate: Write channel/preprocessed/baseline QC images.
        mesh_step_size: Marching-cubes stride for the 3D mesh; higher = coarser.
        seed: Random seed for any stochastic step, for reproducibility.
        extra_channels: Further channels to measure at the segmented objects, by
            colour name (RGB images) or by name/number from the file's metadata.
            The main ``channel`` finds the objects; these are only measured, and
            colocalization with the main channel is reported for each.
        conditions: Maps an image name to an experimental condition, for the
            group comparison in the report. Empty = no comparison.
    """

    input_path: str = "input_images"
    output_folder: str = "output_results"
    mode: Literal["2d", "3d"] = "2d"
    channel: Channel = "auto"

    voxel_size_xy: float = 1.0
    voxel_size_z: float = 1.0
    voxel_unit: str = "px"

    save_intermediate: bool = True
    mesh_step_size: int = 2
    seed: int = 0

    extra_channels: list[str] = field(default_factory=list)
    conditions: dict[str, str] = field(default_factory=dict)

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    arbor: ArborConfig = field(default_factory=ArborConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    large_image: LargeImageConfig = field(default_factory=LargeImageConfig)

    def __post_init__(self) -> None:
        """Validate on construction so bad parameters fail before any I/O.

        Catching these at build time means a long batch run cannot die
        halfway through on a parameter that was wrong from the start.
        """
        self.validate()

    def validate(self) -> None:
        """Reject impossible parameter combinations early.

        Raises:
            ValueError: If any parameter is outside its physical domain.
        """
        if self.voxel_size_xy <= 0 or self.voxel_size_z <= 0:
            raise ValueError(
                f"Voxel sizes must be positive, got xy={self.voxel_size_xy}, z={self.voxel_size_z}"
            )
        if self.segmentation.threshold_scale <= 0:
            raise ValueError(
                f"threshold_scale must be positive, got {self.segmentation.threshold_scale}"
            )
        if self.segmentation.min_size < 0:
            raise ValueError(f"min_size must be >= 0, got {self.segmentation.min_size}")
        if 0 < self.segmentation.max_size <= self.segmentation.min_size:
            raise ValueError(
                f"max_size ({self.segmentation.max_size}) must exceed "
                f"min_size ({self.segmentation.min_size}) or be 0 (unlimited)"
            )
        if self.arbor.sholl_step <= 0:
            raise ValueError(f"sholl_step must be positive, got {self.arbor.sholl_step}")
        if self.render.n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {self.render.n_frames}")
        if self.render.downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {self.render.downsample}")
        if self.preprocess.gaussian_sigma < 0:
            raise ValueError(
                f"gaussian_sigma must be >= 0, got {self.preprocess.gaussian_sigma}"
            )
        if min(self.large_image.tile_size, self.large_image.halo, self.large_image.workers) < 0:
            raise ValueError("tile_size, halo and workers must be >= 0 (0 = automatic)")
        if self.large_image.mode not in ("auto", "off", "force"):
            raise ValueError(f"large_image.mode must be auto, off or force, "
                             f"got {self.large_image.mode!r}")

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten to a single-level dict for the report's Settings sheet.

        Returns:
            Mapping of ``"section.parameter"`` to value, so a completed run
            records every parameter that produced it.
        """
        flat: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
            else:
                flat[key] = value
        return flat

    def to_dict(self) -> dict[str, Any]:
        """Nested, JSON-serialisable form of the whole configuration.

        Returns:
            A dict that :meth:`from_dict` turns back into an equal config.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HiveConfig":
        """Rebuild a configuration saved by :meth:`to_dict`.

        Unknown keys are rejected rather than ignored: a setting that is silently
        dropped on reload would give a different result from the saved run
        without any sign of it. Keys missing from an older file take their
        defaults.

        Args:
            data: Nested dict, as read back from JSON.

        Returns:
            The validated configuration.

        Raises:
            ValueError: On an unknown key or an out-of-range value.
        """
        data = dict(data)
        sections = {
            "preprocess": PreprocessConfig, "segmentation": SegmentationConfig,
            "arbor": ArborConfig, "render": RenderConfig, "export": ExportConfig,
            "large_image": LargeImageConfig,
        }
        unknown = set(data) - {f.name for f in fields(cls)}
        for name, section_cls in sections.items():
            if name not in data:
                continue
            values = dict(data[name])
            bad = set(values) - {f.name for f in fields(section_cls)}
            unknown |= {f"{name}.{key}" for key in bad}
            # JSON has no tuples; per-axis radii/sigmas must come back as tuples.
            data[name] = section_cls(**{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in values.items() if key not in bad
            })
        if unknown:
            raise ValueError(
                "Settings contain parameters this version of HIVE does not know: "
                f"{sorted(unknown)}. Refusing to load them, because ignoring a setting "
                "would give a different result from the saved run."
            )
        return cls(**data)
