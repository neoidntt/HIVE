"""Pipeline, preprocessing and measurement tests.

The ``TestIntensityFidelity`` class is the most important file in this suite: it
pins the defect that made the legacy pipeline's quantification invalid. If those
tests ever fail, the exported numbers are not fluorescence.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from hive.cleanup import clean_labels, filter_by_size, normalize_for_display, remove_edge_objects
from hive.config import ArborConfig, HiveConfig, PreprocessConfig, SegmentationConfig
from hive.measurements import measure_objects, summarize_measurements
from hive.pipeline import analyze_folder, analyze_zstack
from hive.preprocessing import prepare_image, prepare_volume, select_channel, select_channel_volume
from hive.segmentation import label_mask, threshold_segmentation, watershed_split


def _write_blob_tif(path, level: int, background: int = 100, size: int = 80):
    """Write a uint16 image with one square blob at a known intensity."""
    img = np.full((size, size), background, dtype=np.uint16)
    img[30:50, 30:50] = level
    tifffile.imwrite(path, img)
    return img


@pytest.fixture
def flat_cfg(tmp_path) -> HiveConfig:
    """A config with no intensity-altering preprocessing, for exact assertions."""
    return HiveConfig(
        input_path=str(tmp_path / "in"),
        output_folder=str(tmp_path / "out"),
        channel="gray",
        preprocess=PreprocessConfig(
            baseline_method="percentile", baseline_percentile=25,
            gaussian_sigma=0, use_clahe=False,
        ),
        segmentation=SegmentationConfig(min_size=50),
        arbor=ArborConfig(enabled=False),
    )


class TestIntensityFidelity:
    """The legacy pipeline min-max scaled every image to its own range, so a
    real 3x biological difference reported as 1.00x. These tests pin the fix."""

    def test_intensity_is_comparable_across_images(self, tmp_path, flat_cfg):
        folder = tmp_path / "in"
        folder.mkdir()
        _write_blob_tif(folder / "dim.tif", 300)     # 200 above background
        _write_blob_tif(folder / "bright.tif", 900)  # 800 above background -> 4x

        result = analyze_folder(flat_cfg, write_per_image=False)
        values = dict(zip(result["measurements"]["image"],
                          result["measurements"]["intensity_mean"]))
        assert values["bright.tif"] / values["dim.tif"] == pytest.approx(4.0, rel=0.05)

    def test_clahe_does_not_leak_into_measurements(self, tmp_path):
        """CLAHE must alter only the segmentation image."""
        folder = tmp_path / "in"
        folder.mkdir()
        _write_blob_tif(folder / "dim.tif", 300)
        _write_blob_tif(folder / "bright.tif", 900)
        cfg = HiveConfig(
            input_path=str(folder), output_folder=str(tmp_path / "out"), channel="gray",
            preprocess=PreprocessConfig(baseline_method="percentile", baseline_percentile=25,
                                        gaussian_sigma=0, use_clahe=True),
            segmentation=SegmentationConfig(min_size=50), arbor=ArborConfig(enabled=False),
        )
        result = analyze_folder(cfg, write_per_image=False)
        values = dict(zip(result["measurements"]["image"],
                          result["measurements"]["intensity_mean"]))
        assert values["bright.tif"] / values["dim.tif"] == pytest.approx(4.0, rel=0.05)

    def test_z_gradient_survives_preprocessing(self):
        """The legacy per-plane min-max returned an identical mean at every
        plane of a stack with a real 33% axial gradient."""
        stack = np.full((6, 60, 60), 100, dtype=np.uint16)
        for z in range(6):
            stack[z, 20:40, 20:40] = 3000 + z * 200
        prepared = prepare_volume(
            stack,
            PreprocessConfig(baseline_method="percentile", baseline_percentile=25,
                             gaussian_sigma=0, use_clahe=False),
            channel="gray",
        )
        means = [prepared.measurement[z, 20:40, 20:40].mean() for z in range(6)]
        assert means[-1] / means[0] == pytest.approx(3900 / 2900, rel=0.02)

    def test_measurement_image_keeps_physical_scale(self):
        img = np.full((60, 60), 100, dtype=np.float32)
        img[20:40, 20:40] = 900
        prepared = prepare_image(
            img,
            PreprocessConfig(baseline_method="percentile", baseline_percentile=25,
                             gaussian_sigma=0, use_clahe=False),
            channel="gray",
        )
        # 900 - 100 background = 800, not rescaled to [0, 1]
        assert prepared.measurement[20:40, 20:40].mean() == pytest.approx(800.0)
        assert prepared.segmentation.max() <= 1.0  # the segmentation image IS normalised

    def test_integrated_intensity_is_mean_times_area(self):
        labels = np.zeros((40, 40), dtype=np.int32)
        labels[10:20, 10:20] = 1
        intensity = np.zeros((40, 40), dtype=np.float32)
        intensity[10:20, 10:20] = 5.0
        df = measure_objects(labels, intensity)
        assert df.loc[0, "integrated_intensity"] == pytest.approx(5.0 * 100)


class TestChannelSelection:
    """Channel reduction happens before baseline subtraction, and is recorded.

    Order matters: the legacy pipeline subtracted baseline from the raw array
    first, which made it reject channel-last RGB stacks outright. These tests
    pin the corrected order, and require that ``channel="auto"`` resolve exactly
    once for a whole stack -- not per slice, where noise could select different
    channels on different planes -- and report the name it chose.
    """

    def test_rgb_zstack_reduces_before_baseline(self):
        """Regression: the legacy pipeline ran baseline subtraction on the raw
        4D array and crashed on every RGB z-stack."""
        stack = np.zeros((6, 40, 40, 3), dtype=np.float32)
        stack[:, 10:30, 10:30, 0] = 200
        stack[:, 10:30, 10:30, 2] = 200
        volume, name = select_channel_volume(stack, "magenta")
        assert volume.shape == (6, 40, 40)
        assert name == "magenta"

    def test_auto_resolves_once_for_the_whole_stack(self):
        """A per-plane 'auto' could pick different channels per plane."""
        stack = np.zeros((4, 40, 40, 3), dtype=np.float32)
        stack[:, 10:30, 10:30, 1] = 200  # green dominates
        volume, name = select_channel_volume(stack, "auto")
        assert name == "green"
        assert volume.shape == (4, 40, 40)

    def test_auto_reports_the_resolved_name(self):
        img = np.zeros((40, 40, 3), dtype=np.float32)
        img[10:30, 10:30, 2] = 200
        _, name = select_channel(img, "auto")
        assert name == "blue"

    def test_grayscale_passes_through(self):
        out, name = select_channel(np.ones((10, 10), dtype=np.float32), "magenta")
        assert out.shape == (10, 10)
        assert name == "gray"

    def test_rejects_unknown_channel(self):
        with pytest.raises(ValueError, match="Unknown channel"):
            select_channel(np.zeros((10, 10, 3), dtype=np.float32), "chartreuse")

    def test_rejects_channel_first_stack_with_guidance(self):
        with pytest.raises(ValueError, match="moveaxis"):
            select_channel_volume(np.zeros((3, 6, 40, 40), dtype=np.float32), "red")


class TestWatershed:
    """Distance-transform watershed splits touching convex objects.

    Two overlapping discs are the textbook case with a known answer: one
    connected component in, two objects out. Also pins that the function
    returns a *label* image rather than a boolean mask -- the legacy version
    returned ``labels > 0``, discarding the split it had just computed.
    """

    def test_splits_touching_discs(self):
        """Regression: the legacy watershed_split ended `return labels > 0`,
        discarding the split and returning the input mask unchanged."""
        from skimage.draw import disk

        mask = np.zeros((100, 160), dtype=bool)
        for centre in ((50, 60), (50, 100)):
            rr, cc = disk(centre, 28, shape=mask.shape)
            mask[rr, cc] = True
        assert label_mask(mask).max() == 1  # they merge into one blob
        assert watershed_split(mask, min_distance=10).max() == 2

    def test_returns_labels_not_a_mask(self):
        """Two touching discs must come back as labels 1 and 2, not a boolean
        mask -- the legacy version collapsed the split back to `labels > 0`."""
        from skimage.draw import disk

        mask = np.zeros((100, 160), dtype=bool)
        for centre in ((50, 60), (50, 100)):
            rr, cc = disk(centre, 28, shape=mask.shape)
            mask[rr, cc] = True
        out = watershed_split(mask, min_distance=10)
        assert out.dtype != bool
        assert sorted(np.unique(out)) == [0, 1, 2]
        assert not np.array_equal(out.astype(bool), out)

    def test_empty_mask_is_safe(self):
        assert watershed_split(np.zeros((20, 20), dtype=bool)).max() == 0


class TestCleanup:
    """Size/edge filters and display normalisation behave predictably.

    Covers both ends of the size window (debris and merged clumps), consecutive
    relabelling after removals -- gaps in label numbering break downstream
    indexing -- and the cleanup report's bookkeeping.

    ``normalize_for_display`` is pinned against hot pixels specifically: scaling
    by the true maximum lets one saturated pixel darken an entire image, so
    normalisation uses a high percentile instead.
    """

    def test_filter_by_size_removes_debris(self):
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[10:20, 10:20] = 1  # 100 px
        labels[40, 40] = 2        # 1 px of debris
        out, n_small, n_large = filter_by_size(labels, min_size=50, max_size=0)
        assert n_small == 1
        assert out.max() == 1

    def test_filter_by_size_removes_clumps(self):
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[10:20, 10:20] = 1   # 100 px
        labels[30:55, 30:55] = 2   # 625 px
        out, n_small, n_large = filter_by_size(labels, min_size=0, max_size=200)
        assert n_large == 1
        assert out.max() == 1

    def test_labels_are_relabelled_consecutively(self):
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[5, 5] = 1
        labels[10:20, 10:20] = 2
        labels[30:40, 30:40] = 3
        out, _, _ = filter_by_size(labels, min_size=50)
        assert sorted(np.unique(out)) == [0, 1, 2]  # no gap where 1 was removed

    def test_remove_edge_objects(self):
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[0:10, 0:10] = 1     # touching the border
        labels[30:40, 30:40] = 2   # interior
        out, n = remove_edge_objects(labels)
        assert n == 1
        assert out.max() == 1

    def test_cleanup_report_counts_everything(self):
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[0:5, 0:5] = 1
        labels[30, 30] = 2
        labels[40:55, 40:55] = 3
        cfg = SegmentationConfig(min_size=50, clear_border=True, hole_size=0)
        _, report = clean_labels(labels, cfg)
        assert report.initial_count == 3
        assert report.final_count == 1
        assert "removed" in report.summary()

    def test_normalize_for_display_is_robust_to_hot_pixels(self):
        """A single saturated pixel must not flatten the whole image.

        The image needs real tonal range for percentiles to be meaningful; a
        constant field plus one hot pixel is degenerate (p1 == p99.5).
        """
        img = np.linspace(0.2, 0.8, 50 * 50, dtype=np.float32).reshape(50, 50)
        img[0, 0] = 1000.0  # one hot pixel

        # Measure the real data only; the outlier itself would dominate any
        # whole-image statistic and mask what we are actually testing.
        out = normalize_for_display(img, method="percentile")
        assert out[1:, 1:].std() > 0.1  # the real gradient survives clipping

        minmax = normalize_for_display(img, method="minmax")
        assert minmax[1:, 1:].std() < 0.001  # the hot pixel crushes it all to ~0
        assert minmax[25, 25] < 0.01

    def test_normalize_rejects_bad_percentiles(self):
        with pytest.raises(ValueError, match="low < high"):
            normalize_for_display(np.zeros((5, 5)), method="percentile", low=90, high=10)


class TestSegmentation:
    """Segmentation degrades gracefully and honours its cleanup parameters.

    A blank image has no bimodal histogram for Otsu to split; it must report
    zero objects rather than raise, so one empty slice cannot abort a batch.
    """

    def test_blank_image_reports_zero_objects_not_an_error(self):
        """A blank field is data ('no cells here'), so it must stay in the
        batch summary rather than being dropped as a failure."""
        flat = np.full((40, 40), 0.5, dtype=np.float32)
        with pytest.warns(UserWarning, match="0 objects"):
            mask = threshold_segmentation(flat, SegmentationConfig())
        assert not mask.any()

    def test_hole_size_is_actually_used(self):
        """Regression: the legacy 3D path accepted hole_size and ignored it."""
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[10:50, 10:50] = 1
        labels[28:32, 28:32] = 0  # a 16-px hole
        filled, report = clean_labels(labels, SegmentationConfig(hole_size=50, min_size=0))
        assert report.holes_filled == 16
        assert filled[30, 30] == 1

        kept, report2 = clean_labels(labels, SegmentationConfig(hole_size=0, min_size=0))
        assert report2.holes_filled == 0
        assert kept[30, 30] == 0


class TestConfig:
    """HiveConfig validates on construction and records itself faithfully.

    Impossible parameters must fail immediately rather than halfway through a
    long batch. ``to_flat_dict`` must expose nested sections so the report's
    Settings sheet can reproduce a run.

    ``test_default_threshold_scale_is_true_otsu`` guards a deliberate break
    from 1.x: the default is ``1.0`` (unmodified Otsu), not the legacy ``1.2``.
    """

    def test_rejects_negative_voxel_size(self):
        with pytest.raises(ValueError, match="Voxel sizes must be positive"):
            HiveConfig(voxel_size_xy=-1.0)

    def test_rejects_inverted_size_window(self):
        with pytest.raises(ValueError, match="must exceed"):
            HiveConfig(segmentation=SegmentationConfig(min_size=100, max_size=50))

    def test_flat_dict_records_nested_parameters(self):
        flat = HiveConfig().to_flat_dict()
        assert "segmentation.threshold_scale" in flat
        assert "arbor.sholl_step" in flat
        assert len(flat) > 30

    def test_default_threshold_scale_is_true_otsu(self):
        """The legacy default of 1.2 silently biased every run."""
        assert HiveConfig().segmentation.threshold_scale == 1.0


class TestPipelineIntegration:
    """End-to-end runs over real files on disk.

    Pins the 2.0 API contract (``analyze_zstack`` returns an object, not the
    legacy tuple), batch robustness (one corrupt file must not kill the run,
    while a genuinely empty folder must raise), and that voxel sizes actually
    propagate into physical measurements rather than being recorded and ignored.
    """

    def test_rgb_zstack_folder_runs(self, tmp_path):
        """The README's documented 3D magenta workflow, which used to crash."""
        from skimage.io import imsave

        folder = tmp_path / "slices"
        folder.mkdir()
        for z in range(6):
            img = np.zeros((60, 60, 3), dtype=np.uint8)
            img[20:40, 20:40, 0] = 200
            img[20:40, 20:40, 2] = 200
            imsave(folder / f"z{z:02d}.png", img, check_contrast=False)

        cfg = HiveConfig(
            input_path=str(folder), output_folder=str(tmp_path / "out"), mode="3d",
            channel="magenta",
            preprocess=PreprocessConfig(tophat_radius=(1, 20, 20)),
            segmentation=SegmentationConfig(min_size=50), arbor=ArborConfig(enabled=False),
        )
        result = analyze_zstack(cfg)
        assert result.object_count >= 1
        assert result.prepared.channel_name == "magenta"

    def test_result_is_an_object_not_a_tuple(self, tmp_path):
        """Regression: Streamlit bound the returned tuple to one name."""
        stack = np.full((5, 40, 40), 100, dtype=np.uint16)
        stack[:, 15:25, 15:25] = 900
        path = tmp_path / "s.tif"
        tifffile.imwrite(path, stack)
        cfg = HiveConfig(
            input_path=str(path), output_folder=str(tmp_path / "out"), mode="3d", channel="gray",
            preprocess=PreprocessConfig(baseline_method="percentile", gaussian_sigma=0),
            segmentation=SegmentationConfig(min_size=20), arbor=ArborConfig(enabled=False),
        )
        result = analyze_zstack(cfg)
        assert hasattr(result, "measurements")
        assert hasattr(result, "summary")
        assert result.object_count >= 1

    def test_batch_survives_a_corrupt_file(self, tmp_path):
        folder = tmp_path / "in"
        folder.mkdir()
        _write_blob_tif(folder / "good.tif", 900, size=60)
        (folder / "corrupt.tif").write_bytes(b"not a tiff")
        cfg = HiveConfig(
            input_path=str(folder), output_folder=str(tmp_path / "out"), channel="gray",
            preprocess=PreprocessConfig(baseline_method="none"),
            segmentation=SegmentationConfig(min_size=20), arbor=ArborConfig(enabled=False),
        )
        result = analyze_folder(cfg, write_per_image=False)
        assert len(result["errors"]) == 1
        assert not result["measurements"].empty

    def test_empty_folder_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        cfg = HiveConfig(input_path=str(tmp_path / "empty"), output_folder=str(tmp_path / "out"))
        with pytest.raises(FileNotFoundError, match="No readable images"):
            analyze_folder(cfg)

    def test_physical_units_scale_measurements(self, tmp_path):
        folder = tmp_path / "in"
        folder.mkdir()
        _write_blob_tif(folder / "a.tif", 900)  # a 20x20 = 400 px blob
        cfg = HiveConfig(
            input_path=str(folder), output_folder=str(tmp_path / "out"), channel="gray",
            voxel_size_xy=0.5, voxel_unit="um",
            preprocess=PreprocessConfig(baseline_method="none", gaussian_sigma=0, use_clahe=False),
            segmentation=SegmentationConfig(min_size=50), arbor=ArborConfig(enabled=False),
        )
        result = analyze_folder(cfg, write_per_image=False)
        row = result["measurements"].iloc[0]
        assert row["area_px"] == pytest.approx(400, rel=0.1)
        assert row["area_um2"] == pytest.approx(row["area_px"] * 0.25)


class TestStackPlanes:
    """Analysing a huge stack page-by-page must equal extracting it to disk first."""

    def _stack(self, tmp_path, z=8, s=128):
        rng = np.random.default_rng(0)
        vol = np.full((z, s, s), 400, dtype=np.uint16)
        for zi in range(z):
            yy, xx = np.ogrid[:s, :s]
            for _ in range(3):
                cy, cx = rng.integers(25, s - 25, 2)
                vol[zi][(yy - cy) ** 2 + (xx - cx) ** 2 < 100] = 3000 + zi * 20
        path = tmp_path / "stack.tif"
        tifffile.imwrite(path, vol)
        return path

    def _cfg(self, out):
        return HiveConfig(
            output_folder=str(out), channel="gray",
            preprocess=PreprocessConfig(baseline_method="percentile", gaussian_sigma=0,
                                        use_clahe=False),
            segmentation=SegmentationConfig(min_size=50), arbor=ArborConfig(enabled=False),
        )

    def test_matches_extract_then_analyze(self, tmp_path):
        from hive.pipeline import analyze_stack_planes
        from prep_stack import extract_2d

        stack = self._stack(tmp_path)
        direct = analyze_stack_planes(stack, self._cfg(tmp_path / "a"), workers=2)

        extract_2d(stack, 0, tmp_path / "split")
        cfg = self._cfg(tmp_path / "b")
        cfg.input_path = str(tmp_path / "split")
        via_disk = analyze_folder(cfg, write_per_image=False, workers=2)

        a = direct["measurements"].sort_values(["image", "label"]).reset_index(drop=True)
        b = via_disk["measurements"].sort_values(["image", "label"]).reset_index(drop=True)
        assert len(a) == len(b)
        assert np.allclose(a["intensity_mean"], b["intensity_mean"])
        assert np.allclose(a["area_px"], b["area_px"])

    def test_unlabelled_axes_still_finds_every_plane(self, tmp_path):
        """Regression: a plain 3D imwrite yields axes 'QYX', not 'ZYX'. Looking up
        'Z' with a default of 1 made extract_2d write ONE plane out of N and report
        success -- silently analysing 5% of the data."""
        from prep_stack import _resolve_dims, extract_2d

        stack = self._stack(tmp_path, z=8)
        with tifffile.TiffFile(stack) as tif:
            series = tif.series[0]
            assert "Z" not in series.axes          # the trap
            n_z, n_c, z_axis = _resolve_dims(series.axes, tuple(series.shape))
        assert n_z == 8
        assert n_c == 1

        extract_2d(stack, 0, tmp_path / "out")
        assert len(list((tmp_path / "out").glob("*.tif"))) == 8

    def test_ambiguous_axes_raise_rather_than_guess(self):
        from prep_stack import _resolve_dims

        with pytest.raises(ValueError, match="Cannot tell which axis is depth"):
            _resolve_dims("TQYX", (3, 4, 64, 64))

    def test_subset_of_planes(self, tmp_path):
        from hive.pipeline import analyze_stack_planes

        stack = self._stack(tmp_path, z=8)
        out = analyze_stack_planes(stack, self._cfg(tmp_path / "c"), planes=[2, 5])
        assert sorted(out["summary"]["image"]) == ["z002", "z005"]

    def test_out_of_range_plane_rejected(self, tmp_path):
        from hive.pipeline import analyze_stack_planes

        stack = self._stack(tmp_path, z=4)
        with pytest.raises(ValueError, match="out of range"):
            analyze_stack_planes(stack, self._cfg(tmp_path / "d"), planes=[99])


class TestParallelBatch:
    """Images are independent, so parallel and sequential must agree exactly."""

    def _folder(self, tmp_path, n=6):
        folder = tmp_path / "in"
        folder.mkdir()
        for i in range(n):
            img = np.full((120, 120), 100, dtype=np.uint16)
            img[40:80, 40:80] = 800 + i * 100   # each image genuinely different
            tifffile.imwrite(folder / f"p{i:02d}.tif", img)
        return folder

    def _cfg(self, folder, out):
        return HiveConfig(
            input_path=str(folder), output_folder=str(out), channel="gray",
            preprocess=PreprocessConfig(baseline_method="percentile", gaussian_sigma=0,
                                        use_clahe=False),
            segmentation=SegmentationConfig(min_size=50), arbor=ArborConfig(enabled=False),
        )

    def test_parallel_matches_sequential(self, tmp_path):
        folder = self._folder(tmp_path)
        seq = analyze_folder(self._cfg(folder, tmp_path / "a"), write_per_image=False, workers=1)
        par = analyze_folder(self._cfg(folder, tmp_path / "b"), write_per_image=False, workers=3)
        a = seq["measurements"].sort_values("image").reset_index(drop=True)
        b = par["measurements"].sort_values("image").reset_index(drop=True)
        assert a.equals(b)

    def test_parallel_preserves_per_image_attribution(self, tmp_path):
        """Each row must stay attached to the image it came from -- a worker pool
        that mixed these up would be silently catastrophic."""
        folder = self._folder(tmp_path)
        par = analyze_folder(self._cfg(folder, tmp_path / "b"), write_per_image=False, workers=3)
        rows = par["measurements"].sort_values("image").reset_index(drop=True)
        # image p{i} was written with intensity 800 + i*100, background 100
        for i, (_, row) in enumerate(rows.iterrows()):
            assert row["image"] == f"p{i:02d}.tif"
            assert row["intensity_mean"] == pytest.approx(700 + i * 100, rel=0.02)

    def test_parallel_surfaces_errors(self, tmp_path):
        folder = self._folder(tmp_path, n=3)
        (folder / "corrupt.tif").write_bytes(b"not a tiff")
        out = analyze_folder(self._cfg(folder, tmp_path / "c"), write_per_image=False, workers=2)
        assert len(out["errors"]) == 1
        assert not out["measurements"].empty

    def test_rejects_zero_workers(self, tmp_path):
        folder = self._folder(tmp_path, n=2)
        with pytest.raises(ValueError, match="workers must be >= 1"):
            analyze_folder(self._cfg(folder, tmp_path / "d"), workers=0)


class TestMemoryGuard:
    """HIVE loads volumes whole. A stack too large to fit must fail fast with an
    actionable message, not swap the machine to death and take the kernel with it."""

    def test_single_file_is_guarded(self, tmp_path):
        """Regression: the guard lived only in the folder branch, after the
        single-file early return — so one oversized TIFF went straight to
        tifffile.imread with no check at all."""
        from hive.io_utils import StackTooLargeError, load_zstack

        stack = np.zeros((10, 64, 64), dtype=np.uint16)
        path = tmp_path / "s.tif"
        tifffile.imwrite(path, stack)
        with pytest.raises(StackTooLargeError, match="float32"):
            load_zstack(path, max_bytes=1000)

    def test_folder_is_guarded(self, tmp_path):
        from hive.io_utils import StackTooLargeError, load_zstack

        folder = tmp_path / "slices"
        folder.mkdir()
        for z in range(4):
            tifffile.imwrite(folder / f"z{z:02d}.tif", np.zeros((64, 64), dtype=np.uint16))
        with pytest.raises(StackTooLargeError, match="float32"):
            load_zstack(folder, max_bytes=1000)

    def test_guard_message_is_actionable(self, tmp_path):
        """The message must say what to do, not just that it failed."""
        from hive.io_utils import StackTooLargeError, load_zstack

        tifffile.imwrite(tmp_path / "s.tif", np.zeros((10, 64, 64), dtype=np.uint16))
        with pytest.raises(StackTooLargeError) as excinfo:
            load_zstack(tmp_path / "s.tif", max_bytes=1000)
        message = str(excinfo.value)
        assert "downsample" in message.lower()
        assert "crop" in message.lower()
        assert "zarr" in message.lower() or "dask" in message.lower()

    def test_normal_stack_is_unaffected(self, tmp_path):
        from hive.io_utils import load_zstack

        stack = np.zeros((6, 32, 32), dtype=np.uint16)
        tifffile.imwrite(tmp_path / "s.tif", stack)
        assert load_zstack(tmp_path / "s.tif").shape == (6, 32, 32)

    def test_header_peek_does_not_load_pixels(self, tmp_path):
        """The size check must read the header only — checking by loading would
        defeat the entire purpose."""
        from hive.io_utils import _peek_stack_bytes

        tifffile.imwrite(tmp_path / "s.tif", np.zeros((10, 64, 64), dtype=np.uint16))
        n_bytes, shape = _peek_stack_bytes(tmp_path / "s.tif")
        assert shape == (10, 64, 64)
        assert n_bytes == 10 * 64 * 64 * 4  # float32, not the uint16 on disk

    def test_fmt_bytes_picks_sensible_units(self):
        from hive.io_utils import _fmt_bytes

        assert _fmt_bytes(70e9) == "70.0 GB"
        assert _fmt_bytes(1.9e6) == "1.9 MB"


class TestMeasurements:
    """Measurement guards against malformed input and empty results.

    A label/intensity shape mismatch means the two images are not the same
    field of view, so it must raise rather than silently measure misaligned
    pixels. Zero detections must still yield a well-formed (empty) frame and a
    summary row, so an image with no objects appears in the batch table as a
    genuine zero instead of disappearing.
    """

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            measure_objects(np.zeros((10, 10), dtype=np.int32), np.zeros((20, 20), dtype=np.float32))

    def test_empty_labels_give_empty_frame(self):
        df = measure_objects(np.zeros((10, 10), dtype=np.int32), np.zeros((10, 10), dtype=np.float32))
        assert df.empty

    def test_summary_of_zero_objects_still_yields_a_row(self):
        """A blank image must stay visible in a batch summary as an explicit 0."""
        import pandas as pd

        summary = summarize_measurements(pd.DataFrame(), image_name="blank.tif")
        assert len(summary) == 1
        assert summary.loc[0, "object_count"] == 0
