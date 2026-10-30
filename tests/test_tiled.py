"""Tiled analysis must agree with the whole-volume run, and say where it cannot."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from hive import io_utils, tiled
from hive.config import HiveConfig, LargeImageConfig
from hive.io_utils import ImageLoadError
from hive.pipeline import analyze_zstack

from helpers import blob_volume, quick_config, write_ome


def _cfg(tmp_path, path, mode, tile=0, halo=0, out="out", **kw):
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(tmp_path / out),
                       mode="3d", **kw)
    cfg.preprocess.tophat_radius = (2.0, 12.0, 12.0)
    cfg.preprocess.gaussian_sigma = 1.0
    cfg.large_image = LargeImageConfig(mode=mode, tile_size=tile, halo=halo)
    return cfg


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    path = tmp_path_factory.mktemp("stack") / "stack.tif"
    tifffile.imwrite(path, blob_volume())
    return path


def _sorted(df):
    return df.sort_values(["centroid-1", "centroid-2"]).reset_index(drop=True)


def test_tiles_find_the_same_objects_as_the_whole_volume(tmp_path, stack):
    whole = analyze_zstack(_cfg(tmp_path, stack, "off", out="whole"))
    tiles = analyze_zstack(_cfg(tmp_path, stack, "force", tile=64, out="tiled"))
    assert whole.object_count == tiles.object_count == 9
    a, b = _sorted(whole.measurements), _sorted(tiles.measurements)
    assert a["volume_voxels"].tolist() == b["volume_voxels"].tolist()
    np.testing.assert_allclose(a["intensity_mean"], b["intensity_mean"], rtol=0.01)
    np.testing.assert_allclose(a[["centroid-0", "centroid-1", "centroid-2"]].to_numpy(),
                               b[["centroid-0", "centroid-1", "centroid-2"]].to_numpy(), atol=0.01)
    assert not b["tile_clipped"].any()


def test_the_object_on_a_tile_seam_is_counted_once(tmp_path, stack):
    result = analyze_zstack(_cfg(tmp_path, stack, "force", tile=128, out="seam"))
    near_seam = result.measurements[(result.measurements["centroid-2"] - 128).abs() < 5]
    assert len(near_seam) == 1 and result.object_count == 9


def test_the_label_image_matches_and_has_unique_ids(tmp_path, stack):
    whole = analyze_zstack(_cfg(tmp_path, stack, "off", out="w2"))
    tiles = analyze_zstack(_cfg(tmp_path, stack, "force", tile=64, out="t2"))
    labels = np.asarray(tiles.labels)
    assert labels.shape == whole.labels.shape and labels.dtype == np.uint32
    assert np.array_equal(labels > 0, whole.labels > 0)
    assert sorted(set(labels[labels > 0])) == sorted(tiles.measurements["label"])
    with tifffile.TiffFile(tiles.outputs["labels_ome_tiff"]) as tif:
        assert tif.is_ome


def test_tiled_results_are_tables_and_files_without_the_whole_volume(tmp_path, stack):
    result = analyze_zstack(_cfg(tmp_path, stack, "force", tile=64, out="files"))
    assert result.prepared is None
    out = tmp_path / "files"
    assert (out / "HIVE_3D_measurements.csv").exists() and (out / "HIVE_3D_report.xlsx").exists()
    notes = json.loads((out / "hive_run.json").read_text())["notes"]["tiled"]
    assert notes["clahe"].startswith("disabled") and notes["tiles"] == 16 and notes["halo"] >= 16


def test_the_threshold_is_shared_so_a_dim_tile_is_not_cut_at_its_own_level(tmp_path):
    vol = blob_volume()
    vol[:, :, 110:] *= 0.0                              # right half empty: Otsu there would find noise
    vol[:, :, 110:] += 20
    path = tmp_path / "half.tif"
    tifffile.imwrite(path, vol)
    result = analyze_zstack(_cfg(tmp_path, path, "force", tile=110, out="half"))
    assert (result.measurements["centroid-2"] < 110).all()       # nothing invented in the empty tiles


def test_objects_larger_than_the_halo_are_flagged_not_hidden(tmp_path):
    vol = np.full((6, 120, 120), 10, np.float32)
    vol[1:5, 20:100, 20:100] += 300                      # one 80-px object spanning several tiles
    path = tmp_path / "big.tif"
    tifffile.imwrite(path, vol)
    cfg = _cfg(tmp_path, path, "force", tile=40, halo=16, out="clip")
    result = analyze_zstack(cfg)
    assert result.measurements["tile_clipped"].any()


def test_auto_mode_tiles_only_when_the_stack_would_not_fit(tmp_path, stack, monkeypatch):
    cfg = _cfg(tmp_path, stack, "auto")
    assert not tiled.needs_tiling(cfg)
    monkeypatch.setattr(tiled, "_physical_memory_bytes", lambda: 1_000_000)   # a 1 MB computer
    assert tiled.needs_tiling(cfg)
    cfg.large_image.mode = "off"
    assert not tiled.needs_tiling(cfg)
    cfg.large_image.mode = "auto"
    result = analyze_zstack(cfg)                           # routed automatically, not refused
    assert result.prepared is None and result.object_count == 9


def test_a_stack_that_does_not_fit_is_now_processed_not_refused(tmp_path, stack, monkeypatch):
    monkeypatch.setattr(io_utils, "_physical_memory_bytes", lambda: 1_000_000)
    monkeypatch.setattr(tiled, "_physical_memory_bytes", lambda: 1_000_000)
    with pytest.raises(io_utils.StackTooLargeError):
        io_utils.load_zstack(stack)                        # the whole-volume loader still refuses
    assert analyze_zstack(_cfg(tmp_path, stack, "auto", out="auto")).object_count == 9


def test_a_folder_of_slices_is_tiled_too(tmp_path):
    folder = tmp_path / "slices"
    folder.mkdir()
    for z, plane in enumerate(blob_volume()):
        tifffile.imwrite(folder / f"z{z:03d}.tif", plane)
    result = analyze_zstack(_cfg(tmp_path, folder, "force", tile=100, out="folder"))
    assert result.object_count == 9


def test_a_channel_can_be_chosen_by_name_in_a_multichannel_stack(tmp_path):
    vol = blob_volume()
    other = np.full_like(vol, 20)
    path = write_ome(tmp_path / "ch.ome.tif", np.stack([other, vol]), "CZYX", ["empty", "cells"],
                     z=1.0)
    result = analyze_zstack(_cfg(tmp_path, path, "force", tile=100, out="named", channel="cells"))
    assert result.object_count == 9
    with pytest.warns(UserWarning, match="uniform"):
        first = analyze_zstack(_cfg(tmp_path, path, "force", tile=100, out="first", channel="auto"))
    assert first.object_count == 0                          # "auto" took the first (empty) channel


def test_unsupported_layouts_and_settings_are_explained(tmp_path):
    series = tmp_path / "t.tif"
    tifffile.imwrite(series, np.zeros((3, 4, 20, 20), np.uint16), metadata={"axes": "TZYX"})
    with pytest.raises(ImageLoadError, match="axis T"):
        analyze_zstack(_cfg(tmp_path, series, "force"))
    cfg = _cfg(tmp_path, tmp_path / "x.tif", "force")
    cfg.segmentation.method = "cellpose"
    with pytest.raises(ValueError, match="threshold"):
        tiled.analyze_zstack_tiled(cfg)
    with pytest.raises(FileNotFoundError):
        analyze_zstack(_cfg(tmp_path, tmp_path / "missing.tif", "force"))


def test_a_uniform_volume_reports_zero_objects_with_a_warning(tmp_path):
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, np.full((6, 50, 50), 7, np.float32), photometric="minisblack")
    with pytest.warns(UserWarning, match="uniform"):
        result = analyze_zstack(_cfg(tmp_path, path, "force", out="flat"))
    assert result.object_count == 0 and not np.asarray(result.labels).any()


def test_halo_and_tile_choices():
    from hive.config import PreprocessConfig
    assert tiled.filter_halo(PreprocessConfig(baseline_method="tophat", tophat_radius=25.0,
                                              gaussian_sigma=1.0)) >= 25 + 4
    assert tiled.filter_halo(PreprocessConfig(baseline_method="none", gaussian_sigma=0)) == 16
    assert tiled.filter_halo(PreprocessConfig(baseline_method="smooth", baseline_sigma=20.0,
                                              gaussian_sigma=0)) >= 80
    side = tiled.choose_tile_size((100, 20000, 20000), halo=32)
    assert 64 <= side <= 20000 and side % 64 == 0
    assert tiled._grid(10, 10, 4)[-1] == (8, 10, 8, 10)
    assert len(tiled._grid(10, 10, 4)) == 9


def test_tiled_is_off_by_default_for_normal_sized_stacks(tmp_path, stack):
    assert not tiled.needs_tiling(HiveConfig(input_path=str(stack)))
