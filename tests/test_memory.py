"""Large images must not crash the process: lean steps give identical results, and the
analysis checks free memory first and degrades or explains instead of being killed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, measure, segmentation

from hive import cleanup, io_utils, pipeline
from hive import visualization as viz
from hive.io_utils import load_image, load_image_native, save_uint8, to_uint8
from hive.pipeline import ImageTooLargeError, _fit_memory, analyze_folder
from hive.preprocessing import select_channel

from helpers import quick_config, write_synthetic_folder

RNG = np.random.default_rng(21)


# ------------------------------------------------------------------ 8-bit conversion

def _reference_uint8(image, rescale):
    """save_uint8's conversion as it was written: whole-array float32 steps."""
    arr = np.asarray(image, dtype=np.float32)
    if rescale:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    arr = np.clip(np.nan_to_num(arr), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


@pytest.mark.parametrize("make", [
    lambda: RNG.random((300, 200)).astype(np.float32) * 900,
    lambda: RNG.random((120, 90, 3)),                                   # float64 RGB overlay
    lambda: np.where(RNG.random((80, 70)) > 0.9, np.nan, RNG.random((80, 70))),
    lambda: np.full((40, 30), 7.0),                                     # no range at all
    lambda: -RNG.random((50, 50)) * 5,
])
@pytest.mark.parametrize("rescale", [True, False])
def test_banded_eight_bit_conversion_equals_the_whole_array_version(make, rescale, monkeypatch):
    image = make()
    monkeypatch.setattr(io_utils, "_BAND_ELEMENTS", 700)               # force many small bands
    assert np.array_equal(to_uint8(image, rescale), _reference_uint8(image, rescale))


def test_saved_pngs_hold_exactly_those_bytes(tmp_path):
    image = RNG.random((60, 80, 3)).astype(np.float32)
    save_uint8(tmp_path / "p.png", image)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "p.png")), _reference_uint8(image, True))
    with pytest.raises(ValueError, match="empty"):
        to_uint8(np.empty((0, 4)))


# ------------------------------------------------------------------ overlays

def _labels(shape, seed=0, drop=5):
    lab = measure.label(np.random.default_rng(seed).random(shape) > 0.9, connectivity=2)
    lab[lab % drop == 0] = 0
    return lab


@pytest.mark.parametrize("band_pixels", [4_000_000, 600])
def test_the_lean_label_overlay_equals_label2rgb(band_pixels):
    image = (RNG.random((150, 220)) * 700).astype(np.float32)
    for labels in (_labels(image.shape), np.zeros(image.shape, int), _labels(image.shape, 3) + 4):
        expected = color.label2rgb(labels, image=viz.normalize_for_display(image), bg_label=0,
                                   alpha=0.45).astype(np.float32)
        assert np.array_equal(viz.label_overlay_f32(image, labels, band_pixels=band_pixels), expected)


def test_the_boundary_overlay_equals_mark_boundaries():
    image = (RNG.random((120, 160)) * 300).astype(np.float32)
    labels = _labels(image.shape, 1)
    expected = segmentation.mark_boundaries(viz.to_rgb_display(image), labels, color=(1, 1, 0),
                                            mode="thick")
    assert np.array_equal(viz.create_boundary_overlay(image, labels), expected)


# ------------------------------------------------------------------ channel selection

def _old_select(arr, channel):
    """The channel selection as it was: a float32 copy of the whole image first."""
    rgb = arr[:, :, :3].astype(np.float32)
    if channel in ("red", "green", "blue"):
        return np.ascontiguousarray(rgb[:, :, "rgb".index(channel[0])]), channel
    if channel == "magenta":
        return ((rgb[:, :, 0] + rgb[:, :, 2]) / 2.0).astype(np.float32), "magenta"
    if channel == "gray":
        return color.rgb2gray(rgb).astype(np.float32), "gray"
    candidates = {"red": rgb[:, :, 0], "green": rgb[:, :, 1], "blue": rgb[:, :, 2],
                  "magenta": (rgb[:, :, 0] + rgb[:, :, 2]) / 2.0, "gray": color.rgb2gray(rgb)}
    best = max(candidates, key=lambda k: float(np.std(candidates[k])))
    return candidates[best].astype(np.float32), best


@pytest.mark.parametrize("image", [
    RNG.integers(0, 255, (80, 90, 3)).astype(np.uint8),
    (RNG.random((60, 70, 3)) * 4000).astype(np.uint16),
    np.repeat(RNG.integers(0, 255, (70, 60, 1)), 3, axis=2).astype(np.uint8),     # grey saved as RGB
])
@pytest.mark.parametrize("channel", ["auto", "red", "green", "blue", "magenta", "gray"])
def test_lean_channel_selection_returns_the_same_plane_and_name(image, channel):
    expected, name = _old_select(image.astype(np.float32), channel)
    for given in (image, image.astype(np.float32)):                       # stored dtype or float32
        got, got_name = select_channel(given, channel)
        assert got_name == name and got.dtype == np.float32 and np.array_equal(got, expected)


def test_images_load_in_their_stored_type_without_changing_a_pixel(tmp_path):
    Image.fromarray(RNG.integers(0, 255, (30, 40, 3)).astype(np.uint8)).save(tmp_path / "a.png")
    native, as_float = load_image_native(tmp_path / "a.png"), load_image(tmp_path / "a.png")
    assert native.dtype == np.uint8 and as_float.dtype == np.float32
    assert np.array_equal(native.astype(np.float32), as_float)


# ------------------------------------------------------------------ hole filling

def _old_fill_holes(labels, hole_size):
    mask = labels > 0
    filled_all = ndi.binary_fill_holes(mask)
    all_holes = filled_all & ~mask
    if hole_size < 0 or not all_holes.any():
        added = all_holes
    else:
        hole_labels = measure.label(all_holes)
        keep = np.bincount(hole_labels.ravel()) <= int(hole_size)
        keep[0] = False
        added = keep[hole_labels] & all_holes
    if int(added.sum()) == 0:
        return labels, 0
    _, indices = ndi.distance_transform_edt(~mask, return_indices=True)
    out = labels.copy()
    out[added] = labels[tuple(idx[added] for idx in indices)]
    return out, int(added.sum())


@pytest.mark.parametrize("shape", [(120, 150), (6, 40, 40)])
def test_hole_filling_fills_exactly_the_same_pixels(shape):
    for trial in range(6):
        grown = ndi.binary_dilation(RNG.random(shape) > 0.992, iterations=int(RNG.integers(2, 6)))
        holes = ndi.binary_dilation(RNG.random(shape) > 0.995, iterations=int(RNG.integers(0, 3)))
        labels = measure.label(grown & ~holes, connectivity=len(shape))
        for size in (0, 4, 40, -1):
            expected, n_expected = (labels, 0) if size == 0 else _old_fill_holes(labels, size)
            got, n_got = cleanup.fill_object_holes(labels, size)
            assert n_got == n_expected and np.array_equal(got, expected)


# ------------------------------------------------------------------ the memory guard

@pytest.fixture
def one_image(tmp_path):
    Image.fromarray(RNG.integers(0, 255, (1000, 1000, 3)).astype(np.uint8)).save(tmp_path / "x.png")
    return tmp_path / "x.png"


def _free(monkeypatch, nbytes):
    monkeypatch.setattr(pipeline, "_available_memory_bytes", lambda: nbytes)


def test_a_comfortable_image_is_left_alone(one_image, monkeypatch, tmp_path):
    _free(monkeypatch, 10 * 10**9)
    cfg = quick_config(tmp_path)
    got, note = _fit_memory(one_image, cfg)
    assert got is cfg and note is None


def test_clahe_is_switched_off_with_a_note_when_only_that_makes_it_fit(monkeypatch, tmp_path):
    big = tmp_path / "big.png"
    Image.new("L", (8000, 8000)).save(big)                  # 64 MP: CLAHE estimate 4.6 GB, without 2.6 GB
    _free(monkeypatch, 3_300_000_000)
    cfg = quick_config(tmp_path)
    cfg.preprocess.use_clahe = True
    got, note = _fit_memory(big, cfg)
    assert got.preprocess.use_clahe is False and cfg.preprocess.use_clahe is True
    assert "CLAHE was switched off for big.png" in note and "measurements are not affected" in note


def test_an_image_that_cannot_fit_is_refused_with_advice_before_any_work(monkeypatch, tmp_path):
    big = tmp_path / "big.png"
    Image.new("L", (8000, 8000)).save(big)
    _free(monkeypatch, 1_000_000_000)
    with pytest.raises(ImageTooLargeError, match="needs about .* but only .* is free"):
        _fit_memory(big, quick_config(tmp_path))
    assert issubclass(ImageTooLargeError, MemoryError)


def test_a_batch_reports_the_refused_image_and_still_analyses_the_others(monkeypatch, tmp_path):
    write_synthetic_folder(tmp_path / "in")
    Image.new("L", (8000, 8000)).save(tmp_path / "in" / "huge.png")
    _free(monkeypatch, 500_000_000)
    out = analyze_folder(quick_config(tmp_path), workers=1)
    assert [name for name, _ in out["errors"]] == ["huge.png"]
    assert len(out["summary"]) == 3 and "is free" in out["errors"][0][1]


def test_notes_reach_the_excel_report_and_the_run_record(monkeypatch, tmp_path):
    import json
    folder = tmp_path / "in"
    write_synthetic_folder(folder)
    monkeypatch.setattr(pipeline, "_fit_memory", lambda path, cfg: (cfg, f"note about {path.name}"))
    cfg = quick_config(tmp_path)
    out = analyze_folder(cfg, workers=1)
    assert sorted(out["notes"]) == [f"note about {n}.tif" for n in ("synthetic_bright", "synthetic_dim", "synthetic_reference")]
    with pd.ExcelFile(tmp_path / "out" / "HIVE_2D_report.xlsx") as book:
        assert "Notes" in book.sheet_names
    record = json.loads((tmp_path / "out" / "hive_run.json").read_text())
    assert len(record["notes"]["notes"]) == 3


def test_available_memory_is_never_more_than_the_machine_has(monkeypatch):
    free = io_utils._available_memory_bytes()
    assert free is None or free <= io_utils._physical_memory_bytes()
    monkeypatch.setattr(io_utils, "_physical_memory_bytes", lambda: 1_000_000)
    assert io_utils._available_memory_bytes() <= 1_000_000


def test_the_estimate_counts_pixels_not_colour_channels(tmp_path):
    Image.new("RGB", (1000, 1000)).save(tmp_path / "rgb.png")
    Image.new("L", (1000, 1000)).save(tmp_path / "gray.png")
    assert pipeline._peak_bytes_per_image(tmp_path / "rgb.png") == pipeline._peak_bytes_per_image(tmp_path / "gray.png")


def test_two_d_mode_refuses_z_stacks_and_estimate_counts_loaded_bytes(tmp_path):
    import numpy as np
    import tifffile
    from hive.config import HiveConfig
    from hive.io_utils import ImageLoadError
    from hive.pipeline import _peak_bytes_per_image, analyze_image

    stack = tmp_path / "stack.tif"
    tifffile.imwrite(stack, np.zeros((6, 32, 32), np.uint16), metadata={"axes": "ZYX"})
    assert _peak_bytes_per_image(stack) is None
    with pytest.raises(ImageLoadError, match="z-stack"):
        analyze_image(stack, HiveConfig())

    plane = tmp_path / "plane.tif"
    tifffile.imwrite(plane, np.zeros((3, 64, 64), np.uint16), metadata={"axes": "CYX"})
    assert _peak_bytes_per_image(plane) >= 3 * 64 * 64 * 2 + 64 * 64 * 4


def test_folder_of_stacks_is_refused_before_anything_is_read(tmp_path):
    import numpy as np
    import tifffile
    from hive.io_utils import ImageLoadError, load_zstack
    from hive.tiled import open_source

    tifffile.imwrite(tmp_path / "a_z01.tif", np.zeros((5, 16, 16), np.uint16), metadata={"axes": "ZYX"})
    with pytest.raises(ImageLoadError, match="one plane per file"):
        load_zstack(tmp_path)
    with pytest.raises(ImageLoadError, match="one plane per file"):
        with open_source(tmp_path):
            pass
