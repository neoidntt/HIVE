"""The 0.0.4 speed-ups must give exactly the results of the slower code they replaced.

Each test keeps the old, slower calculation as a reference (or compares with a function that
is still in the package) and checks the new one against it.
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

from hive import arborization as arb
from hive.io_utils import save_uint8, save_uint8_many
from hive.pipeline import _peak_bytes_per_image, analyze_array, analyze_folder, auto_workers

from helpers import quick_config, write_synthetic_folder

RNG = np.random.default_rng(11)


def _branchy_mask(shape=(100, 120), lines=45, length=30):
    mask = np.zeros(shape, bool)
    for _ in range(lines):
        y, x = RNG.integers(5, shape[0] - 5), RNG.integers(5, shape[1] - 5)
        angle = RNG.random() * 6.28
        for t in range(length):
            yy, xx = int(y + t * np.sin(angle)), int(x + t * np.cos(angle))
            if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
                mask[yy, xx] = True
    return mask | np.roll(mask, 1, 1)


def _blobs_with_arms(size=160, count=14, seed=3):
    """Separate objects, each a disc with a few arms, as in a field of cells."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 20.0, np.float32)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(count):
        cy, cx = rng.integers(25, size - 25, 2)
        img[((yy - cy) ** 2 + (xx - cx) ** 2) < 25] += 200
        for _arm in range(3):
            a = rng.random() * 6.28
            for t in range(4, 22):
                y, x = int(cy + t * np.sin(a)), int(cx + t * np.cos(a))
                img[y - 1:y + 1, x - 1:x + 1] += 150
    return img


# ---------------------------------------------------------------- branch segment lengths


@pytest.mark.parametrize("spacing", [1.0, 0.35])
def test_all_segment_lengths_equal_measuring_each_segment_alone(spacing):
    skeleton = arb.skeletonize_mask(_branchy_mask())
    segments = arb._branch_segments(skeleton, arb.find_branch_points(skeleton))
    one_by_one = [arb.skeleton_length(segments == r.label, spacing)
                  for r in measure.regionprops(segments)]
    assert len(one_by_one) > 20
    assert np.array_equal(arb._segment_lengths(segments, spacing), np.array(one_by_one))


def test_segment_lengths_in_3d_with_anisotropic_spacing():
    labels = measure.label(RNG.random((9, 25, 25)) > 0.8)
    spacing = (1.0, 0.3, 0.3)
    one_by_one = [arb.skeleton_length(labels == r.label, spacing) for r in measure.regionprops(labels)]
    assert np.array_equal(arb._segment_lengths(labels, spacing), np.array(one_by_one))
    assert arb._segment_lengths(np.zeros((4, 4), int)).size == 0


# ---------------------------------------------------------------- spur pruning


def _reference_prune(skeleton, min_length, spacing):
    """The pruning loop as it was: array-indexing walk, one pixel at a time."""
    skel = np.asarray(skeleton, dtype=bool).copy()
    spac = arb._as_spacing(skel.ndim, spacing)
    removed = 0
    for _ in range(10):
        endpoints = arb.find_endpoints(skel)
        if not endpoints.any():
            break
        junctions = arb.find_branch_points(skel)
        doomed = set()
        for tip in map(tuple, np.argwhere(endpoints)):
            if tip in doomed:
                continue
            spur = arb._walk_spur(skel, junctions, tip, float(min_length), spac)
            if spur is not None:
                doomed.update(spur)
                removed += 1
        if not doomed:
            break
        for pixel in doomed:
            skel[pixel] = False
    return skel, removed


@pytest.mark.parametrize("shape,spacing", [((100, 120), 1.0), ((100, 120), 0.4), ((8, 40, 40), (1.0, 0.3, 0.3))])
def test_pruning_removes_exactly_the_same_spurs(shape, spacing):
    mask = _branchy_mask(shape) if len(shape) == 2 else RNG.random(shape) > 0.78
    skeleton = arb.skeletonize_mask(mask)
    for length in (3, 6):
        expected, n_expected = _reference_prune(skeleton, length, spacing)
        got, n_got = arb.prune_skeleton(skeleton, length, spacing)
        assert n_got == n_expected and np.array_equal(got, expected)


# ---------------------------------------------------------------- Sholl analysis


def _reference_intersections(skeleton, center, step, spacing):
    """Sholl crossings as they were counted: a full-size shell mask, labelled, per radius."""
    spac = arb._as_spacing(skeleton.ndim, spacing)
    dist = arb._distance_map(skeleton.shape, center, spac)
    max_radius = float(dist[skeleton].max())
    thickness = float(np.linalg.norm(spac))
    radii = np.arange(step, max_radius + step, step, dtype=np.float32)
    counts = []
    for radius in radii:
        crossing = skeleton & (np.abs(dist - radius) <= thickness / 2.0)
        counts.append(int(measure.label(crossing).max()) if crossing.any() else 0)
    return radii, np.array(counts)


@pytest.mark.parametrize("step", [2.0, 5.0])
def test_sholl_counts_the_same_crossings_in_2d(step):
    skeleton = arb.skeletonize_mask(_branchy_mask())
    center = tuple(float(c) for c in ndi.center_of_mass(skeleton))
    radii, counts = _reference_intersections(skeleton, center, step, 1.0)
    got = arb.sholl_analysis(skeleton, center, step)
    assert np.array_equal(got.radii, radii) and np.array_equal(got.intersections, counts)


def test_sholl_counts_the_same_crossings_in_3d():
    skeleton = arb.skeletonize_mask(RNG.random((9, 36, 36)) > 0.8)
    center = tuple(float(c) for c in ndi.center_of_mass(skeleton))
    spacing = (1.0, 0.3, 0.3)
    radii, counts = _reference_intersections(skeleton, center, 1.5, spacing)
    got = arb.sholl_analysis(skeleton, center, 1.5, 0.0, spacing)
    assert np.array_equal(got.intersections, counts)


@pytest.mark.parametrize("size", [1, 2, 10, 47, 48, 49, 400])
def test_component_count_matches_labelling_for_small_and_large_groups(size):
    mask = np.zeros((60, 60), bool)
    flat = RNG.choice(mask.size, size=size, replace=False)
    mask.flat[flat] = True
    coords = np.argwhere(mask)
    assert arb._count_components(coords, mask.shape) == (int(measure.label(mask).max()))
    assert arb._count_components(np.empty((0, 2), int), mask.shape) == 0


# ---------------------------------------------------------------- combined skeleton


def test_the_shared_skeleton_equals_skeletonising_the_whole_mask(tmp_path):
    image = _blobs_with_arms()
    cfg = quick_config(tmp_path)
    cfg.arbor.enabled = True
    result = analyze_array(image, "cells", cfg)
    assert result.object_count >= 8
    whole = arb.skeleton_metrics(result.labels > 0, cfg.arbor.prune_length, 1.0).skeleton
    assert np.array_equal(result.skeleton, whole)


def test_watershed_mode_still_skeletonises_the_whole_mask(tmp_path):
    cfg = quick_config(tmp_path)
    cfg.arbor.enabled = True
    cfg.segmentation.watershed_split = True
    result = analyze_array(_blobs_with_arms(), "cells", cfg)
    whole = arb.skeleton_metrics(result.labels > 0, cfg.arbor.prune_length, 1.0).skeleton
    assert np.array_equal(result.skeleton, whole)


# ---------------------------------------------------------------- pictures


def test_pictures_are_the_same_pixels_just_encoded_faster(tmp_path):
    gray = RNG.random((50, 70)).astype(np.float32)
    rgb = RNG.random((40, 60, 3)).astype(np.float32)
    save_uint8(tmp_path / "g.png", gray)
    save_uint8(tmp_path / "c.png", rgb, rescale=False)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "g.png")),
                          (((gray - gray.min()) / (gray.max() - gray.min())) * 255).astype(np.uint8))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "c.png")), (rgb * 255).astype(np.uint8))


def test_several_pictures_can_be_written_together_and_errors_still_surface(tmp_path):
    jobs = [(tmp_path / f"{i}.png", RNG.random((30, 30)).astype(np.float32), True) for i in range(6)]
    save_uint8_many(jobs)
    assert all((tmp_path / f"{i}.png").exists() for i in range(6))
    with pytest.raises(ValueError, match="empty"):
        save_uint8_many([(tmp_path / "a.png", np.ones((4, 4)), True),
                         (tmp_path / "b.png", np.empty((0, 0)), True)])


# ---------------------------------------------------------------- workers


def test_auto_workers_never_exceeds_cores_images_or_memory(tmp_path, monkeypatch):
    paths = write_synthetic_folder(tmp_path / "in")
    assert 1 <= auto_workers(paths) <= min(len(paths), os.cpu_count() or 1)
    assert auto_workers(paths[:1]) == 1
    from hive import io_utils
    monkeypatch.setattr(io_utils, "_physical_memory_bytes", lambda: 1_000_000)    # a 1 MB computer
    assert auto_workers(paths) == 1


def test_the_memory_estimate_comes_from_the_image_header(tmp_path):
    Image.new("RGB", (2000, 1000)).save(tmp_path / "a.png")
    assert _peak_bytes_per_image(tmp_path / "a.png") == 2000 * 1000 * 4 * 10     # one channel at a time
    assert _peak_bytes_per_image(tmp_path / "missing.png") is None


def test_automatic_workers_give_the_same_tables_as_one_worker(tmp_path):
    write_synthetic_folder(tmp_path / "in")
    one = analyze_folder(quick_config(tmp_path, output_folder=str(tmp_path / "one")), workers=1)
    auto = analyze_folder(quick_config(tmp_path, output_folder=str(tmp_path / "auto")), workers=None)
    pd.testing.assert_frame_equal(one["measurements"], auto["measurements"])
    pd.testing.assert_frame_equal(one["summary"], auto["summary"])


def test_the_command_line_accepts_auto_or_a_number(tmp_path):
    import argparse
    from hive_cli import _workers_arg, build_parser
    assert _workers_arg("auto") is None and _workers_arg("3") == 3
    for bad in ("0", "-2", "many"):
        with pytest.raises(argparse.ArgumentTypeError):
            _workers_arg(bad)
    assert build_parser().parse_args([]).workers is None


# ---------------------------------------------------------------- Excel and zip


def test_excel_report_has_the_same_content_with_either_writer(tmp_path, monkeypatch):
    import sys
    from hive import reporting
    summary = pd.DataFrame({"image": ["a", "b"], "object_count": [1, 2]})
    meas = pd.DataFrame({"image": ["a"] * 3, "label": [1, 2, 3], "x": [0.5, 1.5, np.nan]})
    reporting.export_excel_report(tmp_path / "fast.xlsx", summary, meas, settings={"k": 1})
    monkeypatch.setitem(sys.modules, "xlsxwriter", None)             # pretend it is not installed
    reporting.export_excel_report(tmp_path / "slow.xlsx", summary, meas, settings={"k": 1})
    fast, slow = (pd.read_excel(tmp_path / f"{n}.xlsx", sheet_name=None) for n in ("fast", "slow"))
    assert list(fast) == list(slow) and all(fast[k].equals(slow[k]) for k in fast)


def test_the_results_zip_keeps_every_file_and_stores_compressed_ones(tmp_path):
    import app
    (tmp_path / "res").mkdir()
    Image.new("L", (40, 40)).save(tmp_path / "res" / "a.png")
    (tmp_path / "res" / "t.csv").write_text("a,b\n" + "1,2\n" * 500)
    (tmp_path / "res" / "sub").mkdir()
    (tmp_path / "res" / "sub" / "n.txt").write_text("hello")
    data = app._zip_bytes(tmp_path / "res")
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        assert sorted(bundle.namelist()) == ["a.png", "sub/n.txt", "t.csv"]
        kinds = {i.filename: i.compress_type for i in bundle.infolist()}
        assert kinds["a.png"] == zipfile.ZIP_STORED and kinds["t.csv"] == zipfile.ZIP_DEFLATED
        assert bundle.read("sub/n.txt") == b"hello" and bundle.testzip() is None
