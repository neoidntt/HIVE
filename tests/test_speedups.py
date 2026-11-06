"""The fast routines must give exactly what the reference implementations give."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.morphology import convex_hull_image

from hive import cleanup
from hive.arborization import _neighbour_count
from hive.measurements import convex_voxel_count, measure_objects
from hive.segmentation import _dilate_cross, _erode_cross


def _blobs(rng, shape, sigma=(1.0, 2.5), keep=(0.4, 0.55)):
    field = ndi.gaussian_filter(rng.random(shape), rng.uniform(*sigma))
    return field > np.quantile(field, rng.uniform(*keep))


def test_convex_voxel_count_equals_scikit_image():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(60):
        mask = _blobs(rng, tuple(rng.integers(5, 36, 3)))
        if mask.sum() < 5:
            continue
        try:
            expected = int(convex_hull_image(mask).sum())
        except Exception:  # noqa: BLE001 - degenerate shapes scikit-image itself refuses
            continue
        assert convex_voxel_count(mask) == expected
        checked += 1
    assert checked > 30


def test_3d_solidity_column_equals_regionprops():
    rng = np.random.default_rng(2)
    labels, _ = ndi.label(_blobs(rng, (20, 48, 48), keep=(0.55, 0.65)))
    image = rng.random(labels.shape, dtype=np.float32)
    table = measure_objects(labels, image)
    expected = measure.regionprops_table(labels, properties=("label", "solidity"))
    np.testing.assert_array_equal(table["label"], expected["label"])
    np.testing.assert_array_equal(table["solidity"], expected["solidity"])
    assert list(table.columns).index("solidity") == list(table.columns).index("extent") + 1


def test_cross_morphology_equals_scikit_image_in_2d_and_3d():
    rng = np.random.default_rng(3)
    for trial in range(120):
        ndim = 2 + trial % 2
        mask = rng.random(tuple(rng.integers(1, 28, ndim))) < rng.uniform(0.05, 0.95)
        footprint = morphology.disk(1) if ndim == 2 else morphology.ball(1)
        np.testing.assert_array_equal(
            _erode_cross(_dilate_cross(_dilate_cross(_erode_cross(mask)))),
            morphology.closing(morphology.opening(mask, footprint), footprint))


def test_neighbour_count_equals_convolution():
    rng = np.random.default_rng(4)
    for trial in range(80):
        ndim = 2 + trial % 2
        skeleton = rng.random(tuple(rng.integers(1, 22, ndim))) < rng.uniform(0.05, 0.9)
        kernel = np.ones((3,) * ndim, dtype=np.uint8)
        kernel[(1,) * ndim] = 0
        np.testing.assert_array_equal(
            _neighbour_count(skeleton), ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant"))


@pytest.mark.parametrize("hole_size", [-1, 10, 200])
def test_local_hole_filling_equals_whole_image_filling(hole_size, monkeypatch):
    rng = np.random.default_rng(5)
    for trial in range(25):
        shape = (12, 40, 40) if trial % 2 else (60, 60)
        labels, _ = ndi.label(_blobs(rng, shape, keep=(0.4, 0.6)))
        punched = rng.random(shape) < 0.02
        labels = np.where(punched & ndi.binary_dilation(labels > 0), 0, labels)
        local, n_local = cleanup.fill_object_holes(labels.copy(), hole_size)
        monkeypatch.setattr(cleanup, "_MAX_LOCAL_HOLES", -1)       # force the whole-image route
        whole, n_whole = cleanup.fill_object_holes(labels.copy(), hole_size)
        monkeypatch.undo()
        assert n_local == n_whole
        np.testing.assert_array_equal(local, whole)


def test_clahe_matches_scikit_image_bit_for_bit():
    from skimage import exposure

    from hive import clahe

    rng = np.random.default_rng(6)
    for _ in range(25):
        shape = (int(rng.integers(16, 220)), int(rng.integers(16, 220)))
        image = (rng.random(shape) ** rng.uniform(0.5, 3)).astype(np.float32)
        for limit in (0.01, 0.03, 0.0):
            expected = exposure.equalize_adapthist(image, clip_limit=limit)
            got = clahe.equalize_adapthist(image, clip_limit=limit)
            assert got.dtype == expected.dtype
            np.testing.assert_array_equal(got, expected)
    assert clahe._checked is True            # the self-check passed, so the fast path was used


def test_png_writer_round_trips_exactly(tmp_path, monkeypatch):
    from PIL import Image

    from hive import io_utils

    rng = np.random.default_rng(7)
    monkeypatch.setattr(io_utils, "_PNG_BAND_BYTES", 500)         # force several bands
    for shape in ((1, 1), (7, 13), (40, 33, 3), (65, 2, 3), (300, 17)):
        picture = rng.integers(0, 256, shape, dtype=np.uint8)
        io_utils.write_png(tmp_path / "a.png", picture)
        np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "a.png")), picture)


def test_overlays_equal_the_scikit_image_versions():
    from skimage import color, segmentation

    from hive import visualization
    from hive.cleanup import normalize_for_display

    rng = np.random.default_rng(8)
    for trial in range(60):
        shape = tuple(rng.integers(1, 60, 2))
        labels, _ = ndi.label(_blobs(rng, shape, keep=(0.5, 0.7)))
        labels = (labels * (3 if trial % 4 == 0 else 1)).astype(np.int32)
        image = (rng.random(shape) * rng.integers(1, 5000)).astype(np.float32)
        np.testing.assert_array_equal(
            visualization._thick_boundaries(labels),
            segmentation.find_boundaries(labels, mode="thick", background=0))
        np.testing.assert_array_equal(
            visualization.label_overlay_f32(image, labels),
            color.label2rgb(labels, image=normalize_for_display(image), bg_label=0,
                            alpha=0.45).astype(np.float32))
