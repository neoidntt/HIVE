"""An object lying where two tiles overlap is found by both; it must be counted once.

Before this was handled, such objects were listed twice in the measurement table while the label image held
only the later copy, so real stacks had about 15 % too many objects and regions missing from the label image.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from hive.config import LargeImageConfig
from hive.pipeline import analyze_zstack

from helpers import quick_config


def _stack(seed: int, n=70, shape=(8, 220, 240)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    volume = 20 + rng.normal(0, 3, shape).astype(np.float32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    for _ in range(n):
        cz, cy, cx = rng.integers(1, shape[0] - 1), rng.integers(10, shape[1] - 10), rng.integers(10, shape[2] - 10)
        radius = rng.uniform(5, 11)
        volume[((zz - cz) ** 2 * 3 + (yy - cy) ** 2 + (xx - cx) ** 2) < radius * radius] += rng.uniform(60, 160)
    return volume


def _run(tmp_path, seed: int, mode: str, tile: int = 0):
    path = tmp_path / f"stack{seed}.tif"
    tifffile.imwrite(path, _stack(seed))
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(tmp_path / f"out_{mode}_{seed}"), mode="3d")
    cfg.preprocess.tophat_radius = (2.0, 12.0, 12.0)
    cfg.preprocess.gaussian_sigma = 1.0
    cfg.large_image = LargeImageConfig(mode=mode, tile_size=tile)
    return analyze_zstack(cfg)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tiles_count_each_object_once_like_the_whole_volume(tmp_path, seed):
    whole = _run(tmp_path, seed, "off")
    tiles = _run(tmp_path, seed, "force", tile=40)
    assert tiles.object_count == whole.object_count


def test_every_listed_object_is_in_the_label_image_and_the_repeats_are_reported(tmp_path):
    result = _run(tmp_path, 0, "force", tile=40)
    labels = np.asarray(result.labels)
    in_table = set(result.measurements["label"].tolist())
    assert in_table == set(np.unique(labels).tolist()) - {0}                  # same objects, none missing
    assert all(result.measurements["volume_voxels"] > 0)
    assert any("lay where two tiles overlap" in note for note in result.notes)


# ---- the painter on its own: both ways an object can turn out to be a repeat -----------------------

def _tile_result(mosaic_shape, objects, window=(0, 30, 0, 30)):
    """A hand-made tile result: ``objects`` maps a tile-local label to a boolean (z, y, x) mask."""
    import pandas as pd

    rows, index, local = [], [], []
    for label, mask in objects.items():
        z, y, x = np.nonzero(mask)
        rows.append({"label": label, "bbox-0": z.min(), "bbox-1": y.min(), "bbox-2": x.min(),
                     "bbox-3": z.max() + 1, "bbox-4": y.max() + 1, "bbox-5": x.max() + 1, "tile_clipped": False})
        index.append(np.ravel_multi_index((z, y, x), mosaic_shape))
        local.append(np.full(len(z), label))
    order = np.argsort(np.concatenate(index))
    return {"table": pd.DataFrame(rows), "index": np.concatenate(index)[order], "local": np.concatenate(local)[order],
            "box": window, "shape": mosaic_shape, "clipped": 0}


def _blob(shape, centre, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2) <= radius ** 2


def test_a_smaller_repeat_from_a_later_tile_is_dropped():
    from hive.tiled import _Painter

    shape = (10, 30, 30)
    mosaic = np.zeros(shape, np.uint32)
    painter = _Painter(mosaic, plane=False)
    painter.add(_tile_result(shape, {1: _blob(shape, (5, 15, 15), 8)}))
    painter.add(_tile_result(shape, {1: _blob(shape, (5, 15, 15), 5), 2: _blob(shape, (5, 4, 4), 3)}))
    table = painter.finish()
    assert table["label"].tolist() == [1, 2] and painter.repeats == 1         # the second sighting of the big one
    assert set(np.unique(mosaic)) == {0, 1, 2}


def test_an_earlier_object_that_a_larger_later_one_covers_is_dropped_with_its_leftovers():
    from hive.tiled import _Painter

    shape = (10, 30, 30)
    mosaic = np.zeros(shape, np.uint32)
    painter = _Painter(mosaic, plane=False)
    painter.add(_tile_result(shape, {1: _blob(shape, (5, 15, 15), 4)}))       # a small version first
    painter.add(_tile_result(shape, {1: _blob(shape, (5, 15, 16), 9)}))       # the larger version swallows it
    table = painter.finish()
    assert len(table) == 1 and painter.repeats == 1
    assert set(np.unique(mosaic)) == {0, int(table["label"].iloc[0])}         # no ghost id left in the image
    assert (mosaic == table["label"].iloc[0]).sum() == _blob(shape, (5, 15, 16), 9).sum()


def test_two_neighbouring_objects_that_do_not_overlap_both_stay():
    from hive.tiled import _Painter

    shape = (10, 30, 30)
    mosaic = np.zeros(shape, np.uint32)
    painter = _Painter(mosaic, plane=False)
    painter.add(_tile_result(shape, {1: _blob(shape, (5, 15, 8), 5)}))
    painter.add(_tile_result(shape, {1: _blob(shape, (5, 15, 21), 5)}))
    assert len(painter.finish()) == 2 and painter.repeats == 0
