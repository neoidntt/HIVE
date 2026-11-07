"""When the channel is "auto", the result says which channel HIVE chose."""

from __future__ import annotations

import json

import numpy as np
import tifffile

from hive import tiled
from hive.config import LargeImageConfig
from hive.pipeline import analyze_array, analyze_folder, analyze_zstack

from helpers import blob_volume, quick_config


def _colour(seed=0, shape=(120, 140)):
    """RGB image whose green channel carries the signal, so "auto" must choose green."""
    rng = np.random.default_rng(seed)
    image = rng.normal(20, 2, shape + (3,)).clip(0, None).astype(np.float32)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    for cy, cx in ((30, 30), (30, 100), (90, 60), (90, 110)):
        image[((yy - cy) ** 2 + (xx - cx) ** 2) < 64, 1] += 400
    return image


def test_auto_reports_the_channel_it_chose(tmp_path):
    cfg = quick_config(tmp_path)                      # channel is "auto" by default
    result = analyze_array(_colour(), "cells.tif", cfg)
    assert result.prepared.channel_name == "green"
    assert len(result.notes) == 1 and "cells.tif" in result.notes[0]
    assert "analysed the green channel" in result.notes[0] and "auto" in result.notes[0]


def test_a_named_channel_or_a_grey_image_gets_no_note(tmp_path):
    cfg = quick_config(tmp_path)
    assert analyze_array(_colour(), "x.tif", quick_config(tmp_path, channel="red")).notes == []
    assert analyze_array(_colour()[:, :, 1], "gray.tif", cfg).notes == []


def test_the_note_reaches_a_batch_and_the_saved_record(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    tifffile.imwrite(folder / "a.tif", _colour(1))
    tifffile.imwrite(folder / "b.tif", _colour(2))
    cfg = quick_config(tmp_path, input_path=str(folder), output_folder=str(tmp_path / "out"))
    batch = analyze_folder(cfg)
    assert len(batch["notes"]) == 2 and all("green" in n for n in batch["notes"])
    record = json.loads((tmp_path / "out" / "hive_run.json").read_text())
    assert any("green" in n for n in record["notes"]["notes"])


def test_a_colour_stack_in_3d_reports_it_too(tmp_path):
    volume = blob_volume()
    stack = np.stack([np.zeros_like(volume), volume, np.zeros_like(volume)], axis=-1)   # (Z, Y, X, 3)
    path = tmp_path / "rgb_stack.tif"
    tifffile.imwrite(path, stack, photometric="rgb")
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(tmp_path / "o3"), mode="3d")
    cfg.large_image = LargeImageConfig(mode="off")
    result = analyze_zstack(cfg)
    assert any("analysed the green channel" in n for n in result.notes)
    record = json.loads((tmp_path / "o3" / "hive_run.json").read_text())
    assert any("green" in n for n in record["notes"]["notes"])


def test_a_tiled_folder_of_colour_slices_uses_one_channel_for_every_slice(tmp_path):
    folder = tmp_path / "slices"
    folder.mkdir()
    rng = np.random.default_rng(3)
    for z in range(4):
        plane = rng.normal(20, 2, (60, 70, 3)).astype(np.float32)
        plane[20:40, 20:40, 1] += 300 if z != 2 else 0.0     # slice 2 alone has no green signal
        plane[:, :, 0] += rng.normal(0, 30, (60, 70)) if z == 2 else 0.0   # ...but a noisy red
        tifffile.imwrite(folder / f"s{z}.tif", plane)
    with tiled.open_source(folder, "auto") as source:
        assert any("analysed the green channel" in n for n in source.notes)
        green = source.read(0, slice(None), slice(None))
        # slice 2 must be read with the same (green) channel, not whichever has most contrast there
        assert np.allclose(source.read(2, slice(None), slice(None)),
                           tifffile.imread(folder / "s2.tif")[:, :, 1])
    assert green.shape == (60, 70)
