"""Files larger than memory: tiled analysis of single images, parallel tiles, reading in place."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from hive import filedialog, io_utils, pipeline, tiled
from hive.config import LargeImageConfig
from hive.io_utils import ImageLoadError
from hive.pipeline import ImageTooLargeError, analyze_array, analyze_folder, analyze_image

from helpers import blob_volume, quick_config

RNG = np.random.default_rng(31)


def _plane(h=240, w=300, seed=2):
    """A noisy 2D field with separate bright discs (one sits on a tile seam)."""
    rng = np.random.default_rng(seed)
    img = (100 + rng.normal(0, 6, (h, w))).astype(np.float32)
    yy, xx = np.ogrid[:h, :w]
    for cy, cx in [(40, 40), (40, 150), (120, 60), (120, 128), (200, 40), (200, 220), (60, 270), (130, 200)]:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < 100] += 900
    return img


def _cfg(tmp_path, path, mode="force", tile=64, halo=0, workers=0, out="out", **kw):
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(tmp_path / out), **kw)
    cfg.preprocess.gaussian_sigma = 1.0
    cfg.large_image = LargeImageConfig(mode=mode, tile_size=tile, halo=halo, workers=workers)
    return cfg


@pytest.fixture
def plane_file(tmp_path):
    path = tmp_path / "plane.tif"
    tifffile.imwrite(path, _plane())
    return path


# ------------------------------------------------------------------ 2D tiles equal the whole image

def test_a_tiled_plane_finds_the_same_objects_as_the_whole_image(tmp_path, plane_file):
    whole = analyze_array(_plane(), "plane.tif", _cfg(tmp_path, plane_file, mode="off"))
    tiles = analyze_image(plane_file, _cfg(tmp_path, plane_file, tile=64, workers=1))
    assert whole.object_count == tiles.object_count == 8
    key = ["centroid-0", "centroid-1"]
    a = whole.measurements.sort_values(key).reset_index(drop=True)
    b = tiles.measurements.sort_values(key).reset_index(drop=True)
    assert a["area_px"].tolist() == b["area_px"].tolist()
    np.testing.assert_allclose(a["intensity_mean"], b["intensity_mean"], rtol=0.01)
    np.testing.assert_allclose(a[key].to_numpy(), b[key].to_numpy(), atol=0.01)
    assert not b["tile_clipped"].any()
    # the same table as a normal 2D run, plus the tile flag, minus branching
    assert {"area_px", "perimeter", "solidity", "eccentricity", "integrated_intensity"} <= set(b.columns)


def test_the_label_file_matches_and_is_sparse(tmp_path, plane_file):
    whole = analyze_array(_plane(), "plane.tif", _cfg(tmp_path, plane_file, mode="off"))
    tiles = analyze_image(plane_file, _cfg(tmp_path, plane_file, tile=64, workers=1))
    labels = np.asarray(tiles.labels)
    assert labels.shape == whole.labels.shape and labels.dtype == np.uint32
    assert np.array_equal(labels > 0, whole.labels > 0)
    assert sorted(set(labels[labels > 0])) == sorted(tiles.measurements["label"])
    with tifffile.TiffFile(tiles.outputs["labels_ome_tiff"]) as tif:
        assert tif.is_ome and tif.series[0].axes == "YX"
    assert tiles.outputs["measurements_csv"].exists()
    assert tiles.prepared is None and "tiles" in tiles.notes[0]


def test_parallel_tiles_give_exactly_the_same_result_as_one_worker(tmp_path, plane_file):
    one = analyze_image(plane_file, _cfg(tmp_path, plane_file, workers=1, out="a"))
    many = analyze_image(plane_file, _cfg(tmp_path, plane_file, workers=3, out="b"))
    pd.testing.assert_frame_equal(one.measurements, many.measurements)
    assert np.array_equal(np.asarray(one.labels), np.asarray(many.labels))


def test_parallel_tiles_also_work_for_stacks(tmp_path):
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, blob_volume())
    outputs = []
    for workers in (1, 3):
        cfg = _cfg(tmp_path, path, workers=workers, out=f"s{workers}", mode="force", tile=64)
        cfg.mode = "3d"
        cfg.preprocess.tophat_radius = (2.0, 12.0, 12.0)
        outputs.append(pipeline.analyze_zstack(cfg))
    assert outputs[0].object_count == outputs[1].object_count == 9
    pd.testing.assert_frame_equal(outputs[0].measurements, outputs[1].measurements)
    assert np.array_equal(np.asarray(outputs[0].labels), np.asarray(outputs[1].labels))


def test_tiled_blank_images_and_clear_border(tmp_path):
    flat = tmp_path / "flat.tif"
    tifffile.imwrite(flat, np.full((120, 120), 50, np.float32))
    with pytest.warns(UserWarning, match="uniform"):
        assert analyze_image(flat, _cfg(tmp_path, flat, out="f")).object_count == 0
    plane = tmp_path / "p.tif"
    img = _plane()
    img[0:16, 90:106] += 900                                # a bright block on the top edge
    tifffile.imwrite(plane, img)
    keep = analyze_image(plane, _cfg(tmp_path, plane, out="k", workers=1))
    cfg = _cfg(tmp_path, plane, out="c", workers=1)
    cfg.segmentation.clear_border = True
    dropped = analyze_image(plane, cfg)
    assert keep.object_count == 9 and dropped.object_count == 8   # the edge block is dropped


# ------------------------------------------------------------------ when tiles are used

def test_only_tiffs_that_do_not_fit_are_tiled(tmp_path, plane_file, monkeypatch):
    cfg = _cfg(tmp_path, plane_file, mode="auto")
    assert not pipeline._needs_tiled_2d(plane_file, cfg)
    monkeypatch.setattr(io_utils, "_physical_memory_bytes", lambda: 100_000)         # a 100 KB computer
    assert pipeline._needs_tiled_2d(plane_file, cfg)
    cfg.large_image.mode = "off"
    assert not pipeline._needs_tiled_2d(plane_file, cfg)
    png = tmp_path / "x.png"
    Image.new("L", (300, 300)).save(png)
    cfg.large_image.mode = "force"
    assert not pipeline._needs_tiled_2d(png, cfg)           # a PNG cannot be read in pieces


def test_a_batch_mixes_normal_and_tiled_images_in_name_order(tmp_path, monkeypatch):
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("a.tif", "b.tif", "c.tif"):
        tifffile.imwrite(folder / name, _plane(seed=hash(name) % 7))
    real = pipeline._needs_tiled_2d
    monkeypatch.setattr(pipeline, "_needs_tiled_2d", lambda p, cfg: p.name == "b.tif" or real(p, cfg))
    cfg = _cfg(tmp_path, folder, mode="auto", out="mix")
    out = analyze_folder(cfg, workers=1)
    assert out["summary"]["image"].tolist() == ["a.tif", "b.tif", "c.tif"]
    assert not out["errors"] and any("analysed in" in n for n in out["notes"])
    assert (tmp_path / "mix" / "b_labels.ome.tif").exists()       # the tiled one wrote its own
    assert (tmp_path / "mix" / "a_boundary_overlay.png").exists()  # the others the usual pictures


def test_a_single_file_can_be_analysed_where_it_is(tmp_path, plane_file):
    cfg = _cfg(tmp_path, plane_file, mode="off")
    out = analyze_folder(cfg, workers=1)                     # input_path is a file, not a folder
    assert len(out["summary"]) == 1 and out["summary"]["object_count"].iloc[0] == 8
    assert not any(tmp_path.glob("in/*"))                    # nothing was copied anywhere


def test_an_image_that_cannot_be_tiled_or_fit_still_gets_a_clear_message(tmp_path, monkeypatch):
    big = tmp_path / "big.png"
    Image.new("L", (8000, 8000)).save(big)
    monkeypatch.setattr(pipeline, "_available_memory_bytes", lambda: 500_000_000)
    with pytest.raises(ImageTooLargeError, match="is free"):
        analyze_image(big, quick_config(tmp_path))


def test_a_format_that_can_only_be_read_whole_is_refused_when_too_big(tmp_path, monkeypatch):
    from hive import formats
    fake = tmp_path / "huge.nd2"
    fake.write_bytes(b"")
    meta = formats.MicroscopeImage(None, (2, 100, 20000, 20000), ["a", "b"])
    monkeypatch.setattr(formats, "read_microscope", lambda path, load_pixels=True: meta)
    monkeypatch.setattr(pipeline, "_available_memory_bytes", lambda: 4_000_000_000)
    with pytest.raises(ImageTooLargeError, match="Export it as an \\(OME-\\)TIFF"):
        pipeline._check_vendor_file_fits(fake)
    monkeypatch.setattr(pipeline, "_available_memory_bytes", lambda: 400_000_000_000)
    pipeline._check_vendor_file_fits(fake)                   # plenty of room: no error


# ------------------------------------------------------------------ reading compressed files in pieces

def test_a_compressed_tiled_tiff_is_read_in_pieces_with_the_same_pixels(tmp_path):
    pytest.importorskip("zarr")
    data = (RNG.random((900, 1100)) * 4000).astype(np.uint16)
    path = tmp_path / "c.tif"
    tifffile.imwrite(path, data, compression="zlib", tile=(128, 128), bigtiff=True)
    with tiled.open_source(path, "auto") as src:
        assert any("Compressed file" in n for n in src.notes)
        piece = src.read(0, slice(300, 700), slice(500, 1000))
    assert np.array_equal(piece, data[300:700, 500:1000].astype(np.float32))


def test_without_zarr_a_big_compressed_plane_is_refused_and_a_small_one_still_works(tmp_path, monkeypatch):
    data = (RNG.random((300, 400)) * 4000).astype(np.uint16)
    path = tmp_path / "c.tif"
    tifffile.imwrite(path, data, compression="zlib")
    monkeypatch.setitem(sys.modules, "zarr", None)             # pretend zarr is not installed
    with tiled.open_source(path, "auto") as src:
        assert any("every plane is decoded" in n for n in src.notes)
        assert np.array_equal(src.read(0, slice(0, 50), slice(0, 60)), data[:50, :60].astype(np.float32))
    monkeypatch.setattr(tiled, "_available_memory_bytes", lambda: 1_000_000)
    with pytest.raises(ImageLoadError, match="pip install zarr"):
        with tiled.open_source(path, "auto"):
            pass


def test_a_multichannel_ome_plane_is_tiled_by_channel_name(tmp_path):
    from helpers import write_ome
    both = np.stack([np.full((200, 200), 30, np.float32), _plane(200, 200)])
    path = write_ome(tmp_path / "two.ome.tif", both, "CYX", ["empty", "cells"])
    result = analyze_image(path, _cfg(tmp_path, path, channel="cells", workers=1))
    assert result.object_count >= 5


# ------------------------------------------------------------------ the file dialog

def test_each_system_gets_its_own_dialog_command(monkeypatch):
    monkeypatch.setattr(filedialog.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sys, "platform", "linux")
    assert filedialog._command("file")[:2] == ["zenity", "--file-selection"]
    assert "--directory" in filedialog._command("folder")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert filedialog._command("folder")[0] == "osascript" and "choose folder" in filedialog._command("folder")[2]
    monkeypatch.setattr(sys, "platform", "win32")
    command = filedialog._command("file")
    assert command[0] == "powershell" and "OpenFileDialog" in command[-1]
    assert "FolderBrowserDialog" in filedialog._command("folder")[-1]


def test_the_chosen_path_is_returned_and_cancelling_gives_none(monkeypatch):
    import subprocess
    monkeypatch.setattr(filedialog, "_command", lambda kind: ["dialog"])
    done = lambda out, code=0: subprocess.CompletedProcess(["dialog"], code, out, "")
    monkeypatch.setattr(filedialog.subprocess, "run", lambda *a, **k: done("/data/big stack.tif\n"))
    assert filedialog.pick("file") == "/data/big stack.tif"
    monkeypatch.setattr(filedialog.subprocess, "run", lambda *a, **k: done("", 1))
    assert filedialog.pick("file") is None                  # cancelled
    monkeypatch.setattr(filedialog.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert filedialog.pick("folder") is None
    with pytest.raises(ValueError):
        filedialog.pick("both")


def test_with_no_dialog_tool_the_app_falls_back_to_typing(monkeypatch):
    monkeypatch.setattr(filedialog, "_command", lambda kind: None)
    assert not filedialog.available() and filedialog.pick("file") is None


# ------------------------------------------------------------------ disk use

def test_a_sparse_file_counts_only_what_is_really_written(tmp_path):
    import app
    sparse = tmp_path / "big.bin"
    with open(sparse, "wb") as handle:
        handle.truncate(2_000_000_000)                       # 2 GB apparent, nothing written
        handle.seek(1000)
        handle.write(b"x" * 4096)
    if not hasattr(sparse.stat(), "st_blocks"):
        pytest.skip("this platform does not report allocated blocks")
    assert sparse.stat().st_size == 2_000_000_000 and app._disk_bytes(sparse) < 5_000_000


# ------------------------------------------------------------------ what millions of objects would break

def test_a_table_too_long_for_excel_is_replaced_by_a_note_not_a_crash(tmp_path, monkeypatch):
    from hive import reporting
    monkeypatch.setattr(reporting, "EXCEL_MAX_ROWS", 5)
    summary = pd.DataFrame({"image": ["a"], "object_count": [9]})
    measurements = pd.DataFrame({"label": range(9), "x": 1.0})
    reporting.export_excel_report(tmp_path / "r.xlsx", summary, measurements, settings={"k": 1},
                                  sholl_curves=pd.DataFrame({"label": range(9), "radius": 1.0}))
    sheets = pd.read_excel(tmp_path / "r.xlsx", sheet_name=None)
    assert "more than an Excel sheet can hold" in sheets["Measurements"]["note"].iloc[0]
    assert "9 rows" in sheets["Sholl_Curves"]["note"].iloc[0]
    assert len(sheets["Summary"]) == 1                                  # small tables are untouched
    pd.testing.assert_frame_equal(reporting._fit_sheet(summary, "Summary"), summary)


def test_the_real_excel_limit_is_the_one_excel_has():
    from hive import reporting
    assert reporting.EXCEL_MAX_ROWS == 2**20 - 1
