"""\"Don't save results to this computer\": the analysis runs and returns results, nothing is written,
and any scratch file is gone afterwards."""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from streamlit.testing.v1 import AppTest

from hive import reporting
from hive.config import LargeImageConfig
from hive.pipeline import analyze_folder, analyze_stacks, analyze_zstack
from hive.synthetic import make_synthetic_2d

from helpers import blob_volume, quick_config

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Point the system temp folder at an empty folder we can inspect afterwards."""
    folder = tmp_path / "system_tmp"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder


def _images(folder: Path, count=2) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, image in make_synthetic_2d(0)[:count]:
        tifffile.imwrite(folder / name, image.astype(np.float32))
    return folder


def _nothing_left(*folders: Path) -> bool:
    return all(not folder.exists() or not any(folder.iterdir()) for folder in folders)


def test_2d_batch_writes_nothing_and_still_returns_everything(tmp_path, scratch):
    data, out = _images(tmp_path / "data"), tmp_path / "results"
    cfg = quick_config(tmp_path, input_path=str(data), output_folder=str(out))
    batch = analyze_folder(cfg, workers=1, save=False)
    assert not out.exists() and _nothing_left(scratch)                  # no folder, no temp file
    assert batch["saved"] is False and batch["outputs"] == {}
    assert len(batch["summary"]) == 2 and len(batch["measurements"]) > 0
    assert json.loads(batch["settings_json"])["config"]["mode"] == "2d"  # the settings, in memory
    saved = analyze_folder(cfg, workers=1)                              # and the default is unchanged
    assert saved["saved"] is True and (out / "HIVE_2D_report.xlsx").exists()


def test_a_huge_2d_image_uses_a_scratch_folder_that_is_removed(tmp_path, scratch):
    data, out = _images(tmp_path / "data", 1), tmp_path / "results"
    cfg = quick_config(tmp_path, input_path=str(data), output_folder=str(out))
    cfg.large_image = LargeImageConfig(mode="force", tile_size=64)       # tiles need a disk file
    batch = analyze_folder(cfg, workers=1, save=False)
    assert len(batch["measurements"]) > 0
    assert not out.exists() and _nothing_left(scratch)


def test_3d_in_memory_writes_nothing_and_keeps_its_views_in_memory(tmp_path, scratch):
    path, out = tmp_path / "stack.tif", tmp_path / "results"
    tifffile.imwrite(path, blob_volume())
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(out), mode="3d")
    cfg.large_image = LargeImageConfig(mode="off")
    cfg.render.n_frames = 4
    result = analyze_zstack(cfg, render_gif=True, save=False)
    assert not out.exists() and _nothing_left(scratch)
    assert result.saved is False and result.outputs == {} and result.object_count == 9
    assert result.pages["volume"].startswith("<!doctype html>")          # the interactive view
    assert result.blobs["gif"][:6] in (b"GIF87a", b"GIF89a")             # the rotation, as bytes
    assert json.loads(result.blobs["settings_json"])["config"]["mode"] == "3d"


def test_3d_in_tiles_keeps_no_label_file(tmp_path, scratch):
    path, out = tmp_path / "stack.tif", tmp_path / "results"
    tifffile.imwrite(path, blob_volume())
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(out), mode="3d")
    cfg.large_image = LargeImageConfig(mode="force", tile_size=96)
    result = analyze_zstack(cfg, save=False)
    assert not out.exists() and _nothing_left(scratch)                  # the scratch label file is gone
    assert result.saved is False and result.object_count == 9 and result.labels.size == 0
    assert any("nothing was saved" in note for note in result.notes)


def test_a_folder_of_stacks_writes_nothing(tmp_path, scratch):
    folder, out = tmp_path / "stacks", tmp_path / "results"
    folder.mkdir()
    tifffile.imwrite(folder / "a.tif", blob_volume(seed=1))
    cfg = quick_config(tmp_path, input_path=str(folder), output_folder=str(out), mode="3d")
    batch = analyze_stacks(cfg, save=False)
    assert not out.exists() and _nothing_left(scratch)
    assert batch["saved"] is False and batch["stacks"][0]["folder"] is None and len(batch["measurements"]) == 9


def test_the_excel_report_can_be_made_in_memory():
    buffer = io.BytesIO()
    reporting.export_excel_report(buffer, pd.DataFrame({"image": ["a"], "n": [1]}),
                                  pd.DataFrame({"label": [1, 2]}), settings={"seed": 0})
    sheets = pd.read_excel(io.BytesIO(buffer.getvalue()), sheet_name=None)
    assert {"Summary", "Measurements", "Settings"} <= set(sheets) and len(sheets["Measurements"]) == 2


# ---- the app --------------------------------------------------------------------------------

def _app():
    return AppTest.from_file(str(APP_PATH), default_timeout=300).run()


def _choose_local(app, path, results):
    next(r for r in app.sidebar.radio if r.label == "Data source").set_value(
        "File or folder on this computer").run()
    app.sidebar.text_input(key="local_path").set_value(str(path)).run()
    app.sidebar.text_input(key="results_folder").set_value(str(results)).run()


def test_the_checkbox_is_greyed_out_unless_the_data_is_on_this_computer(tmp_path):
    app = _app()
    box = app.sidebar.checkbox(key="no_save")
    assert box.label == "Don't save results to this computer" and box.disabled     # demo data: greyed out
    _choose_local(app, tmp_path, tmp_path / "res")
    assert not app.sidebar.checkbox(key="no_save").disabled
    assert not app.sidebar.text_input(key="results_folder").disabled


def test_the_app_shows_results_but_writes_nothing_when_asked(tmp_path, scratch):
    data = _images(tmp_path / "data", 2)
    results = tmp_path / "my results"
    results.mkdir()
    app = _app()
    _choose_local(app, data, results)
    app.sidebar.checkbox(key="no_save").set_value(True).run()
    assert app.sidebar.text_input(key="results_folder").disabled          # the save folder is greyed out
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[1].value == "2"                                     # results are on screen
    assert any("Nothing was saved" in i.value for i in app.info)
    assert list(results.iterdir()) == []                                   # no run folder was made
    assert _nothing_left(scratch)
    assert not any(p.name.startswith("hive_nosave_") for p in Path(tempfile.gettempdir()).iterdir())
    # untick it and the same run saves, as before
    app.sidebar.checkbox(key="no_save").set_value(False).run()
    app.button[0].click().run()
    assert len([p for p in results.iterdir() if p.is_dir()]) == 1


def test_a_3d_run_shows_its_viewer_without_saving(tmp_path, scratch):
    stack = tmp_path / "stack.tif"
    tifffile.imwrite(stack, blob_volume())
    results = tmp_path / "res"
    app = _app()
    _choose_local(app, stack, results)
    next(r for r in app.sidebar.radio if r.label == "Mode").set_value("3D z-stack").run()
    app.sidebar.text_input(key="local_path").set_value(str(stack)).run()
    app.sidebar.checkbox(key="no_save").set_value(True).run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert any("3D volume (rotate, zoom, pan)" in s.value for s in app.subheader)
    assert not results.exists() and _nothing_left(scratch)
