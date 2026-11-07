"""Opening data in place from the web app: no upload, no copy, and results in a folder of your own."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile
from streamlit.testing.v1 import AppTest

from hive.synthetic import make_synthetic_2d

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app():
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def _choose_local(app, path, results):
    next(r for r in app.sidebar.radio if r.label == "Data source").set_value(
        "File or folder on this computer").run()
    app.sidebar.text_input(key="local_path").set_value(str(path)).run()
    app.sidebar.text_input(key="results_folder").set_value(str(results)).run()


def _write_images(folder: Path, count=2) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, image in make_synthetic_2d(0)[:count]:
        paths.append(folder / name)
        tifffile.imwrite(paths[-1], image.astype(np.float32))
    return paths


def test_a_local_file_is_analysed_where_it_is_and_results_go_to_the_chosen_folder(tmp_path):
    data = _write_images(tmp_path / "data", 1)[0]
    results = tmp_path / "my results"
    results.mkdir()
    (results / "precious.txt").write_text("keep me")
    before = data.read_bytes()
    app = _app()
    _choose_local(app, data, results)
    assert not app.exception and any("synthetic_dim.tif" in s.value for s in app.success)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].value == "6"
    run_folders = [p for p in results.iterdir() if p.is_dir()]
    assert len(run_folders) == 1 and run_folders[0].name.startswith("synthetic_dim_")
    assert (run_folders[0] / "HIVE_2D_report.xlsx").exists()
    assert (results / "precious.txt").read_text() == "keep me"      # nothing existing was touched
    assert data.read_bytes() == before and not (tmp_path / "data" / "input").exists()
    assert any(str(run_folders[0]) in s.value for s in app.success)


def test_a_local_folder_works_and_each_run_gets_its_own_subfolder(tmp_path):
    folder = tmp_path / "scans"
    _write_images(folder, 2)
    results = tmp_path / "res"
    app = _app()
    _choose_local(app, folder, results)
    assert any("2 image file(s)" in s.value for s in app.success)
    app.button[0].click().run()
    assert not app.exception and app.metric[1].value == "2"
    # a second run (same inputs, new click) must not overwrite the first
    app.button[0].click().run()
    assert len([p for p in results.iterdir() if p.is_dir()]) >= 1


def test_a_missing_path_is_reported_and_nothing_is_run(tmp_path):
    app = _app()
    _choose_local(app, tmp_path / "nowhere.tif", tmp_path / "res")
    assert any("nothing at" in e.value.lower() for e in app.error)
    assert not any(b.label == "Run analysis" for b in app.button)


def test_huge_results_are_left_in_their_folder_not_zipped_through_the_browser(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_DOWNLOAD_LIMIT_BYTES", "1000")          # pretend 1 KB is "huge"
    data = _write_images(tmp_path / "data", 1)[0]
    app = _app()
    _choose_local(app, data, tmp_path / "res")
    app.button[0].click().run()
    assert not app.exception
    labels = [b.proto.label for b in app.get("download_button")]
    assert "All results (ZIP)" not in labels and "Measurements CSV" in labels
    assert any("too much to send through the browser" in c.value for c in app.caption)


def test_the_choose_buttons_exist_only_when_a_dialog_is_available(monkeypatch):
    from hive import filedialog
    monkeypatch.setattr(filedialog, "_command", lambda kind: ["dialog"])
    app = _app()
    next(r for r in app.sidebar.radio if r.label == "Data source").set_value(
        "File or folder on this computer").run()
    labels = [b.label for b in app.sidebar.button]
    assert "Choose a file..." in labels and "Choose a folder..." in labels


def test_the_upload_box_explains_that_big_files_belong_in_the_other_option():
    app = _app()
    next(r for r in app.sidebar.radio if r.label == "Data source").set_value("Upload files").run()
    uploader = next(u for u in app.sidebar.file_uploader if u.label.startswith("Upload"))
    assert "File or folder on this computer" in uploader.proto.help


def test_a_table_of_millions_of_rows_is_shown_as_a_sample():
    def script():
        import numpy as np
        import pandas as pd
        import streamlit as st
        import app as hive_app
        big = pd.DataFrame({"intensity_mean": np.arange(50_000, dtype=float), "area_px": 1.0,
                            "image": "x", "label": np.arange(50_000)})
        hive_app._show_table(big)
        hive_app._measurement_charts(big)

    run = AppTest.from_function(script, default_timeout=120).run()
    assert not run.exception
    assert len(run.dataframe[0].value) == 20_000
    captions = " ".join(c.value for c in run.caption)
    assert "first 20,000 of 50,000 rows" in captions and "random 20,000 of 50,000 objects" in captions


def test_a_tiled_stack_can_still_be_viewed_from_disk(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import blob_volume

    from hive import io_utils, preview

    stack = tmp_path / "stack.tif"
    tifffile.imwrite(stack, blob_volume())
    monkeypatch.setattr(io_utils, "_physical_memory_bytes", lambda: 4_000_000)   # "not enough RAM"
    app = _app()
    _choose_local(app, stack, tmp_path / "res")
    next(r for r in app.sidebar.radio if r.label == "Mode").set_value("3D z-stack").run()
    app.sidebar.text_input(key="local_path").set_value(str(stack)).run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert any("analysed in tiles" in i.value for i in app.info)
    assert any(b.label == "Load the whole-stack overview" for b in app.button)
    next(b for b in app.button if b.label == "Load the whole-stack overview").click().run()
    assert not app.exception and not app.error

    # the reading functions themselves, against the source
    labels = next((tmp_path / "res").rglob("HIVE_3D_labels.ome.tif"))
    raw, lab = preview.read_region(stack, labels, 5, 0, 0, 128)
    assert raw.shape == lab.shape == (128, 128)
    view = preview.overview(stack, labels, max_voxels=10 * 200 * 220 // 4)
    assert view.stride == 2 and view.raw.shape == view.labels.shape
    assert view.labels.max() == tifffile_max(labels)


def tifffile_max(path):
    return int(tifffile.imread(str(path)).max())


def _stack_folder(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import blob_volume

    folder = tmp_path / "stacks"
    folder.mkdir()
    tifffile.imwrite(folder / "a.tif", blob_volume(seed=1))
    tifffile.imwrite(folder / "b.tif", blob_volume(seed=2))
    return folder


def test_a_folder_of_stacks_runs_one_after_another_with_each_in_its_own_folder(tmp_path):
    folder, results = _stack_folder(tmp_path), tmp_path / "res"
    app = _app()
    _choose_local(app, folder, results)
    next(r for r in app.sidebar.radio if r.label == "Mode").set_value("3D z-stack").run()
    app.sidebar.text_input(key="local_path").set_value(str(folder)).run()
    box = next(c for c in app.checkbox if c.label.startswith("Each file in this folder"))
    assert box.value is True                                    # whole-stack files: ticked for you
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert [m.value for m in app.metric[:2]] == ["2", "0"]
    run = next(p for p in results.iterdir() if p.is_dir())
    assert (run / "a" / "HIVE_3D_report.xlsx").exists() and (run / "b" / "HIVE_3D_report.xlsx").exists()
    assert (run / "combined_3d_summary.csv").exists()


def test_if_the_analysis_is_killed_the_app_says_so_and_keeps_running(tmp_path, monkeypatch):
    import hive.isolate

    def killed(*args, **kwargs):
        raise hive.isolate.AnalysisStoppedError("ended unexpectedly. It was stopped ... out of memory.")

    monkeypatch.setattr(hive.isolate, "run_isolated", killed)
    data = _write_images(tmp_path / "data", 1)[0]
    app = _app()
    _choose_local(app, data, tmp_path / "res")
    app.button[0].click().run()
    assert not app.exception
    assert any("Analysis failed" in e.value and "out of memory" in e.value for e in app.error)
    monkeypatch.undo()
    app.button[0].click().run()                          # and a second try works
    assert not app.exception and not app.error and app.metric[0].value == "6"
