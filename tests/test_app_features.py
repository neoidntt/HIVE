"""Streamlit app: reloading settings, named channels, file scale, background preview."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import tifffile
from streamlit.testing.v1 import AppTest

from hive.config import HiveConfig

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app():
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def _sidebar(app, kind, label):
    return next(w for w in getattr(app.sidebar, kind) if w.label == label)


def _ome_bytes():
    buffer = io.BytesIO()
    data = np.full((2, 64, 64), 20, np.float32)
    data[0, 10:30, 10:30] += 400
    data[1, 30:50, 30:50] += 400
    tifffile.imwrite(buffer, data, ome=True, metadata={
        "axes": "CYX", "PhysicalSizeX": 0.4, "PhysicalSizeY": 0.4,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": ["DAPI", "GFP"]}})
    return buffer.getvalue()


def test_saved_settings_restore_the_sidebar():
    saved = HiveConfig(seed=42, voxel_size_xy=0.65, voxel_unit="um")
    saved.segmentation.min_size = 123
    saved.segmentation.watershed_split = True
    saved.export.imagej_rois = False
    app = _app()
    next(u for u in app.sidebar.file_uploader if u.label == "Reload saved settings").set_value(
        ("hive_run.json", json.dumps({"config": saved.to_dict()}).encode(), "application/json"))
    app.run()
    assert not app.exception
    assert _sidebar(app, "number_input", "Min object size (debris filter)").value == 123
    assert _sidebar(app, "number_input", "Random seed").value == 42
    assert _sidebar(app, "number_input", "Pixel size (XY)").value == 0.65
    assert _sidebar(app, "checkbox", "Split touching objects (watershed)").value is True
    assert _sidebar(app, "checkbox", "Export objects as ImageJ ROIs").value is False


def test_an_unusable_settings_file_is_explained_and_the_defaults_stay():
    app = _app()
    next(u for u in app.sidebar.file_uploader if u.label == "Reload saved settings").set_value(
        ("hive_run.json", b'{"nonsense": 1}', "application/json"))
    app.run()
    assert not app.exception and any("nonsense" in e.value for e in app.sidebar.error)
    assert _sidebar(app, "number_input", "Min object size (debris filter)").value == 80


def test_an_ome_tiff_offers_its_channel_names_and_fills_in_the_pixel_size():
    app = _app()
    app.sidebar.radio[1].set_value("Upload files").run()
    app.file_uploader[0].set_value(("scan.ome.tif", _ome_bytes(), "image/tiff")).run()
    assert not app.exception
    channel = _sidebar(app, "selectbox", "Channel")
    assert list(channel.options) == ["auto", "DAPI", "GFP"]
    assert _sidebar(app, "number_input", "Pixel size (XY)").value == 0.4
    assert _sidebar(app, "text_input", "Unit label").value == "um"
    assert set(_sidebar(app, "multiselect", "Also measure (colocalization)").options) == {"DAPI", "GFP"}


def test_a_two_channel_file_runs_with_colocalization_in_the_app():
    app = _app()
    app.sidebar.radio[1].set_value("Upload files").run()
    app.file_uploader[0].set_value(("scan.ome.tif", _ome_bytes(), "image/tiff")).run()
    _sidebar(app, "selectbox", "Channel").set_value("DAPI").run()
    # the main channel is left out of the extras, so fetch the list again after choosing it
    _sidebar(app, "multiselect", "Also measure (colocalization)").set_value(["GFP"]).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "1"
    columns = [df.value.columns for df in app.dataframe]
    assert any("coloc_GFP_pearson" in c for c in columns)


def test_the_background_preview_draws_when_asked():
    app = _app()
    next(c for c in app.checkbox if c.label == "Show the effect of each background method").check()
    app.run()
    assert not app.exception
    assert any("central" in c.value for c in app.caption)


def test_exports_and_settings_are_offered_after_a_run():
    app = _app()
    app.button[0].click().run()
    assert not app.exception
    labels = [b.proto.label for b in app.get("download_button")]
    assert "Settings (hive_run.json)" in labels
