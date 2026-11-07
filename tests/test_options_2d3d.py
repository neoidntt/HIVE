"""2D-only options are enabled only for 2D, 3D-only options only for 3D, shared options always.
Unavailable options are greyed out, never removed, and flip the moment the mode is switched."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from streamlit.testing.v1 import AppTest

from helpers import blob_volume

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

THREE_D_ONLY = ["Z step", "Projection", "Frames", "FPS", "Colormap", "Downsample", "Rotation axis",
                "Also render rotating 3D GIF"]
TWO_D_ONLY = ["Parallel workers (2D batches)"]
BOTH = ["Mode", "Data source", "Channel", "Also measure (colocalization)", "Pixel size (XY)", "Unit label",
        "Background estimator", "Gaussian sigma (segmentation only)", "CLAHE contrast enhancement",
        "Method", "Otsu threshold scale", "Min object size (debris filter)", "Max object size (0 = no limit)",
        "Fill holes up to", "Drop edge-touching objects", "Split touching objects (watershed)",
        "Analyse branching / Sholl", "Random seed", "Export labels as OME-TIFF",
        "Export objects as ImageJ ROIs", "Images and stacks larger than memory"]


def _widgets(app, label):
    found = []
    for group in (app.sidebar, app.main):
        for kind in ("number_input", "selectbox", "slider", "checkbox", "radio", "multiselect", "text_input"):
            found += [w for w in getattr(group, kind) if w.label == label]
    return found


def _state(app, label):
    widgets = _widgets(app, label)
    assert len(widgets) == 1, f"{label!r}: expected one widget, found {len(widgets)}"
    return widgets[0].disabled


def _switch(app, mode):
    next(r for r in app.sidebar.radio if r.label == "Mode").set_value(mode).run()


def test_three_d_only_options_are_greyed_out_in_2d_and_enabled_in_3d():
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    _switch(app, "2D images")
    assert all(_state(app, label) for label in THREE_D_ONLY), [l for l in THREE_D_ONLY if not _state(app, l)]
    assert not any(_state(app, label) for label in TWO_D_ONLY)
    _switch(app, "3D z-stack")                                    # right away, in the same run
    assert not any(_state(app, label) for label in THREE_D_ONLY), [l for l in THREE_D_ONLY if _state(app, l)]
    assert all(_state(app, label) for label in TWO_D_ONLY)
    _switch(app, "2D images")                                     # and back again
    assert all(_state(app, label) for label in THREE_D_ONLY) and not any(_state(app, l) for l in TWO_D_ONLY)


def test_options_for_both_stay_available_in_both_modes():
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    for mode in ("2D images", "3D z-stack", "2D images"):
        _switch(app, mode)
        assert [label for label in BOTH if _state(app, label)] == []


def test_a_stack_option_for_folders_is_present_and_greyed_out_when_it_does_not_apply(tmp_path):
    folder = tmp_path / "stacks"
    folder.mkdir()
    tifffile.imwrite(folder / "a.tif", blob_volume())
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    next(r for r in app.sidebar.radio if r.label == "Data source").set_value("File or folder on this computer").run()
    app.sidebar.text_input(key="local_path").set_value(str(folder)).run()
    label = "Each file in this folder is its own stack (analysed one after another)"
    _switch(app, "2D images")
    assert _state(app, label)                                     # there, but greyed out
    _switch(app, "3D z-stack")
    assert not _state(app, label)                                 # a folder in 3D: available
    assert next(w for w in app.checkbox if w.label == label).value is True    # and ticked for whole stacks
    app.sidebar.text_input(key="local_path").set_value(str(folder / "a.tif")).run()
    assert _state(app, label)                                     # a single file: greyed out again


def test_switching_to_2d_does_not_use_the_3d_settings(tmp_path):
    """A greyed-out value must not leak into the analysis (the z step stays 1 in 2D)."""
    data = tmp_path / "data"
    data.mkdir()
    tifffile.imwrite(data / "x.tif", (np.random.default_rng(0).random((64, 64)) * 100).astype(np.float32))
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    _switch(app, "3D z-stack")
    next(w for w in app.sidebar.number_input if w.label == "Z step").set_value(7.5).run()
    _switch(app, "2D images")
    assert _state(app, "Z step")
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["cfg"].voxel_size_z == 1.0 and app.session_state["cfg"].mode == "2d"
