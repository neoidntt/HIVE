from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import tifffile
from streamlit.testing.v1 import AppTest

from hive.synthetic import make_synthetic_2d, make_synthetic_3d

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_synthetic_phantoms_are_reproducible():
    first_2d = make_synthetic_2d(seed=12)
    second_2d = make_synthetic_2d(seed=12)
    first_3d = make_synthetic_3d(seed=12)
    second_3d = make_synthetic_3d(seed=12)

    assert [name for name, _ in first_2d] == [name for name, _ in second_2d]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(first_2d, second_2d))
    assert first_3d.shape == (20, 90, 90)
    assert np.array_equal(first_3d, second_3d)


def test_default_synthetic_2d_demo_runs():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert any(heading.value == "Built-in synthetic sample" for heading in app.subheader)

    app.button[0].click().run()

    assert not app.exception
    assert app.metric[0].value == "18"
    assert any(heading.value == "Interactive measurement plots" for heading in app.subheader)


def test_synthetic_3d_demo_runs():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.radio[0].set_value("3D z-stack").run()
    app.button[0].click().run()

    assert not app.exception
    assert int(app.metric[0].value) > 0
    assert any(heading.value == "Interactive measurement plots" for heading in app.subheader)
    assert len(app.get("plotly_chart")) >= 3


def test_uploaded_tiff_runs_through_2d_demo_app():
    buffer = io.BytesIO()
    image = np.full((96, 96), 100, dtype=np.uint16)
    image[30:60, 30:60] = 500
    tifffile.imwrite(buffer, image)

    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.radio[1].set_value("Upload files").run()
    app.file_uploader[0].set_value(("external.tif", buffer.getvalue(), "image/tiff")).run()
    app.button[0].click().run()

    assert not app.exception
    assert app.metric[0].value == "1"
    assert any(heading.value == "2D image inspection" for heading in app.subheader)