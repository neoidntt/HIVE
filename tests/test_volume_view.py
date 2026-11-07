"""The interactive 3D view is built from every z-plane, true to scale, and reaches the app."""

from __future__ import annotations

import base64
import json
import re
import zlib

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

from hive import volume_view as vv
from hive.pipeline import analyze_zstack

from helpers import blob_volume, quick_config

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _decode(page: str):
    meta = json.loads(re.search(r"const META = (\{.*?\});", page).group(1))
    payload = re.search(r'const B64 = "([^"]+)"', page).group(1)
    raw = np.frombuffer(zlib.decompress(base64.b64decode(payload)), dtype=np.uint8)
    return meta, raw.reshape((meta["d"], meta["h"], meta["w"], 4))       # R, G, B: channels; A: objects


def test_every_z_plane_is_in_the_page_and_planes_are_not_flattened():
    volume = blob_volume()                              # (10, 200, 220), spheres at different depths
    meta, data = _decode(vv.volume_viewer_html(volume))
    red = data[..., 0]
    assert meta["d"] == volume.shape[0] and red.shape == volume.shape       # nothing dropped
    assert len({plane.tobytes() for plane in red}) == volume.shape[0]       # each plane is its own
    assert not np.array_equal(red.max(axis=0), red[volume.shape[0] // 2])   # more than one slice's worth


def test_a_big_volume_is_reduced_in_x_and_y_only():
    volume = np.random.default_rng(0).random((12, 64, 80), dtype=np.float32)
    small, k = vv.reduce_volume(volume, max_voxels=12 * 64 * 80 // 4)
    assert k == 2 and small.shape == (12, 32, 40)                          # all 12 planes kept
    np.testing.assert_allclose(small[3, 5, 7], volume[3, 10:12, 14:16].mean(), rtol=1e-6)
    assert vv.reduce_volume(volume, max_voxels=10**9)[1] == 1


def test_a_thin_sample_is_stretched_in_depth_and_a_deep_one_is_not():
    assert vv.auto_depth_exaggeration((109, 2217, 4953), (1, 1, 1)) == pytest.approx(9.09, rel=0.01)
    assert vv.auto_depth_exaggeration((100, 100, 100), (1, 1, 1)) == 1.0
    assert vv.auto_depth_exaggeration((2, 5000, 5000), (1, 1, 1)) == 12.0          # capped
    meta, _ = _decode(vv.volume_viewer_html(blob_volume(), voxel_size=(1.0, 1.0, 1.0),
                                            depth_exaggeration=1.0))
    assert meta["stretch"] == 1.0 and meta["scale"] == [1.0, 1.0, 1.0]


def test_the_mask_rides_along_and_odd_input_is_refused():
    volume = blob_volume()
    meta, data = _decode(vv.volume_viewer_html(volume, volume > 100))
    assert meta["hasMask"] and meta["channels"] == 1
    assert data[..., 3].any() and set(np.unique(data[..., 3])) <= {0, 255}
    assert not _decode(vv.volume_viewer_html(volume))[1][..., 3].any()      # no mask: nothing highlighted
    with pytest.raises(ValueError, match="at least 2 z-planes"):
        vv.volume_viewer_html(volume[0])
    with pytest.raises(ValueError, match="mask shape"):
        vv.volume_viewer_html(volume, np.zeros((3, 3, 3), bool))


def test_a_file_name_cannot_break_out_of_the_page():
    page = vv.volume_viewer_html(blob_volume(), title='x</script><script>alert(1)</script>')
    assert "</script><script>alert(1)" not in page
    assert page.count("</script>") == 1


def test_the_analysis_writes_the_view_from_the_whole_stack(tmp_path):
    volume = blob_volume()
    import tifffile

    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, volume)
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(tmp_path / "out"), mode="3d")
    cfg.large_image.mode = "off"
    result = analyze_zstack(cfg)
    page = result.outputs["volume_viewer"].read_text(encoding="utf-8")
    meta, data = _decode(page)
    assert data.shape[:3] == volume.shape and meta["hasMask"] and meta["unit"] == cfg.voxel_unit


def test_the_app_shows_the_3d_view_after_a_3d_run():
    app = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    next(r for r in app.sidebar.radio if r.label == "Mode").set_value("3D z-stack").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert any("3D volume (rotate, zoom, pan)" in s.value for s in app.subheader)


def test_further_channels_get_their_own_colour_channel_and_name():
    a, b = blob_volume(seed=1), blob_volume(seed=2)
    meta, data = _decode(vv.volume_viewer_html(a, extra_channels=[b], channel_names=["DAPI", "Neuron"], unit="um"))
    assert meta["channels"] == 2 and meta["names"] == ["DAPI", "Neuron"] and meta["unit"] == "um"
    assert data[..., 0].any() and data[..., 1].any() and not np.array_equal(data[..., 0], data[..., 1])
    assert not data[..., 2].any()                                           # the third slot is unused
    assert _decode(vv.volume_viewer_html(a, extra_channels=[b, b, b]))[0]["channels"] == 3   # at most 3 in all
    with pytest.raises(ValueError, match="channel shape"):
        vv.volume_viewer_html(a, extra_channels=[np.zeros((3, 3, 3))])


def test_the_page_offers_the_professional_controls():
    page = vv.volume_viewer_html(blob_volume(), blob_volume() > 100)
    for control in ["Volume (see-through, depth)", "Surface (first structure hit)", "Brightest voxel",
                    "Shadows (full quality only)", "Ambient occlusion", "Light direction (left-right)",
                    "Specular (shine)", "Light follows the camera", "Black point", "White point", "Gamma",
                    "Opacity", "Viridis", "Crop box", "Cutting plane 1", "Cutting plane 2", "Any direction",
                    "Orthographic", "Background", "Orientation axes", "Scale bar", "Bounding box",
                    "Faster render while moving", "Quality (samples per ray)", "Save screenshot (PNG)",
                    "Rotate", "Screenshot", "Depth exaggeration"]:
        assert control in page, control
    assert "https://" not in page.replace("http://www.w3.org", "") and "src=" not in page    # nothing fetched
