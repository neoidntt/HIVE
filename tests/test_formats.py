"""Microscope formats: channel names and voxel size must come out of the files."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import tifffile

from hive.config import HiveConfig
from hive.formats import apply_metadata, is_microscope_file, read_microscope
from hive.io_utils import ImageLoadError
from hive.pipeline import analyze_image, analyze_zstack

from helpers import quick_config, write_ome

RNG = np.random.default_rng(0)


def test_ome_tiff_gives_channels_pixels_and_voxel_size(tmp_path):
    data = RNG.integers(0, 1000, (3, 4, 32, 40)).astype(np.uint16)
    path = write_ome(tmp_path / "x.ome.tif", data, "CZYX", ["DAPI", "GFP", "RFP"], xy=0.25, z=1.5)
    meta = read_microscope(path)
    assert meta.shape == (3, 4, 32, 40)
    assert meta.channel_names == ["DAPI", "GFP", "RFP"]
    assert meta.voxel_size_xy == pytest.approx(0.25) and meta.voxel_size_z == pytest.approx(1.5)
    assert np.array_equal(meta.data, data)              # nothing rescaled or reordered wrongly
    assert is_microscope_file(path)


def test_axis_order_in_the_file_does_not_matter(tmp_path):
    data = RNG.integers(0, 1000, (4, 2, 16, 20)).astype(np.uint16)          # Z, C, Y, X
    path = write_ome(tmp_path / "zc.ome.tif", data, "ZCYX", ["A", "B"])
    meta = read_microscope(path)
    assert meta.shape == (2, 4, 16, 20)
    assert np.array_equal(meta.volume("B"), data[:, 1].astype(np.float32))


def test_metadata_can_be_read_without_pixels(tmp_path):
    path = write_ome(tmp_path / "m.ome.tif", np.zeros((2, 3, 8, 8), np.uint16), "CZYX", ["a", "b"])
    meta = read_microscope(path, load_pixels=False)
    assert meta.data is None and meta.shape == (2, 3, 8, 8)
    with pytest.raises(ImageLoadError):
        meta.volume("a")


def test_missing_channel_names_become_numbers_not_blanks(tmp_path):
    path = write_ome(tmp_path / "n.ome.tif", np.zeros((2, 8, 8), np.uint16), "CYX")
    assert read_microscope(path).channel_names == ["channel_0", "channel_1"]


def test_imagej_hyperstack_scale_is_read(tmp_path):
    path = tmp_path / "ij.tif"
    tifffile.imwrite(path, np.zeros((5, 20, 20), np.float32), imagej=True,
                     resolution=(2.0, 2.0), metadata={"spacing": 2.0, "unit": "um", "axes": "ZYX"})
    meta = read_microscope(path)
    assert meta.shape == (1, 5, 20, 20)
    assert meta.voxel_size_xy == pytest.approx(0.5) and meta.voxel_size_z == pytest.approx(2.0)
    assert not is_microscope_file(path)            # single-channel: the ordinary path is fine


def test_a_time_series_is_refused_with_advice(tmp_path):
    path = write_ome(tmp_path / "t.ome.tif", np.zeros((3, 8, 8), np.uint16), "TYX")
    with pytest.raises(ImageLoadError, match="3 time points"):
        read_microscope(path)


def test_channel_lookup(tmp_path):
    path = write_ome(tmp_path / "c.ome.tif", np.zeros((2, 8, 8), np.uint16), "CYX", ["DAPI", "GFP"])
    meta = read_microscope(path, load_pixels=False)
    assert meta.channel_index("gfp") == 1 and meta.channel_index("0") == 0
    assert meta.channel_index("channel_1") == 1
    with pytest.raises(ValueError, match="DAPI"):
        meta.channel_index("magenta")


def test_plain_images_use_the_ordinary_path(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((8, 8), np.uint16))
    assert not is_microscope_file(path)
    assert not is_microscope_file(tmp_path / "missing.tif")


def test_file_scale_fills_in_defaults_but_never_overrides_the_user(tmp_path):
    path = write_ome(tmp_path / "s.ome.tif", np.zeros((2, 8, 8), np.uint16), "CYX", xy=0.3)
    meta = read_microscope(path, load_pixels=False)
    fresh = HiveConfig()
    assert apply_metadata(fresh, meta) and fresh.voxel_size_xy == pytest.approx(0.3)
    assert fresh.voxel_unit == "um"
    chosen = HiveConfig(voxel_size_xy=0.65, voxel_unit="um")
    assert apply_metadata(chosen, meta) == [] and chosen.voxel_size_xy == 0.65


def test_analysis_uses_the_named_channel_and_the_files_pixel_size(tmp_path):
    ch0, ch1 = np.full((64, 64), 50, np.float32), np.full((64, 64), 50, np.float32)
    ch0[10:30, 10:30] += 400                            # objects only in DAPI
    ch1[40:60, 40:60] += 900                            # a different object in GFP
    path = write_ome(tmp_path / "two.ome.tif", np.stack([ch0, ch1]), "CYX", ["DAPI", "GFP"], xy=0.5)
    cfg = quick_config(tmp_path, channel="DAPI")
    result = analyze_image(path, cfg)
    assert result.prepared.channel_name == "DAPI"
    assert result.object_count == 1
    assert result.measurements["area_px"].iloc[0] == pytest.approx(400, abs=10)
    assert result.measurements["area_um2"].iloc[0] == pytest.approx(
        result.measurements["area_px"].iloc[0] * 0.25)              # 0.5 um pixels from the file
    assert cfg.voxel_size_xy == 1.0                                  # the caller's config is untouched
    other = analyze_image(path, quick_config(tmp_path, channel="GFP"))
    assert other.measurements["centroid-0"].iloc[0] > 40             # found the GFP object instead


def test_auto_channel_picks_the_most_variable_and_records_which(tmp_path):
    flat, busy = np.full((64, 64), 5, np.float32), np.full((64, 64), 5, np.float32)
    busy[20:40, 20:40] += 800
    path = write_ome(tmp_path / "a.ome.tif", np.stack([flat, busy]), "CYX", ["quiet", "busy"])
    result = analyze_image(path, quick_config(tmp_path, channel="auto"))
    assert result.prepared.channel_name == "busy"


def test_a_multichannel_ome_stack_runs_in_3d(tmp_path):
    vol = np.full((6, 40, 40), 10, np.float32)
    vol[2:5, 10:25, 10:25] += 300
    path = write_ome(tmp_path / "s.ome.tif", np.stack([vol, vol * 0.5]), "CZYX", ["a", "b"],
                     xy=0.2, z=1.0)
    cfg = quick_config(tmp_path, input_path=str(path), mode="3d", channel="a")
    result = analyze_zstack(cfg)
    assert result.object_count == 1 and "volume_um3" in result.measurements.columns


# --- vendor formats: stand-ins for the libraries, since no real files are available -------


def _install_fake(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_czi_axes_scale_and_channel_names(tmp_path, monkeypatch):
    xml = """<ImageDocument><Metadata><Scaling><Items>
      <Distance Id="X"><Value>2.5E-07</Value></Distance>
      <Distance Id="Z"><Value>1E-06</Value></Distance></Items></Scaling>
      <DisplaySetting><Channels><Channel Name="Alexa488"/><Channel Name="DAPI"/></Channels>
      </DisplaySetting></Metadata></ImageDocument>"""
    pixels = RNG.integers(0, 100, (1, 1, 2, 3, 6, 8, 1)).astype(np.uint16)   # B T C Z Y X 0

    class FakeCzi:
        axes, shape = "BTCZYX0", pixels.shape

        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def asarray(self): return pixels
        def metadata(self, raw=True): return xml

    _install_fake(monkeypatch, "czifile", CziFile=FakeCzi)
    path = tmp_path / "x.czi"
    path.write_bytes(b"")
    meta = read_microscope(path)
    assert meta.shape == (2, 3, 6, 8)
    assert meta.channel_names == ["Alexa488", "DAPI"]
    assert meta.voxel_size_xy == pytest.approx(0.25) and meta.voxel_size_z == pytest.approx(1.0)
    assert np.array_equal(meta.data[1, 2], pixels[0, 0, 1, 2, :, :, 0])


def test_nd2_axes_scale_and_channel_names(tmp_path, monkeypatch):
    pixels = RNG.integers(0, 100, (4, 2, 6, 8)).astype(np.uint16)             # Z C Y X

    class FakeNd2:
        sizes = {"Z": 4, "C": 2, "Y": 6, "X": 8}
        metadata = types.SimpleNamespace(channels=[
            types.SimpleNamespace(channel=types.SimpleNamespace(name="TRITC")),
            types.SimpleNamespace(channel=types.SimpleNamespace(name="FITC"))])

        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def voxel_size(self): return types.SimpleNamespace(x=0.325, y=0.325, z=2.0)
        def asarray(self): return pixels

    _install_fake(monkeypatch, "nd2", ND2File=FakeNd2)
    path = tmp_path / "x.nd2"
    path.write_bytes(b"")
    meta = read_microscope(path)
    assert meta.shape == (2, 4, 6, 8) and meta.channel_names == ["TRITC", "FITC"]
    assert meta.voxel_size_xy == pytest.approx(0.325) and meta.voxel_size_z == pytest.approx(2.0)
    assert np.array_equal(meta.volume("FITC"), pixels[:, 1].astype(np.float32))


def test_lif_frames_scale_and_lut_names(tmp_path, monkeypatch):
    frames = {(c, z): RNG.integers(0, 100, (5, 7)).astype(np.uint16) for c in range(2) for z in range(3)}

    class FakeImage:
        name, channels, nz, nt, n_mosaic = "Series001", 2, 3, 1, 1
        dims = types.SimpleNamespace(x=7, y=5, z=3, t=1, m=1)
        scale_n = {1: 4.0, 2: 4.0, 3: 1.0}

        def get_frame(self, z=0, t=0, c=0, m=0): return frames[(c, z)]

    class FakeLif:
        def __init__(self, path):
            import xml.etree.ElementTree as ET
            self.xml_root = ET.fromstring(
                '<LMSDataContainerHeader><Element Name="Series001"><Data><Image><ImageDescription>'
                '<Channels><ChannelDescription LUTName="Green"/><ChannelDescription LUTName="Red"/>'
                '</Channels></ImageDescription></Image></Data></Element></LMSDataContainerHeader>')

        def get_iter_image(self): return iter([FakeImage()])

    _install_fake(monkeypatch, "readlif")
    _install_fake(monkeypatch, "readlif.reader", LifFile=FakeLif)
    path = tmp_path / "x.lif"
    path.write_bytes(b"")
    meta = read_microscope(path)
    assert meta.shape == (2, 3, 5, 7) and meta.channel_names == ["Green", "Red"]
    assert meta.voxel_size_xy == pytest.approx(0.25) and meta.voxel_size_z == pytest.approx(1.0)
    assert np.array_equal(meta.data[1, 2], frames[(1, 2)])


def test_a_missing_vendor_library_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "nd2", None)           # makes `import nd2` fail
    path = tmp_path / "x.nd2"
    path.write_bytes(b"")
    with pytest.raises(ImageLoadError, match="pip install nd2"):
        read_microscope(path)
