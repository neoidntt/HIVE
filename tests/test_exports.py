"""Exports for Fiji and napari must reproduce the labels, not just exist."""

from __future__ import annotations

import sys
import warnings
import zipfile

import numpy as np
import pytest
import tifffile
from skimage.draw import polygon

from hive.exports import export_imagej_rois, export_label_files, export_labels_ome_tiff
from hive.pipeline import analyze_folder, analyze_zstack

from helpers import blob_volume, quick_config, write_synthetic_folder

roifile = pytest.importorskip("roifile")


def _rasterise(roi, shape):
    xy = roi.coordinates()                        # ImageJ: pixel (r, c) covers [c, c+1] x [r, r+1]
    mask = np.zeros(shape, bool)
    rr, cc = polygon(xy[:, 1] - 0.5, xy[:, 0] - 0.5, shape)
    mask[rr, cc] = True
    return mask


def _labels_2d():
    labels = np.zeros((60, 80), np.int32)
    labels[5:25, 8:30] = 1
    yy, xx = np.ogrid[:60, :80]
    labels[(yy - 40) ** 2 + (xx - 55) ** 2 < 100] = 2
    return labels


def test_roi_outlines_rasterise_back_to_the_original_objects(tmp_path):
    labels = _labels_2d()
    count = export_imagej_rois(tmp_path / "r.zip", labels)
    rois = roifile.roiread(tmp_path / "r.zip")
    assert count == len(rois) == 2
    for roi in rois:
        label = int(roi.name[:4])
        target = labels == label
        mask = _rasterise(roi, labels.shape)
        assert (mask & target).sum() / (mask | target).sum() > 0.95


def test_3d_rois_carry_their_plane_and_skip_empty_planes(tmp_path):
    labels = np.zeros((5, 30, 30), np.int32)
    labels[1:4, 5:15, 5:15] = 1
    export_imagej_rois(tmp_path / "r.zip", labels)
    rois = roifile.roiread(tmp_path / "r.zip")
    assert sorted(r.position for r in rois) == [2, 3, 4]        # ImageJ numbers planes from 1
    assert all(r.name.startswith("0001-z") for r in rois)


def test_an_object_in_two_pieces_gets_two_rois_and_a_hole_does_not_split_it(tmp_path):
    labels = np.zeros((30, 50), np.int32)
    labels[5:12, 5:12] = 1
    labels[5:12, 30:40] = 1                         # same label, separate piece
    labels[20:28, 5:20] = 2
    labels[22:25, 8:12] = 0                         # a hole inside object 2
    assert export_imagej_rois(tmp_path / "r.zip", labels) == 3


def test_no_objects_writes_nothing(tmp_path):
    assert export_imagej_rois(tmp_path / "r.zip", np.zeros((10, 10), int)) == 0
    assert not (tmp_path / "r.zip").exists()


def test_roi_input_must_be_2d_or_3d(tmp_path):
    with pytest.raises(ValueError):
        export_imagej_rois(tmp_path / "r.zip", np.zeros((2, 2, 2, 2), int))


def test_ome_tiff_labels_keep_values_and_physical_size(tmp_path):
    labels = np.zeros((4, 20, 30), np.int32)
    labels[1:3, 5:10, 5:10] = 7
    path = export_labels_ome_tiff(tmp_path / "l.ome.tif", labels, 0.4, 2.0, "um")
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome and tif.series[0].axes == "ZYX"
        assert np.array_equal(tif.asarray(), labels)
        xml = tif.ome_metadata
    assert 'PhysicalSizeX="0.4"' in xml and 'PhysicalSizeZ="2.0"' in xml and "µm" in xml


def test_pixel_units_do_not_pretend_to_be_micrometres(tmp_path):
    path = export_labels_ome_tiff(tmp_path / "p.ome.tif", _labels_2d(), 1.0, 1.0, "px")
    with tifffile.TiffFile(path) as tif:
        assert "PhysicalSizeX" not in tif.ome_metadata


def test_many_objects_switch_to_32_bit_labels(tmp_path):
    labels = np.zeros((4, 4), np.int64)
    labels[0, 0] = 70_000
    path = export_labels_ome_tiff(tmp_path / "big.ome.tif", labels)
    assert tifffile.imread(path).dtype == np.uint32 and tifffile.imread(path)[0, 0] == 70_000
    with pytest.raises(ValueError, match="negative"):
        export_labels_ome_tiff(tmp_path / "bad.ome.tif", -labels)


def test_2d_batch_writes_labels_and_rois_for_each_image(tmp_path):
    write_synthetic_folder(tmp_path / "in")
    analyze_folder(quick_config(tmp_path))
    out = tmp_path / "out"
    for stem in ("synthetic_dim", "synthetic_reference", "synthetic_bright"):
        labels = tifffile.imread(out / f"{stem}_labels.ome.tif")
        assert labels.max() == 6                       # six objects in every synthetic field
        assert len(roifile.roiread(out / f"{stem}_rois.zip")) == 6


def test_3d_run_writes_labels_and_rois(tmp_path):
    path = tmp_path / "v.tif"
    tifffile.imwrite(path, blob_volume((8, 120, 120))[:, :, :120])
    cfg = quick_config(tmp_path, input_path=str(path), mode="3d")
    cfg.preprocess.tophat_radius = (2.0, 12.0, 12.0)
    result = analyze_zstack(cfg)
    assert {"labels_ome_tiff", "imagej_rois"} <= set(result.outputs)
    assert tifffile.imread(result.outputs["labels_ome_tiff"]).shape == (8, 120, 120)
    assert len(roifile.roiread(result.outputs["imagej_rois"])) >= result.object_count


def test_exports_can_be_switched_off(tmp_path):
    write_synthetic_folder(tmp_path / "in")
    cfg = quick_config(tmp_path)
    cfg.export.ome_tiff_labels = cfg.export.imagej_rois = False
    analyze_folder(cfg)
    assert not list((tmp_path / "out").glob("*_labels.ome.tif"))
    assert not list((tmp_path / "out").glob("*_rois.zip"))


def test_a_missing_roifile_package_warns_instead_of_failing_the_run(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "roifile", None)
    cfg = quick_config(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = export_label_files(tmp_path, "x", _labels_2d(), cfg)
    assert "labels_ome_tiff" in written and "imagej_rois" not in written
    assert any("roifile" in str(w.message) for w in caught)
