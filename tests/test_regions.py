"""Real 3D surfaces of every segmented region, and the page that shows them."""

from __future__ import annotations

import base64
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from streamlit.testing.v1 import AppTest

from hive import regions3d as R
from hive.config import LargeImageConfig
from hive.pipeline import analyze_zstack
from hive.regions_view import regions_page_for, regions_viewer_html

from helpers import blob_volume, quick_config

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _balls(shape=(40, 60, 70)):
    labels = np.zeros(shape, np.uint32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    labels[((zz - 20) ** 2 + (yy - 25) ** 2 + (xx - 20) ** 2) <= 8 ** 2] = 1
    labels[((zz - 20) ** 2 + (yy - 30) ** 2 + (xx - 50) ** 2) <= 5 ** 2] = 2
    labels[5, 5, 5] = 3                                          # one voxel
    return labels


def _volume_of(meshes, index):
    first = sum(t["triangles"] for t in meshes.table[:index])
    faces = meshes.faces[first:first + meshes.table[index]["triangles"]]
    p = meshes.positions.astype(np.float64)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def test_each_region_gets_its_own_closed_surface_of_the_right_size():
    labels = _balls()
    meshes = R.build_region_meshes(labels, R.boxes_from_volume(labels))
    assert [t["label"] for t in meshes.table] == [1, 2, 3]                  # largest first
    assert meshes.skipped == [] and len(meshes.table) == 3
    for index in (0, 1):                                                    # the mesh encloses about the voxels
        assert _volume_of(meshes, index) == pytest.approx(meshes.table[index]["voxels"], rel=0.12)
    assert meshes.table[2]["triangles"] > 0                                 # even a single voxel gets a surface
    centre = meshes.positions[meshes.region == 0].mean(axis=0)
    outward = (meshes.normals[meshes.region == 0] * (meshes.positions[meshes.region == 0] - centre)).sum(axis=1)
    assert (outward > 0).all()                                              # normals point out of the region
    assert meshes.faces.max() < len(meshes.positions)


def test_surfaces_are_in_physical_units_so_a_thick_z_step_is_not_squashed():
    labels = _balls()
    boxes = R.boxes_from_volume(labels)
    flat = R.build_region_meshes(labels, boxes, spacing=(1.0, 1.0, 1.0))
    tall = R.build_region_meshes(labels, boxes, spacing=(3.0, 1.0, 1.0))
    extent = lambda m: np.ptp(m.positions[m.region == 0], axis=0)
    assert extent(tall)[2] == pytest.approx(3 * extent(flat)[2], rel=0.02)
    assert extent(tall)[0] == pytest.approx(extent(flat)[0], rel=0.08)      # (the light blur depends on spacing)


def test_a_thin_wide_sheet_keeps_its_surface_when_the_grid_is_thinned():
    """Regions a few planes deep but hundreds of pixels wide (real stacks have them) must not vanish."""
    labels = np.zeros((6, 160, 300), np.uint32)
    labels[1:4, 10:150, 20:290] = 1                                  # 3 planes thick, 140 x 270 across
    labels[2:4, 5:20, 5:15] = 2                                       # a small one too
    meshes = R.build_region_meshes(labels, R.boxes_from_volume(labels))
    assert meshes.skipped == [] and [t["label"] for t in meshes.table] == [1, 2]
    z = meshes.positions[meshes.region == 0][:, 2]
    assert 2.0 < np.ptp(z) < 4.0                                      # about the sheet's own 3 planes of depth


def test_only_the_voxels_around_each_region_are_read_from_a_memory_map(tmp_path):
    labels = _balls()
    path = tmp_path / "labels.tif"
    tifffile.imwrite(path, labels)
    mapped = tifffile.memmap(path)
    table = pd.DataFrame({"label": [1, 2, 3], **{f"bbox-{i}": [b[i] for b in
                          (R.boxes_from_volume(labels)[k] for k in (1, 2, 3))] for i in range(6)}})
    from_table = R.build_region_meshes(mapped, R.region_boxes(table))
    from_memory = R.build_region_meshes(labels, R.boxes_from_volume(labels))
    np.testing.assert_array_equal(from_table.faces, from_memory.faces)
    with pytest.raises(ValueError, match="bbox"):
        R.region_boxes(pd.DataFrame({"label": [1]}))


def test_a_triangle_budget_and_a_region_cap_are_respected():
    labels = np.zeros((50, 100, 100), np.uint32)
    zz, yy, xx = np.ogrid[:50, :100, :100]
    for i, (cy, cx) in enumerate([(25, 25), (25, 75), (75, 25), (75, 75)], start=1):
        labels[((zz - 25) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= (19 - i) ** 2] = i
    boxes = R.boxes_from_volume(labels)
    fine = R.build_region_meshes(labels, boxes)
    coarse = R.build_region_meshes(labels, boxes, max_triangles=len(fine.faces) // 3)
    assert len(coarse.faces) < len(fine.faces) and coarse.grid_side < fine.grid_side
    capped = R.build_region_meshes(labels, boxes, max_regions=2)
    assert len(capped.table) == 2 and capped.omitted == 2 and capped.table[0]["label"] == 1


def _decode(page: str):
    meta = json.loads(re.search(r"const META = (\{.*?\});", page).group(1))
    blob = zlib.decompress(base64.b64decode(re.search(r'const B64 = "([^"]+)"', page).group(1)))
    n, m = meta["n"], meta["m"]
    region = np.frombuffer(blob, "<u4", n)
    faces = np.frombuffer(blob, "<u4", 3 * m, 4 * n).reshape(m, 3)
    return meta, region, faces


def test_the_page_carries_every_region_and_the_controls_for_them():
    labels = _balls()
    meshes = R.build_region_meshes(labels, R.boxes_from_volume(labels))
    page = regions_viewer_html(meshes, "x</script><b>", "um", volume_extent=(70, 60, 40))
    meta, region, faces = _decode(page)
    assert [r[0] for r in meta["regions"]] == [1, 2, 3] and meta["unit"] == "um" and meta["extent"] == [70, 60, 40]
    assert set(np.unique(region)) == {0, 1, 2} and faces.max() < meta["n"]
    assert page.count("</script>") == 1                                   # a title cannot break out of the page
    for control in ["Transparency of all regions", "Show only the largest", "Show all", "Hide all", "Invert",
                    "A different colour for each", "By volume (viridis)", "Find a region by its number",
                    "Click a region", "Cutting plane 1", "Crop box", "Light follows the camera", "Specular (shine)",
                    "Bounding box of the sample", "Scale bar", "Orientation axes", "Save screenshot (PNG)",
                    "order-independent", "Orthographic"]:
        assert control in page, control
    assert "https://" not in page and "src=" not in page                  # nothing is fetched
    with pytest.raises(ValueError, match="no region surfaces"):
        regions_viewer_html(R.build_region_meshes(np.zeros((4, 4, 4), np.uint32), {}))


def test_the_3d_analysis_builds_the_surfaces_from_the_whole_stack(tmp_path):
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, blob_volume())
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(tmp_path / "out"), mode="3d")
    cfg.large_image = LargeImageConfig(mode="off")
    result = analyze_zstack(cfg)
    meta, region, faces = _decode(result.pages["regions"])
    assert len(meta["regions"]) == result.object_count == 9
    assert (tmp_path / "out" / "HIVE_3D_regions_viewer.html").read_text(encoding="utf-8") == result.pages["regions"]
    z_extent = np.ptp(R.build_region_meshes(np.asarray(result.labels), R.region_boxes(result.measurements)).positions[:, 2])
    assert z_extent > 2          # the regions have depth in z: they are not flat
    unsaved = analyze_zstack(cfg, save=False)
    assert "regions" in unsaved.pages and not (tmp_path / "out2").exists()


def test_tiled_surfaces_come_from_the_label_file(tmp_path):
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, blob_volume())
    cfg = quick_config(tmp_path, input_path=str(path), output_folder=str(tmp_path / "out"), mode="3d")
    cfg.large_image = LargeImageConfig(mode="force", tile_size=96)
    result = analyze_zstack(cfg)
    page = regions_page_for(tifffile.memmap(str(result.outputs["labels_ome_tiff"])), result.measurements,
                            (1.0, 1.0, 1.0), "px", "tiled")
    assert len(_decode(page)[0]["regions"]) == result.object_count


def test_the_app_shows_surfaces_not_a_dot_cloud():
    app = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    next(r for r in app.sidebar.radio if r.label == "Mode").set_value("3D z-stack").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    titles = [s.value for s in app.subheader]
    assert any("Interactive 3D segmentation (surfaces)" in t for t in titles)
    assert not [s for s in app.slider if "point limit" in s.label]        # the old dots are gone
