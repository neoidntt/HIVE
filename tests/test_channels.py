"""Multi-channel measurement and colocalization."""

from __future__ import annotations

import numpy as np
import pytest

from hive.channels import (
    add_channel_measurements, colocalization, rgb_extras, rgb_extras_volume, safe_name,
)
from hive.pipeline import analyze_array, analyze_zstack

from helpers import quick_config, write_ome


def _two_square_objects():
    labels = np.zeros((40, 60), np.int32)
    labels[5:20, 5:20] = 1
    labels[5:20, 35:50] = 2
    return labels


def test_perfectly_correlated_channels_give_r_one_and_independent_ones_near_zero():
    labels = _two_square_objects()
    rng = np.random.default_rng(3)
    main = rng.uniform(10, 100, labels.shape).astype(np.float32)
    copy = main * 2 + 5
    table, whole = colocalization(labels, main, copy, np.array([1, 2]))
    assert table["pearson"].tolist() == pytest.approx([1.0, 1.0])
    assert whole["pearson"] == pytest.approx(1.0)

    unrelated = rng.uniform(10, 100, labels.shape).astype(np.float32)
    table, _ = colocalization(labels, main, unrelated, np.array([1, 2]))
    assert table["pearson"].abs().max() < 0.25


def test_pearson_matches_numpy_for_an_object():
    labels = _two_square_objects()
    rng = np.random.default_rng(4)
    x, y = rng.normal(50, 9, labels.shape), rng.normal(50, 9, labels.shape) + 0.5 * labels
    table, _ = colocalization(labels, x.astype(np.float32), y.astype(np.float32), np.array([1, 2]))
    inside = labels == 2
    assert table["pearson"].iloc[1] == pytest.approx(np.corrcoef(x[inside], y[inside])[0, 1], abs=1e-5)


def test_manders_fractions_on_a_hand_built_case():
    labels = np.zeros((20, 20), np.int32)
    labels[0:10, 0:10] = 1
    main = np.zeros_like(labels, dtype=np.float32)
    main[0:10, 0:10] = 10.0                          # main channel everywhere in the object
    other = np.zeros_like(main)
    other[0:10, 0:5] = 100.0                         # other channel in the left half only
    other[15:20, 15:20] = 100.0                      # and a bright patch outside the object
    table, _ = colocalization(labels, main, other, np.array([1]))
    assert table["manders_m1"].iloc[0] == pytest.approx(0.5)     # half the main signal overlaps
    assert table["manders_m2"].iloc[0] == pytest.approx(1.0)     # all of the other's signal inside is on main


def test_undefined_cases_are_nan_not_numbers():
    labels = _two_square_objects()
    main = np.random.default_rng(1).uniform(1, 9, labels.shape).astype(np.float32)
    flat = np.full(labels.shape, 7.0, np.float32)
    table, whole = colocalization(labels, main, flat, np.array([1, 2]))
    assert table["pearson"].isna().all()                 # no variation in one channel
    assert table["manders_m1"].isna().all()              # a flat channel has no threshold
    tiny = np.zeros((6, 6), np.int32)
    tiny[1:3, 1:2] = 1                                   # 2 pixels: too few for a correlation
    t, _ = colocalization(tiny, main[:6, :6], main[:6, :6], np.array([1]))
    assert np.isnan(t["pearson"].iloc[0])


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must match"):
        colocalization(np.zeros((4, 4), int), np.zeros((4, 4)), np.zeros((5, 5)), np.array([1]))


def test_rgb_extras_and_clear_error_for_single_channel_images():
    rgb = np.random.default_rng(0).uniform(0, 255, (16, 16, 3)).astype(np.float32)
    out = rgb_extras(rgb, ["red", "green"])
    assert np.array_equal(out["red"], rgb[:, :, 0]) and out["green"].shape == (16, 16)
    with pytest.raises(ValueError, match="multi-channel"):
        rgb_extras(rgb[:, :, 0], ["red"])
    vol = np.stack([rgb] * 3)
    assert rgb_extras_volume(vol, ["blue"])["blue"].shape == (3, 16, 16)
    with pytest.raises(ValueError, match="multi-channel"):
        rgb_extras_volume(vol[..., 0], ["blue"])


def test_channel_names_are_safe_inside_column_names():
    assert safe_name("Alexa 488 (nm)") == "Alexa_488_nm"
    assert safe_name("  ") == "channel"


def test_analysis_adds_columns_for_every_extra_channel(tmp_path):
    main = np.full((80, 80), 20, np.float32)
    green = np.full((80, 80), 20, np.float32)
    main[10:30, 10:30] += 300                        # object A: in both channels
    green[10:30, 10:30] += 600
    main[50:70, 50:70] += 300                        # object B: only in the main channel
    cfg = quick_config(tmp_path, extra_channels=["GFP"])
    result = analyze_array(main, "img", cfg, extras={"GFP": green}, channel_name="DAPI")
    m = result.measurements.sort_values("centroid-0").reset_index(drop=True)
    assert {"GFP_intensity_mean", "GFP_integrated_intensity", "coloc_GFP_pearson",
            "coloc_GFP_manders_m1", "coloc_GFP_manders_m2"} <= set(m.columns)
    assert m["GFP_intensity_mean"].iloc[0] > 400 > m["GFP_intensity_mean"].iloc[1]
    assert m["GFP_integrated_intensity"].iloc[0] == pytest.approx(
        m["GFP_intensity_mean"].iloc[0] * m["area_px"].iloc[0])
    assert {"GFP_mean_intensity", "coloc_GFP_pearson", "coloc_GFP_manders_m1"} <= set(result.summary.columns)
    # the main channel's own numbers are unchanged by adding a second channel
    plain = analyze_array(main, "img", quick_config(tmp_path))
    assert plain.measurements["intensity_mean"].tolist() == pytest.approx(
        result.measurements["intensity_mean"].tolist())


def test_rgb_image_with_extra_colours(tmp_path):
    rgb = np.full((60, 60, 3), 10, np.float32)
    rgb[10:30, 10:30] = [10, 400, 200]
    cfg = quick_config(tmp_path, channel="green", extra_channels=["red", "blue"])
    result = analyze_array(rgb, "rgb", cfg)
    assert {"red_intensity_mean", "blue_intensity_mean"} <= set(result.measurements.columns)
    assert result.measurements["blue_intensity_mean"].iloc[0] > result.measurements[
        "red_intensity_mean"].iloc[0]


def test_extra_channels_on_a_single_channel_image_explain_themselves(tmp_path):
    with pytest.raises(ValueError, match="multi-channel"):
        analyze_array(np.zeros((30, 30), np.float32) + 5, "x",
                      quick_config(tmp_path, extra_channels=["red"]))


def test_extra_channels_are_measured_in_3d_files_too(tmp_path):
    a = np.full((6, 40, 40), 10, np.float32)
    a[2:5, 10:25, 10:25] += 300
    b = a * 2
    path = write_ome(tmp_path / "v.ome.tif", np.stack([a, b]), "CZYX", ["main", "second"], z=1.0)
    cfg = quick_config(tmp_path, input_path=str(path), mode="3d", channel="main",
                       extra_channels=["second"])
    result = analyze_zstack(cfg)
    row = result.measurements.iloc[0]
    assert row["second_intensity_mean"] == pytest.approx(2 * row["intensity_mean"], rel=0.02)
    assert row["coloc_second_pearson"] > 0.9


def test_no_objects_means_no_new_columns_and_no_crash():
    import pandas as pd
    m, s = add_channel_measurements(pd.DataFrame(), pd.DataFrame(), np.zeros((4, 4), int),
                                    np.zeros((4, 4)), {"x": np.zeros((4, 4))}, None)
    assert m.empty
