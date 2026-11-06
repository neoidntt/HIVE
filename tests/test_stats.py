"""Group comparison: tests must fit the sample size, and say what they did."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from hive import stats
from hive.pipeline import analyze_folder

from helpers import quick_config, write_synthetic_folder


def _table(groups: dict[str, list[float]], metric="m") -> pd.DataFrame:
    rows = [(f"{g}{i}.tif", g, v) for g, values in groups.items() for i, v in enumerate(values)]
    return pd.DataFrame(rows, columns=["image", "condition", metric])


def test_hedges_g_against_the_textbook_formula():
    a, b = np.array([5.0, 6, 7, 8]), np.array([2.0, 3, 4, 5])
    pooled = np.sqrt((3 * a.var(ddof=1) + 3 * b.var(ddof=1)) / 6)
    expected = (a.mean() - b.mean()) / pooled * (1 - 3 / (4 * 8 - 9))
    g, low, high = stats.hedges_g(a, b)
    assert g == pytest.approx(expected) and low < g < high
    assert stats.hedges_g(np.array([1.0]), b)[0] != stats.hedges_g(np.array([1.0]), b)[0]   # NaN
    assert np.isnan(stats.hedges_g(np.array([2.0, 2.0]), np.array([2.0, 2.0]))[0])


def test_cliffs_delta_extremes_and_sign():
    assert stats.cliffs_delta(np.array([5, 6, 7]), np.array([1, 2, 3])) == 1.0
    assert stats.cliffs_delta(np.array([1, 2, 3]), np.array([5, 6, 7])) == -1.0
    assert stats.cliffs_delta(np.array([1, 2]), np.array([1, 2])) == 0.0


def test_holm_correction_on_a_known_example():
    assert stats.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(stats.holm_adjust([0.01, float("nan")])[1])


def test_small_groups_use_the_rank_test_and_report_its_floor():
    t = _table({"a": [1, 2, 3], "b": [4, 5, 6]})
    row = stats.compare_conditions(t, ["m"]).iloc[0]
    assert row["test"] == "Mann-Whitney U"
    assert row["p_value"] == pytest.approx(sps.mannwhitneyu([1, 2, 3], [4, 5, 6]).pvalue)
    assert row["min_possible_p"] == pytest.approx(0.1)        # 2 / C(6,3): cannot go lower
    assert row["cliffs_delta"] == -1.0 and row["hedges_g"] < 0   # first group is lower


def test_large_normal_groups_use_welch_and_match_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 1, 12), rng.normal(12, 3, 12)
    row = stats.compare_conditions(_table({"x": a, "y": b}), ["m"]).iloc[0]
    assert row["test"] == "Welch t-test"
    assert row["p_value"] == pytest.approx(sps.ttest_ind(a, b, equal_var=False).pvalue)


def test_large_but_skewed_groups_fall_back_to_the_rank_test():
    rng = np.random.default_rng(1)
    skewed = rng.exponential(1.0, 15) ** 3
    row = stats.compare_conditions(_table({"x": skewed, "y": skewed * 3}), ["m"]).iloc[0]
    assert row["test"] == "Mann-Whitney U"


def test_too_few_replicates_give_an_effect_size_but_no_p_value():
    row = stats.compare_conditions(_table({"a": [1, 2], "b": [5, 6]}), ["m"]).iloc[0]
    assert row["test"] == "none" and np.isnan(row["p_value"])
    assert not np.isnan(row["hedges_g"]) and "fewer than 3" in row["note"]


def test_three_groups_get_an_overall_test_and_corrected_pairs():
    t = _table({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10, 11, 12]})
    out = stats.compare_conditions(t, ["m"])
    overall = out[out["comparison"] == "all groups"].iloc[0]
    assert overall["test"] == "Kruskal-Wallis"
    assert overall["p_value"] == pytest.approx(sps.kruskal([1, 2, 3, 4], [5, 6, 7, 8],
                                                           [9, 10, 11, 12]).pvalue)
    pairs = out[out["comparison"] != "all groups"]
    assert len(pairs) == 3 and (pairs["p_holm"] >= pairs["p_value"]).all()


def test_one_condition_means_nothing_to_compare():
    assert stats.compare_conditions(_table({"a": [1, 2, 3]})).empty
    assert stats.compare_conditions(pd.DataFrame()).empty
    assert stats.compare_conditions(_table({"a": [1, 2, 3], "b": [4, 5, 6]})).empty   # no known metric


def test_conditions_match_by_name_or_stem_and_unlabelled_images_are_left_out():
    summary = pd.DataFrame({"image": ["a.ome.tif", "b.tif", "c.tif"], "object_count": [1, 2, 3]})
    table = stats.replicate_table(summary, pd.DataFrame(), {"a.ome.tif": "x", "b": "y"})
    assert table["image"].tolist() == ["a.ome.tif", "b.tif"]
    assert table["condition"].tolist() == ["x", "y"]
    with pytest.raises(ValueError, match="level"):
        stats.replicate_table(summary, pd.DataFrame(), {}, level="cell")


def test_object_level_is_labelled_as_treating_cells_as_independent():
    summary = pd.DataFrame({"image": ["a", "b"], "object_count": [2, 2]})
    meas = pd.DataFrame({"image": ["a"] * 4 + ["b"] * 4, "label": range(8),
                         "intensity_mean": [1, 2, 3, 4, 5, 6, 7, 8.0]})
    table = stats.replicate_table(summary, meas, {"a": "ctl", "b": "trt"}, level="object")
    out = stats.compare_conditions(table, level="object")
    assert (out["unit_of_replication"].str.contains("independent")).all()
    assert out.iloc[0]["n_first"] == 4          # four cells, though only one image per group


def test_conditions_csv_is_read_and_validated(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("image, condition\nA.tif, ctl \nB.tif,trt\n")
    assert stats.load_conditions(path) == {"A.tif": "ctl", "B.tif": "trt"}
    path.write_text("name,group\nA,x\n")
    with pytest.raises(ValueError, match="image and condition"):
        stats.load_conditions(path)


def test_the_plot_is_reproducible_for_a_given_seed():
    t = _table({"a": [1.0, 2, 3, 4], "b": [3.0, 4, 5, 6]})
    first, second = stats.plot_groups(t, "m", seed=5), stats.plot_groups(t, "m", seed=5)
    offsets = lambda fig: fig.axes[0].collections[0].get_offsets().data.tolist()
    assert offsets(first) == offsets(second)
    assert offsets(first) != stats.plot_groups(t, "m", seed=6).axes[0].collections[0].get_offsets().data.tolist()


def test_a_batch_run_with_conditions_writes_stats_figures_and_an_excel_sheet(tmp_path):
    write_synthetic_folder(tmp_path / "in", names=["ctl1.tif", "ctl2.tif", "ctl3.tif"])
    cfg = quick_config(tmp_path, conditions={"ctl1.tif": "control", "ctl2.tif": "control",
                                             "ctl3.tif": "treated"})
    out = analyze_folder(cfg)
    folder = tmp_path / "out" / "group_comparison"
    assert (folder / "group_statistics.csv").exists() and (folder / "object_count.png").exists()
    with pd.ExcelFile(tmp_path / "out" / "HIVE_2D_report.xlsx") as book:
        sheets = book.sheet_names
    assert "Group_Stats" in sheets and "Group_Data" in sheets and "Settings" in sheets
    assert out["group_stats"] is not None


def test_a_run_without_conditions_makes_no_group_output(tmp_path):
    write_synthetic_folder(tmp_path / "in")
    analyze_folder(quick_config(tmp_path))
    assert not (tmp_path / "out" / "group_comparison").exists()
    with pd.ExcelFile(tmp_path / "out" / "HIVE_2D_report.xlsx") as book:
        assert "Group_Stats" not in book.sheet_names
