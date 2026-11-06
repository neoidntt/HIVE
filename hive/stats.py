"""Comparing experimental conditions: tests, effect sizes and plots.

The unit of replication matters more than the test. Cells (or objects) in one image
are not independent samples, so testing thousands of cells as if they were is
pseudoreplication and gives p-values that are far too small. The default here
(``level="image"``) therefore first reduces each image to one number per metric and
compares images. ``level="object"`` is available, and every row it produces says
that it treats objects as independent.

Choice of test, in the order the code applies it:

* A group needs at least 3 replicates for a p-value; with fewer, only the effect
  size is given.
* **Two groups.** Welch's t-test (Welch, 1947) only when both groups have >= 8
  replicates and Shapiro-Wilk (Shapiro & Wilk, 1965) does not reject normality in
  either; otherwise the Mann-Whitney U test (Mann & Whitney, 1947). Small groups
  cannot show normality, so the rank test is the safe default. With few replicates
  its smallest possible p-value can be large, and that value is reported in
  ``min_possible_p`` so a non-significant result is not over-read.
* **Three or more groups.** Kruskal-Wallis (Kruskal & Wallis, 1952) as the overall
  test, then every pair with the two-group rule above, with Holm's correction
  (Holm, 1979) across all pairwise p-values in the table.
* **Effect sizes**, always reported: Hedges' g with a 95 % confidence interval
  (Hedges, 1981; interval from Hedges & Olkin, 1985) and Cliff's delta (Cliff, 1993).
  Positive means the first-listed group is higher.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

MIN_REPLICATES = 3          # fewest replicates per group for a p-value
WELCH_MIN_N = 8             # fewest per group before a t-test is considered
NORMALITY_ALPHA = 0.05
_SUMMARY_METRICS = (
    "object_count", "mean_intensity", "total_integrated_intensity",
)


def hedges_g(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Hedges' g (small-sample corrected standardised mean difference).

    Args:
        a: Values of the first group.
        b: Values of the second group.

    Returns:
        ``(g, ci_low, ci_high)``; all NaN when the pooled spread is zero or a
        group has fewer than 2 values.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
                     / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan"), float("nan"), float("nan")
    d = (np.mean(a) - np.mean(b)) / pooled
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    g = d * correction
    se = np.sqrt((n1 + n2) / (n1 * n2) + g * g / (2.0 * (n1 + n2 - 2)))
    half = sps.t.ppf(0.975, n1 + n2 - 2) * se
    return float(g), float(g - half), float(g + half)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b), from -1 to 1, no distribution assumed."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm's step-down correction; NaN entries are left out and stay NaN.

    Args:
        p_values: Raw p-values, in any order.

    Returns:
        Adjusted p-values in the same order.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def _normal_enough(x: np.ndarray) -> bool:
    """Whether Shapiro-Wilk fails to reject normality (needs >= 3 values)."""
    return len(x) >= 3 and np.ptp(x) > 0 and sps.shapiro(x).pvalue >= NORMALITY_ALPHA


def _pair(a: np.ndarray, b: np.ndarray, test: str) -> dict:
    """Test two groups and describe the difference."""
    row: dict = {"test": "none", "statistic": np.nan, "p_value": np.nan,
                 "min_possible_p": np.nan, "note": ""}
    if min(len(a), len(b)) < MIN_REPLICATES:
        row["note"] = f"fewer than {MIN_REPLICATES} replicates in a group: no p-value"
    else:
        use = test
        if use == "auto":
            ok = min(len(a), len(b)) >= WELCH_MIN_N and _normal_enough(a) and _normal_enough(b)
            use = "welch" if ok else "mannwhitney"
        if use == "welch":
            res = sps.ttest_ind(a, b, equal_var=False)
            row.update(test="Welch t-test", statistic=float(res.statistic),
                       p_value=float(res.pvalue))
        elif use == "mannwhitney":
            res = sps.mannwhitneyu(a, b, alternative="two-sided", method="auto")
            row.update(test="Mann-Whitney U", statistic=float(res.statistic),
                       p_value=float(res.pvalue))
            from math import comb
            row["min_possible_p"] = min(1.0, 2.0 / comb(len(a) + len(b), len(a)))
        else:
            raise ValueError(f"test must be auto, welch or mannwhitney, got {test!r}")
    g, low, high = hedges_g(a, b)
    row.update(hedges_g=g, g_ci_low=low, g_ci_high=high, cliffs_delta=cliffs_delta(a, b))
    return row


def load_conditions(path: str | Path) -> dict[str, str]:
    """Read a CSV with the columns ``image`` and ``condition``.

    Args:
        path: CSV file; extra columns are ignored.

    Returns:
        Image name -> condition.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing.
    """
    table = pd.read_csv(path, dtype=str).rename(columns=str.strip)
    missing = {"image", "condition"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} needs the columns image and condition (missing: "
                         f"{', '.join(sorted(missing))})")
    table = table.dropna(subset=["image", "condition"])
    return {str(i).strip(): str(c).strip() for i, c in zip(table["image"], table["condition"])}


def replicate_table(
    summary: pd.DataFrame, measurements: pd.DataFrame, conditions: dict[str, str],
    level: str = "image",
) -> pd.DataFrame:
    """Attach a condition to each replicate and choose the metrics to compare.

    Args:
        summary: Per-image summary table.
        measurements: Per-object table.
        conditions: Image name -> condition. Names are matched exactly, then by
            file stem. Images without a condition are left out.
        level: ``"image"`` (one value per image, the default) or ``"object"``.

    Returns:
        A table with ``image``, ``condition`` and the metric columns.

    Raises:
        ValueError: On an unknown ``level``.
    """
    if level not in ("image", "object"):
        raise ValueError(f"level must be 'image' or 'object', got {level!r}")
    source = summary if level == "image" else measurements
    if source is None or source.empty:
        return pd.DataFrame()
    stems = {Path(k).stem: v for k, v in conditions.items()}
    table = source.copy()
    table.insert(1, "condition", [
        conditions.get(str(name), stems.get(Path(str(name)).stem)) for name in table["image"]
    ])
    table = table[table["condition"].notna() & (table["condition"].astype(str) != "")]
    return table.reset_index(drop=True)


def default_metrics(table: pd.DataFrame, level: str = "image") -> list[str]:
    """Metric columns worth comparing in a replicate table."""
    if level == "image":
        wanted = list(_SUMMARY_METRICS) + [
            c for c in table.columns
            if c.startswith(("mean_area", "mean_volume", "coloc_")) or c.endswith("_mean_intensity")
        ]
    else:
        wanted = [c for c in table.columns if c.startswith(("area_px", "volume_voxels"))
                  ] + ["intensity_mean", "integrated_intensity"]
    seen: list[str] = []
    for column in wanted:
        if column in table.columns and column not in seen and pd.api.types.is_numeric_dtype(table[column]):
            seen.append(column)
    return seen


def compare_conditions(
    table: pd.DataFrame, metrics: list[str] | None = None, test: str = "auto",
    level: str = "image",
) -> pd.DataFrame:
    """Compare every metric between the conditions in a replicate table.

    Args:
        table: Output of :func:`replicate_table`.
        metrics: Columns to compare; ``None`` picks sensible ones.
        test: ``"auto"`` (see the module notes), ``"welch"`` or ``"mannwhitney"``.
        level: Only used to label the unit of replication in the output.

    Returns:
        One row per metric and comparison, with group sizes, means, test, p-value,
        Holm-adjusted p-value and effect sizes. Empty if there is nothing to compare.
    """
    if table is None or table.empty or table["condition"].nunique() < 2:
        return pd.DataFrame()
    metrics = metrics or default_metrics(table, level)
    if not metrics:
        return pd.DataFrame()
    groups = list(dict.fromkeys(table["condition"].astype(str)))   # order of appearance
    unit = "image" if level == "image" else "object (treated as independent)"
    rows: list[dict] = []
    for metric in metrics:
        values = {g: table.loc[table["condition"].astype(str) == g, metric]
                  .dropna().to_numpy(dtype=float) for g in groups}
        if len(groups) > 2 and all(len(v) >= MIN_REPLICATES for v in values.values()):
            h = sps.kruskal(*values.values())
            n_total, k = sum(len(v) for v in values.values()), len(groups)
            rows.append({
                "metric": metric, "comparison": "all groups", "test": "Kruskal-Wallis",
                "statistic": float(h.statistic), "p_value": float(h.pvalue),
                "effect_size_epsilon_sq": float((h.statistic - k + 1) / (n_total - k)),
                "unit_of_replication": unit,
            })
        for first, second in combinations(groups, 2):
            a, b = values[first], values[second]
            row = _pair(a, b, test)
            row.update(
                metric=metric, comparison=f"{first} vs {second}",
                n_first=len(a), n_second=len(b),
                mean_first=float(np.mean(a)) if len(a) else np.nan,
                mean_second=float(np.mean(b)) if len(b) else np.nan,
                unit_of_replication=unit,
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    pairwise = out["comparison"] != "all groups"
    out["p_holm"] = np.nan
    out.loc[pairwise, "p_holm"] = holm_adjust(out.loc[pairwise, "p_value"].tolist())
    first = ["metric", "comparison", "test", "n_first", "n_second", "mean_first", "mean_second",
             "statistic", "p_value", "p_holm", "min_possible_p", "hedges_g", "g_ci_low",
             "g_ci_high", "cliffs_delta"]
    ordered = [c for c in first if c in out.columns]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def plot_groups(table: pd.DataFrame, metric: str, unit_label: str = "image", seed: int = 0):
    """Box plot with every replicate shown as a point.

    Args:
        table: Output of :func:`replicate_table`.
        metric: Column to plot.
        unit_label: What one point is (for the axis text).
        seed: Seeds the horizontal jitter so the figure is reproducible.

    Returns:
        A matplotlib ``Figure`` (not attached to pyplot, so no global state).
    """
    from matplotlib.figure import Figure

    groups = list(dict.fromkeys(table["condition"].astype(str)))
    rng = np.random.default_rng(seed)
    fig = Figure(figsize=(1.6 + 1.2 * len(groups), 4.2), layout="constrained")
    ax = fig.subplots()
    data = [table.loc[table["condition"].astype(str) == g, metric].dropna().to_numpy(float)
            for g in groups]
    ax.boxplot(data, tick_labels=groups, showfliers=False, widths=0.5)
    for i, values in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=22, zorder=3,
                   color="#0d2942", alpha=0.85)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}  (each point = one {unit_label})", fontsize=9)
    return fig


def write_group_outputs(
    folder: str | Path, summary: pd.DataFrame, measurements: pd.DataFrame,
    conditions: dict[str, str], seed: int = 0, level: str = "image", test: str = "auto",
) -> dict[str, pd.DataFrame]:
    """Run the comparison and write its tables and figures into ``folder``.

    Args:
        folder: Output folder; a ``group_comparison`` subfolder is created.
        summary: Per-image summary.
        measurements: Per-object table.
        conditions: Image name -> condition.
        seed: Seed for plot jitter.
        level: ``"image"`` or ``"object"``.
        test: Test choice, see :func:`compare_conditions`.

    Returns:
        ``{"Group_Stats": ..., "Group_Data": ...}`` for the Excel report. Both are
        empty when fewer than two conditions are present.
    """
    data = replicate_table(summary, measurements, conditions, level)
    stats = compare_conditions(data, test=test, level=level)
    if stats.empty:
        return {"Group_Stats": pd.DataFrame(), "Group_Data": pd.DataFrame()}
    out = Path(folder) / "group_comparison"
    out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out / "group_statistics.csv", index=False)
    data.to_csv(out / "group_data.csv", index=False)
    for metric in stats["metric"].unique():
        figure = plot_groups(data, metric, "image" if level == "image" else "object", seed)
        figure.savefig(out / f"{metric}.png", dpi=150)
    return {"Group_Stats": stats, "Group_Data": data}
