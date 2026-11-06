"""Consolidated data export (Phase 3.2).

A run's numbers are only reusable if the parameters that produced them travel
with them, so every export carries a Settings sheet holding the full flattened
config. The legacy Excel export wrote a hand-built dict of five or six fields,
so most parameters that shaped a result were never recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def consolidate(
    morphology: pd.DataFrame,
    arborization: pd.DataFrame | None = None,
    image_name: str = "",
) -> pd.DataFrame:
    """Merge morphology and arborisation tables into one per-object table.

    Joined on ``label`` with a left join, so an object always survives even if
    arborisation produced nothing for it (an object too small to skeletonise
    keeps its morphology row with empty arbor columns, rather than vanishing).

    Args:
        morphology: Output of :func:`hive.measurements.measure_objects`.
        arborization: Output of :func:`hive.arborization.analyze_arbor`.
        image_name: Image name, backfilled if the morphology table lacks it.

    Returns:
        One row per object with morphology and arborisation side by side.
    """
    if morphology is None or morphology.empty:
        return pd.DataFrame()

    df = morphology.copy()
    if arborization is not None and not arborization.empty:
        df = df.merge(arborization, on="label", how="left")
    if "image" not in df.columns:
        df.insert(0, "image", image_name)
    return df


def export_csv(path: str | Path, df: pd.DataFrame) -> Path:
    """Write a DataFrame to CSV.

    Args:
        path: Output ``.csv`` path.
        df: Table to write.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


EXCEL_MAX_ROWS = 1_048_575      # data rows an .xlsx sheet can hold (the header takes the first)


def _fit_sheet(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """The table itself, or a one-line explanation if it has too many rows for Excel.

    A very large image can hold millions of objects. Writing them would make Excel's own limit
    abort the whole report after hours of analysis; the full table is always in the CSV.
    """
    if len(table) <= EXCEL_MAX_ROWS:
        return table
    return pd.DataFrame({"note": [
        f"The {name} table has {len(table):,} rows, more than an Excel sheet can hold "
        f"({EXCEL_MAX_ROWS:,}). The complete table is in the matching CSV file."]})


def export_excel_report(
    path: str | Path,
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    sholl_curves: pd.DataFrame | None = None,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Write a multi-sheet Excel report.

    Sheets: ``Summary`` (per-image), ``Measurements`` (per-object), optionally
    ``Sholl_Curves`` (the full radius-vs-intersections profiles), and
    ``Settings`` (every parameter used), and any ``extra_sheets`` (for example the
    group comparison).

    Args:
        path: Output ``.xlsx`` path.
        summary: Per-image summary table.
        measurements: Per-object table.
        settings: Flattened run configuration.
        sholl_curves: Long-format Sholl profiles.
        extra_sheets: Further tables to write, by sheet name.

    Returns:
        The written path.

    Raises:
        ImportError: If ``openpyxl`` is not installed.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Excel export needs openpyxl: pip install openpyxl. "
            "(CSV export has no extra dependency.)"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # xlsxwriter writes the same sheets and values several times faster than openpyxl, which
    # is still used when xlsxwriter is not installed.
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"
    with pd.ExcelWriter(path, engine=engine) as writer:
        (summary if summary is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Summary"
        )
        _fit_sheet(measurements if measurements is not None else pd.DataFrame(),
                   "Measurements").to_excel(writer, index=False, sheet_name="Measurements")
        if sholl_curves is not None and not sholl_curves.empty:
            _fit_sheet(sholl_curves, "Sholl_Curves").to_excel(
                writer, index=False, sheet_name="Sholl_Curves")
        if settings:
            pd.DataFrame(
                {"parameter": list(settings.keys()),
                 "value": [str(v) for v in settings.values()]}
            ).to_excel(writer, index=False, sheet_name="Settings")
        for sheet, table in (extra_sheets or {}).items():
            if table is not None and not table.empty:
                _fit_sheet(table, sheet).to_excel(writer, index=False, sheet_name=sheet[:31])
    return path


def sholl_curves_to_frame(curves: dict[int, Any], image_name: str = "") -> pd.DataFrame:
    """Flatten per-object Sholl profiles into one long-format table.

    Long format (one row per object-radius pair) rather than wide, because
    objects have different numbers of shells; a wide table would be ragged and
    padded with meaningless zeros.

    Args:
        curves: Mapping of label -> ``ShollResult``.
        image_name: Image name for the ``image`` column.

    Returns:
        Long-format DataFrame with ``image``, ``label``, ``radius``,
        ``intersections``.
    """
    labels, radii, counts = [], [], []
    for label, sholl in curves.items():
        if sholl.radii.size == 0:
            continue
        labels.append(np.full(sholl.radii.size, label))
        radii.append(sholl.radii)
        counts.append(sholl.intersections)
    if not labels:
        return pd.DataFrame()
    # One table built from arrays: a DataFrame per object, inserted into and concatenated,
    # cost seconds for a few hundred objects.
    return pd.DataFrame({
        "image": image_name, "label": np.concatenate(labels),
        "radius": np.concatenate(radii), "intersections": np.concatenate(counts),
    })
