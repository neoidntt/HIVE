"""Consolidated data export (Phase 3.2).

A run's numbers are only reusable if the parameters that produced them travel
with them, so every export carries a Settings sheet holding the full flattened
config. The legacy Excel export wrote a hand-built dict of five or six fields,
so most parameters that shaped a result were never recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def export_excel_report(
    path: str | Path,
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    sholl_curves: pd.DataFrame | None = None,
) -> Path:
    """Write a multi-sheet Excel report.

    Sheets: ``Summary`` (per-image), ``Measurements`` (per-object), optionally
    ``Sholl_Curves`` (the full radius-vs-intersections profiles), and
    ``Settings`` (every parameter used).

    Args:
        path: Output ``.xlsx`` path.
        summary: Per-image summary table.
        measurements: Per-object table.
        settings: Flattened run configuration.
        sholl_curves: Long-format Sholl profiles.

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
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        (summary if summary is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Summary"
        )
        (measurements if measurements is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Measurements"
        )
        if sholl_curves is not None and not sholl_curves.empty:
            sholl_curves.to_excel(writer, index=False, sheet_name="Sholl_Curves")
        if settings:
            pd.DataFrame(
                {"parameter": list(settings.keys()),
                 "value": [str(v) for v in settings.values()]}
            ).to_excel(writer, index=False, sheet_name="Settings")
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
    frames = []
    for label, sholl in curves.items():
        if sholl.radii.size == 0:
            continue
        frame = sholl.to_frame()
        frame.insert(0, "label", label)
        frame.insert(0, "image", image_name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
