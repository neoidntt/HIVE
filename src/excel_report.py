"""LEGACY (HIVE 1.x) Excel export.

.. deprecated:: 2.0
    Superseded by :mod:`hive.reporting`, whose Settings sheet records the
    complete flattened :class:`hive.config.HiveConfig` (~42 parameters). The
    legacy caller hand-assembles a partial ``settings`` dict, so a run
    exported here cannot be fully reproduced from its own report.
"""

import pandas as pd


def export_excel_report(path, summary_df, measurements_df, settings=None):
    """Write a Summary/Measurements/Settings workbook.

    Args:
        path: Destination ``.xlsx`` path.
        summary_df: Per-image summary rows.
        measurements_df: Per-object rows.
        settings: Optional flat mapping written to a third sheet. Omitted
            entirely when falsy, which leaves the workbook with no record of
            the parameters that produced it.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        measurements_df.to_excel(writer, index=False, sheet_name="Measurements")
        if settings:
            pd.DataFrame([settings]).to_excel(writer, index=False, sheet_name="Settings")
