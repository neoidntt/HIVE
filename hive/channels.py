"""Measuring more than one channel, and colocalization between channels.

Objects are found on ONE channel (``HiveConfig.channel``). Every further channel in
``HiveConfig.extra_channels`` is only *measured* at those objects, after the same
background subtraction as the main channel and with no contrast enhancement, so its
intensities are in the file's own scale and comparable between images (the same
two-image contract as :mod:`hive.preprocessing`).

Colocalization statistics, per object and per image (over all segmented pixels):

* **Pearson's r** between the two channels' intensities (Pearson, 1896; applied to
  colocalization by Manders et al., 1992). Needs at least 3 pixels and intensity
  variation in both channels, otherwise it is reported as NaN rather than invented.
* **Manders' M1 and M2** (Manders et al., 1993): M1 is the fraction of the main
  channel's intensity that lies where the other channel is above its threshold; M2 is
  the reverse. The thresholds are the Otsu threshold of each corrected channel. That
  is a simple, reproducible choice; Costes' iterative threshold (Costes et al., 2004)
  is not implemented.

Because the objects come from the main channel, M2 mostly describes how much of the
other channel's signal sits inside them, and Pearson's r is computed inside objects
only. Both depend on the segmentation; read them together with the object counts.
"""

from __future__ import annotations

import re
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters

from .baseline import apply_baseline_subtraction
from .config import HiveConfig, PreprocessConfig
from .preprocessing import select_channel, select_channel_volume


def safe_name(name: str) -> str:
    """Turn a channel name into something usable inside a column name."""
    return re.sub(r"\W+", "_", str(name).strip()).strip("_") or "channel"


def rgb_extras(image: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Pull further colour channels out of an RGB image.

    Args:
        image: Channel-last colour image ``(Y, X, 3)``.
        names: Colour names such as ``"red"``, ``"green"``, ``"magenta"``.

    Returns:
        Mapping of name to a 2D float32 plane.

    Raises:
        ValueError: If the image has no colour channels to choose from.
    """
    arr = np.asarray(image)
    if arr.ndim < 3:
        raise ValueError(
            "extra_channels needs a multi-channel image, but this image is single-channel. "
            "Use an OME-TIFF / CZI / ND2 / LIF file, or an RGB image."
        )
    return {name: select_channel(arr, name)[0] for name in names}  # type: ignore[arg-type]


def rgb_extras_volume(stack: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Pull further colour channels out of a channel-last z-stack ``(Z, Y, X, C)``."""
    arr = np.asarray(stack)
    if arr.ndim < 4:
        raise ValueError(
            "extra_channels needs a multi-channel stack, but this stack is single-channel."
        )
    return {name: select_channel_volume(arr, name)[0] for name in names}  # type: ignore[arg-type]


def correct_channel(raw: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Background-correct one channel exactly as the main channel is.

    Args:
        raw: 2D or 3D single-channel array in its original scale.
        cfg: The run's preprocessing settings.

    Returns:
        A new float32 array; the input is not modified.
    """
    corrected, _ = apply_baseline_subtraction(
        np.asarray(raw, dtype=np.float32),
        method=cfg.baseline_method, percentile=cfg.baseline_percentile,
        sigma=cfg.baseline_sigma, tophat_radius=cfg.tophat_radius,
        normalize=False, return_baseline=False, copy=True,
    )
    return corrected


def _otsu_or_none(image: np.ndarray) -> float | None:
    """Otsu threshold, or ``None`` for a flat (empty) channel."""
    if float(image.max()) <= float(image.min()):
        return None
    return float(filters.threshold_otsu(image))


def _pearson(n, sx, sy, sxx, syy, sxy) -> np.ndarray:
    """Pearson's r from running sums; NaN where it is undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sy / n
        var = (sxx - sx * sx / n) * (syy - sy * sy / n)
        r = cov / np.sqrt(var)
    r = np.where((n >= 3) & (var > 0), r, np.nan)
    return np.clip(r, -1.0, 1.0)


def colocalization(
    labels: np.ndarray, main: np.ndarray, other: np.ndarray, index: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pearson's r and Manders' M1/M2, per object and over all objects together.

    Args:
        labels: Label image.
        main: Background-corrected main channel, same shape.
        other: Background-corrected second channel, same shape.
        index: Labels to report, in the order of the measurement table.

    Returns:
        A table with ``pearson``, ``manders_m1``, ``manders_m2`` per object (one row
        per entry of ``index``), and the same three numbers over every segmented
        pixel together.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    if not (labels.shape == main.shape == other.shape):
        raise ValueError(
            f"labels {labels.shape}, main {main.shape} and other {other.shape} must match"
        )
    x = main.astype(np.float64)
    y = other.astype(np.float64)
    thr_x, thr_y = _otsu_or_none(main), _otsu_or_none(other)
    above_x = (x > thr_x) if thr_x is not None else np.zeros(x.shape, dtype=bool)
    above_y = (y > thr_y) if thr_y is not None else np.zeros(y.shape, dtype=bool)

    def per_label(values: np.ndarray) -> np.ndarray:
        return np.asarray(ndimage.sum_labels(values, labels, index=index), dtype=np.float64)

    n = per_label(np.ones_like(x))
    sx, sy = per_label(x), per_label(y)
    pearson = _pearson(n, sx, sy, per_label(x * x), per_label(y * y), per_label(x * y))
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = per_label(x * above_y) / sx
        m2 = per_label(y * above_x) / sy
    if thr_y is None:
        m1 = np.full_like(m1, np.nan)
    if thr_x is None:
        m2 = np.full_like(m2, np.nan)
    table = pd.DataFrame({"pearson": pearson, "manders_m1": m1, "manders_m2": m2})

    mask = labels > 0
    xs, ys = x[mask], y[mask]
    whole = {"pearson": float("nan"), "manders_m1": float("nan"), "manders_m2": float("nan")}
    if xs.size:
        whole["pearson"] = float(_pearson(
            np.array(xs.size), xs.sum(), ys.sum(), (xs * xs).sum(), (ys * ys).sum(),
            (xs * ys).sum(),
        ))
        if thr_y is not None and xs.sum() > 0:
            whole["manders_m1"] = float(xs[above_y[mask]].sum() / xs.sum())
        if thr_x is not None and ys.sum() > 0:
            whole["manders_m2"] = float(ys[above_x[mask]].sum() / ys.sum())
    return table, whole


def add_channel_measurements(
    measurements: pd.DataFrame, summary: pd.DataFrame, labels: np.ndarray,
    main_corrected: np.ndarray, extras: Mapping[str, np.ndarray], cfg: HiveConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add each extra channel's intensities and its colocalization with the main one.

    Columns added to the per-object table, for a channel called ``X``:
    ``X_intensity_mean``, ``X_integrated_intensity``, ``coloc_X_pearson``,
    ``coloc_X_manders_m1``, ``coloc_X_manders_m2``. The one-row summary gets the mean
    of ``X`` over objects and the image-level colocalization numbers.

    Args:
        measurements: Per-object table from the main channel.
        summary: One-row image summary.
        labels: Label image the table was measured on.
        main_corrected: The main channel's measurement image.
        extras: Raw extra channels by name, same shape as ``labels``.
        cfg: Run configuration (supplies the background settings).

    Returns:
        The two tables with the new columns.
    """
    if not extras or measurements is None or measurements.empty:
        return measurements, summary

    measurements = measurements.copy()
    summary = summary.copy()
    index = measurements["label"].to_numpy()
    size = measurements["volume_voxels" if "volume_voxels" in measurements else "area_px"]

    for name, raw in extras.items():
        tag = safe_name(name)
        corrected = correct_channel(raw, cfg.preprocess)
        means = np.asarray(ndimage.mean(corrected, labels, index=index), dtype=np.float64)
        measurements[f"{tag}_intensity_mean"] = means
        measurements[f"{tag}_integrated_intensity"] = means * size.to_numpy()
        table, whole = colocalization(labels, main_corrected, corrected, index)
        for column in table.columns:
            measurements[f"coloc_{tag}_{column}"] = table[column].to_numpy()
        summary[f"{tag}_mean_intensity"] = float(np.nanmean(means))
        for column, value in whole.items():
            summary[f"coloc_{tag}_{column}"] = value
        del corrected
    return measurements, summary
