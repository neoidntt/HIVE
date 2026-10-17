"""Baseline / background subtraction for fluorescence images and z-stacks.

Ported from the legacy ``src/baseline_subtraction.py``, which was already
sound; changes here are type hints and integration, not algorithm.

Methods
-------
``none``
    No baseline removed (still honours ``normalize`` for a consistent contract).
``percentile``
    Subtracts a single global scalar (a low intensity percentile). Good for a
    roughly flat background. Cheap and robust, but blind to illumination gradients.
``smooth``
    Subtracts a Gaussian-smoothed background estimate. Handles uneven
    illumination, but a Gaussian is not a lower-envelope estimator and can leave
    a faint dark halo around bright objects.
``tophat``
    Subtracts a morphological white top-hat estimate (grey opening) -- the same
    family as ImageJ's rolling-ball. Handles uneven illumination *without* the
    halo, because it estimates an actual lower envelope. Prefer this for images
    with large or dense bright structures.

Dimensionality contract
-----------------------
These functions accept a 2D image or a 3D z-stack ``(Z, Y, X)`` ONLY. A
channel-last RGB image ``(Y, X, 3)`` passed here would have its channel axis
blurred as if it were Z. Callers must reduce to a single channel first --
:func:`hive.preprocessing.select_channel` does this. The legacy pipeline called
this module *before* channel selection and crashed on every RGB z-stack with
``ValueError: Expected a 2D image or 3D z-stack, got shape (Z, Y, X, 3)``.
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion

BaselineMethod = Literal["none", "percentile", "smooth", "tophat"]
Scale = float | Sequence[float]


def _validate(volume: np.ndarray, copy: bool = True) -> np.ndarray:
    """Coerce input to float32 and reject unusable arrays.

    Args:
        volume: 2D image or 3D z-stack.
        copy: Return an array this module owns and may modify in place. Pass
            ``False`` ONLY when the caller has already made a defensive copy and
            is handing over ownership — otherwise the caller's array gets
            mutated, violating the project's "never mutate raw data" rule.

    Returns:
        float32 array, owned when ``copy=True``.

    Raises:
        ValueError: If the array is not 2D/3D, or contains NaN/Inf. Non-finite
            values would silently poison percentiles and filter kernels.
    """
    arr = np.asarray(volume)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D image or 3D z-stack, got shape {arr.shape}. "
            "Multi-channel data must be reduced to one channel first "
            "(see hive.preprocessing.select_channel)."
        )
    if copy or arr.dtype != np.float32:
        arr = np.array(arr, dtype=np.float32)

    # Finiteness via a float64 sum rather than np.isfinite(arr).all(), which
    # materialises a full boolean array (25% of the volume, purely to be thrown
    # away). NaN and Inf both propagate through the sum, and +Inf/-Inf cancelling
    # yields NaN, so nothing escapes. The float64 accumulator cannot overflow on
    # any real image: 17e9 voxels x 65535 is ~1e15, far below float64's ~1e308.
    if not np.isfinite(arr.sum(dtype=np.float64)):
        raise ValueError(
            "Input contains NaN or infinite values; clean these before baseline subtraction."
        )
    return arr


def _normalize_inplace(arr: np.ndarray, method: str = "percentile",
                       clip_percentile: float = 99.9) -> np.ndarray:
    """Rescale an owned array to ``[0, 1]`` in place.

    Same contract as :func:`_normalize` but allocates nothing. The caller must
    own ``arr``.

    Args:
        arr: Owned, non-negative float32 array. Modified in place.
        method: ``"max"`` or ``"percentile"``.
        clip_percentile: Percentile used when ``method="percentile"``.

    Returns:
        ``arr``, rescaled.

    Raises:
        ValueError: On an unknown ``method``.
    """
    if arr.max() <= 0:
        return arr
    if method == "max":
        scale = float(arr.max())
    elif method == "percentile":
        scale = float(np.percentile(arr, clip_percentile))
        if scale <= 0:
            scale = float(arr.max())
    else:
        raise ValueError(f"Unknown normalize_method: {method!r}; use 'max' or 'percentile'")
    np.divide(arr, scale, out=arr)
    np.clip(arr, 0, 1, out=arr)
    return arr


def _grey_opening_into(arr: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    """Compute a grey opening (erosion then dilation) with explicit buffers.

    ``scipy.ndimage.grey_opening`` allocates its own intermediate and its output.
    Driving erosion and dilation directly lets us reuse one scratch buffer, so
    the peak is the input plus two buffers instead of the input plus three.

    Args:
        arr: Input array.
        size: Structuring-element size per axis.

    Returns:
        The opened array (the background estimate).
    """
    scratch = np.empty_like(arr)
    baseline = np.empty_like(arr)
    grey_erosion(arr, size=size, output=scratch)
    grey_dilation(scratch, size=size, output=baseline)
    del scratch
    return baseline


def _normalize(
    arr: np.ndarray, method: str = "percentile", clip_percentile: float = 99.9,
) -> np.ndarray:
    """Rescale a non-negative array to ``[0, 1]``.

    Args:
        arr: Non-negative input.
        method: ``"max"`` divides by the true maximum -- simple, but one
            hot/saturated pixel dominates the scale. ``"percentile"`` divides by
            a high percentile and clips, which is robust to single-pixel outliers
            at the cost of clipping the brightest fraction of a percent.
        clip_percentile: Percentile used when ``method="percentile"``.

    Returns:
        Rescaled array.

    Raises:
        ValueError: On an unknown ``method``.
    """
    if arr.max() <= 0:
        return arr
    if method == "max":
        scale = float(arr.max())
    elif method == "percentile":
        scale = float(np.percentile(arr, clip_percentile))
        if scale <= 0:
            scale = float(arr.max())
    else:
        raise ValueError(f"Unknown normalize_method: {method!r}; use 'max' or 'percentile'")
    return np.clip(arr / scale, 0, 1)


def _check_anisotropic_scale(arr: np.ndarray, value: Scale, name: str) -> None:
    """Warn when a scalar sigma/radius exceeds a stack's z-depth.

    A scalar >= the number of planes flattens the z-axis baseline to a single
    value with no z-dependence -- verified on a 12-plane stack with sigma=20,
    which produced an identical baseline at every plane.

    Args:
        arr: The array being filtered.
        value: The scalar or per-axis scale.
        name: Parameter name, for the message.
    """
    is_flat_scalar = np.isscalar(value) and float(value) >= arr.shape[0]  # type: ignore[arg-type]
    if arr.ndim == 3 and is_flat_scalar:
        warnings.warn(
            f"{name}={value} is >= the stack depth ({arr.shape[0]} planes); the z-axis "
            f"baseline will collapse to a single flat value with no z-dependence. Pass a "
            f"per-axis tuple instead, e.g. {name}=(2, {value}, {value}).",
            stacklevel=3,
        )


def subtract_percentile_baseline(
    volume: np.ndarray, percentile: float = 10, normalize: bool = False,
    normalize_method: str = "percentile", copy: bool = True,
) -> tuple[np.ndarray, float]:
    """Subtract a single global low-percentile offset.

    Assumes background pixels are at least ``percentile``% of the array. With
    densely labelled samples where signal fills the frame, lower ``percentile``
    or use :func:`subtract_tophat_baseline` instead.

    Args:
        volume: 2D image or 3D z-stack.
        percentile: Intensity percentile taken as background.
        normalize: Rescale the result to ``[0, 1]``.
        normalize_method: See :func:`_normalize`.

    Returns:
        Tuple of (corrected array, subtracted scalar) -- the scalar is kept for
        batch QC logging.

    Raises:
        ValueError: If the input is not 2D/3D or is non-finite.
    """
    arr = _validate(volume, copy=copy)
    baseline = float(np.percentile(arr, percentile))
    # In-place: the scalar offset needs no temporary at all.
    np.subtract(arr, baseline, out=arr)
    np.clip(arr, 0, None, out=arr)
    if normalize:
        _normalize_inplace(arr, method=normalize_method)
    return arr, baseline


def subtract_smooth_baseline(
    volume: np.ndarray, sigma: Scale = 20, normalize: bool = False,
    normalize_method: str = "percentile", return_baseline: bool = True,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Subtract a Gaussian-smoothed background estimate.

    Handles uneven illumination but can leave a dark halo around bright objects,
    since a Gaussian blur is symmetric rather than a lower envelope. Measured on
    a synthetic image, this overestimated a flat background by ~400 units about
    25 px from a bright spot, clipping to a dark ring after subtraction.

    Args:
        volume: 2D image or 3D z-stack.
        sigma: Scalar, or per-axis ``(sigma_z, sigma_y, sigma_x)``. Use a tuple
            whenever z-spacing differs from the xy pixel size.
        normalize: Rescale the result to ``[0, 1]``.
        normalize_method: See :func:`_normalize`.

    Returns:
        Tuple of (corrected array, estimated background array).

    Raises:
        ValueError: If the input is not 2D/3D or is non-finite.
    """
    arr = _validate(volume, copy=copy)
    _check_anisotropic_scale(arr, sigma, "sigma")
    baseline = np.empty_like(arr)
    gaussian_filter(arr, sigma=sigma, output=baseline)
    np.subtract(arr, baseline, out=arr)
    np.clip(arr, 0, None, out=arr)
    if normalize:
        _normalize_inplace(arr, method=normalize_method)
    if not return_baseline:
        del baseline
        return arr, None
    return arr, baseline


def subtract_tophat_baseline(
    volume: np.ndarray, radius: Scale = 20, normalize: bool = False,
    normalize_method: str = "percentile", return_baseline: bool = True,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Subtract a morphological white top-hat background estimate.

    Grey opening (erosion then dilation) computes an actual lower envelope, so
    it does not "see" bright objects when estimating the background beneath
    them -- no halo artifact, unlike Gaussian smoothing. Same algorithm family
    as ImageJ's rolling-ball subtraction.

    Args:
        volume: 2D image or 3D z-stack.
        radius: Scalar, or per-axis radii for anisotropic stacks. Must exceed the
            largest real feature you want to keep; too small and real signal is
            eaten as background.
        normalize: Rescale the result to ``[0, 1]``.
        normalize_method: See :func:`_normalize`.

    Returns:
        Tuple of (corrected array, estimated background array). ``baseline`` is
        ``None`` when ``return_baseline=False``.

    Raises:
        ValueError: If the input is not 2D/3D or is non-finite.
    """
    arr = _validate(volume, copy=copy)
    _check_anisotropic_scale(arr, radius, "radius")
    if np.isscalar(radius):
        struct_size = tuple([int(radius) * 2 + 1] * arr.ndim)  # type: ignore[arg-type]
    else:
        struct_size = tuple(int(r) * 2 + 1 for r in radius)  # type: ignore[union-attr]

    baseline = _grey_opening_into(arr, struct_size)
    # In-place: `arr` is ours (either freshly copied by _validate, or explicitly
    # handed over with copy=False). `np.clip(arr - baseline, 0, None)` would cost
    # two extra full-volume temporaries -- on a 70 GB volume that is 140 GB of
    # avoidable allocation.
    np.subtract(arr, baseline, out=arr)
    np.clip(arr, 0, None, out=arr)
    if normalize:
        _normalize_inplace(arr, method=normalize_method)
    if not return_baseline:
        del baseline
        return arr, None
    return arr, baseline


def apply_baseline_subtraction(
    volume: np.ndarray,
    method: BaselineMethod = "tophat",
    percentile: float = 10,
    sigma: Scale = 20,
    tophat_radius: Scale = 20,
    normalize: bool = False,
    normalize_method: str = "percentile",
    return_baseline: bool = True,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray | float | None]:
    """Dispatch to a baseline estimator.

    ``normalize=False`` by default: output stays in the original intensity scale
    (merely offset), which is what makes corrected fluorescence quantifiable
    rather than merely segmentable.

    Args:
        volume: 2D image or 3D z-stack. Single-channel only.
        method: One of ``"none"``, ``"percentile"``, ``"smooth"``, ``"tophat"``.
        percentile: For ``"percentile"``.
        sigma: For ``"smooth"``.
        tophat_radius: For ``"tophat"``.
        normalize: Rescale output to ``[0, 1]``.
        normalize_method: ``"percentile"`` (safe) or ``"max"``.

    Returns:
        Tuple of (corrected, baseline_info) where ``baseline_info`` is ``None``
        for ``"none"``, a scalar for ``"percentile"``, and the full background
        array for ``"smooth"``/``"tophat"``.

    Raises:
        ValueError: On unknown ``method``, or non-2D/3D or non-finite input.
    """
    # NOTE: this dispatcher deliberately does NOT validate. It used to call
    # _validate() and then hand the result to a subtractor that validated it
    # AGAIN -- a second full-volume astype copy for nothing. Each subtractor now
    # validates exactly once.
    if method == "none":
        arr = _validate(volume, copy=copy)
        if normalize:
            _normalize_inplace(arr, method=normalize_method)
        return arr, None
    if method == "percentile":
        return subtract_percentile_baseline(
            volume, percentile=percentile, normalize=normalize,
            normalize_method=normalize_method, copy=copy,
        )
    if method == "smooth":
        return subtract_smooth_baseline(
            volume, sigma=sigma, normalize=normalize, normalize_method=normalize_method,
            return_baseline=return_baseline, copy=copy,
        )
    if method == "tophat":
        return subtract_tophat_baseline(
            volume, radius=tophat_radius, normalize=normalize,
            normalize_method=normalize_method, return_baseline=return_baseline, copy=copy,
        )
    raise ValueError(
        f"Unknown baseline subtraction method: {method!r}. "
        "Use 'none', 'percentile', 'smooth' or 'tophat'."
    )
