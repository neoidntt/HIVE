"""CLAHE (Zuiderveld, 1994) as scikit-image computes it, with the slow step spread over threads.

:func:`equalize_adapthist` returns exactly what :func:`skimage.exposure.equalize_adapthist`
returns for a 2D image. The algorithm is scikit-image's own (its helper functions are used
as they are); the one change is that the final interpolation, which is independent from one
contextual block to the next and is most of the time, is cut into groups of blocks that are
computed on several threads. Each block gets the same arithmetic in the same order, so the
result is the same to the last bit, and the temporaries are one group's size instead of the
whole image's (scikit-image's version peaks near 18 times the image).

scikit-image's helpers are private. If they are missing, or this function ever disagrees
with scikit-image on a small test image (checked once, on first use), scikit-image's own
function is used instead, so an upgrade can slow things down but cannot change results.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage import exposure

_checked: bool | None = None          # None: not yet checked; True: use this module; False: don't


def _fast(image: np.ndarray, clip_limit: float, nbins: int, threads: int) -> np.ndarray:
    from skimage.exposure._adapthist import NR_OF_GRAY, clip_histogram, map_histogram
    from skimage.util import img_as_uint
    from skimage.exposure import rescale_intensity
    from skimage._shared.utils import _supported_float_type

    float_dtype = _supported_float_type(image.dtype)
    image = img_as_uint(image)
    image = np.round(rescale_intensity(image, out_range=(0, NR_OF_GRAY - 1))).astype(
        np.min_scalar_type(NR_OF_GRAY))
    kernel_size = [max(s // 8, 1) for s in image.shape]
    dtype = image.dtype

    pad_start = [k // 2 for k in kernel_size]
    pad_end = [(k - s % k) % k + int(np.ceil(k / 2.0)) for k, s in zip(kernel_size, image.shape)]
    image = np.pad(image, [[a, b] for a, b in zip(pad_start, pad_end)], mode="reflect")

    bin_size = 1 + NR_OF_GRAY // nbins
    lut = np.arange(NR_OF_GRAY, dtype=np.min_scalar_type(NR_OF_GRAY))
    lut //= bin_size
    image = lut[image]

    ns_hist = [int(s / k) - 1 for s, k in zip(image.shape, kernel_size)]
    hist_blocks_shape = np.array([ns_hist, kernel_size]).T.flatten()
    axis_order = np.array([np.arange(0, 4, 2), np.arange(1, 4, 2)]).flatten()
    hist_slices = [slice(k // 2, k // 2 + n * k) for k, n in zip(kernel_size, ns_hist)]
    hist_blocks = image[tuple(hist_slices)].reshape(hist_blocks_shape)
    hist_blocks = np.transpose(hist_blocks, axes=axis_order)
    assembled_shape = hist_blocks.shape
    hist_blocks = hist_blocks.reshape((math.prod(ns_hist), -1))

    kernel_elements = math.prod(kernel_size)
    clim = int(np.clip(clip_limit * kernel_elements, 1, None)) if clip_limit > 0.0 else kernel_elements
    hist = np.apply_along_axis(np.bincount, -1, hist_blocks, minlength=nbins)
    hist = np.apply_along_axis(clip_histogram, -1, hist, clip_limit=clim)
    hist = map_histogram(hist, 0, NR_OF_GRAY - 1, kernel_elements)
    hist = hist.reshape(assembled_shape[:2] + (-1,))
    map_array = np.pad(hist, [[1, 1], [1, 1], [0, 0]], mode="edge")

    ns_proc = [int(s / k) for s, k in zip(image.shape, kernel_size)]
    blocks_shape = np.array([ns_proc, kernel_size]).T.flatten()
    blocks = np.transpose(image.reshape(blocks_shape), axes=axis_order)
    flattened_shape = blocks.shape
    blocks = np.reshape(blocks, (math.prod(ns_proc), math.prod(blocks.shape[2:])))

    coeffs = np.meshgrid(*[np.arange(k) / k for k in kernel_size[::-1]], indexing="ij")
    coeffs = [np.transpose(c).flatten() for c in coeffs]
    inv_coeffs = [1 - c for c in coeffs]
    edges = list(np.ndindex(2, 2))
    edge_coeffs = [np.prod([[inv_coeffs, coeffs][e][d] for d, e in enumerate(edge[::-1])], 0)
                   for edge in edges]
    edge_maps = [map_array[e[0]:e[0] + ns_proc[0], e[1]:e[1] + ns_proc[1]].reshape(
        (math.prod(ns_proc), -1)) for e in edges]

    result = np.zeros(blocks.shape, dtype=np.float32)

    def interpolate(rows: slice) -> None:
        part = result[rows]
        for maps, weights in zip(edge_maps, edge_coeffs):
            mapped = np.take_along_axis(maps[rows], blocks[rows], axis=-1)
            part += (mapped * weights).astype(part.dtype)

    n_blocks = blocks.shape[0]
    step = max(1, min(n_blocks, -(-n_blocks // (threads * 4))))   # a few groups per thread
    groups = [slice(i, min(i + step, n_blocks)) for i in range(0, n_blocks, step)]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(interpolate, groups))

    result = result.astype(dtype)
    result = result.reshape(flattened_shape)
    result = np.transpose(result, axes=np.array([np.arange(0, 2), np.arange(2, 4)]).T.flatten())
    result = result.reshape(image.shape)
    result = result[tuple(slice(a, s - b) for a, b, s in zip(pad_start, pad_end, image.shape))]
    return rescale_intensity(result.astype(float_dtype, copy=False))


def _agrees_with_scikit_image() -> bool:
    """Once per process: does this module reproduce scikit-image on a small test image?"""
    rng = np.random.default_rng(0)
    test = rng.random((83, 97)).astype(np.float32)
    test[20:50, 30:60] += 0.5
    test /= test.max()
    try:
        return bool(np.array_equal(_fast(test, 0.01, 256, 2), exposure.equalize_adapthist(test)))
    except Exception:  # noqa: BLE001 - a private helper moved or changed: use scikit-image
        return False


def equalize_adapthist(image: np.ndarray, clip_limit: float = 0.01, nbins: int = 256,
                       threads: int | None = None) -> np.ndarray:
    """``skimage.exposure.equalize_adapthist(image, clip_limit=..., nbins=...)`` for a 2D image.

    Returns float64 like scikit-image does (float32 inputs give float32 on newer versions;
    whatever scikit-image returns for the dtype, this returns).
    """
    global _checked
    if image.ndim != 2 or min(image.shape) < 16:
        return exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
    if _checked is None:
        _checked = _agrees_with_scikit_image()
    if not _checked:
        return exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
    return _fast(image, clip_limit, nbins, threads or min(os.cpu_count() or 1, 4))
