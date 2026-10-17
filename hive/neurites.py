"""Detection of thin neurites/axons and bright boutons.

Why this is separate from :mod:`hive.segmentation`
--------------------------------------------------
Otsu-on-intensity plus a size filter is the right tool for compact, well-separated
objects (cell bodies, nuclei). It is the WRONG tool for two structures common in
neuronal images, and both fail in the same run:

* **Boutons / varicosities** are tiny and very bright. A ``min_size`` tuned to
  reject debris deletes them too -- the exact failure seen on the Bee1004 crop,
  where 74 real boutons (median 32 px) were all dropped by ``min_size=200`` and
  the pipeline reported zero objects.
* **Axons** are thin (a few px wide) and dim. Their total area is small and their
  intensity barely clears a global threshold, so an intensity threshold either
  misses them or, lowered enough to catch them, floods the neuropil with noise.

Axons need a **ridge/tubeness filter**. Sato (Sato et al., 1998) and Frangi
(Frangi et al., 1998) analyse the Hessian eigenvalues at each pixel and respond to
locally tube-like structure regardless of absolute brightness -- the established
approach for vessels and neurites. HIVE uses Sato via scikit-image.

Depth note: a single confocal plane clips an axon that travels through z. Detect on
a **maximum-intensity projection** of the background-corrected volume so the tract
is continuous. :func:`detect_neurites` accepts a 2D image; project first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage import filters, measure, morphology


@dataclass
class NeuriteResult:
    """Detected boutons and axon filaments from one image.

    Attributes:
        bouton_mask: Boolean mask of bright punctate structures.
        axon_mask: Boolean mask of thin tube-like structures.
        skeleton: 1-px centreline of ``axon_mask``.
        n_boutons: Number of connected bouton components.
        n_axon_filaments: Number of connected axon components.
        axon_length_px: Total skeleton length in pixels (diagonal-aware count is
            available via :mod:`hive.arborization`; this is the raw pixel count).
        tubeness: The Sato ridge response, normalised to ``[0, 1]``, for display.
    """

    bouton_mask: np.ndarray = field(repr=False)
    axon_mask: np.ndarray = field(repr=False)
    skeleton: np.ndarray = field(repr=False)
    n_boutons: int = 0
    n_axon_filaments: int = 0
    axon_length_px: int = 0
    tubeness: np.ndarray = field(default=None, repr=False)


def detect_boutons(
    measurement: np.ndarray, bright_percentile: float = 99.0, min_size: int = 4
) -> np.ndarray:
    """Detect bright punctate structures (boutons / varicosities).

    Thresholds at a high intensity percentile rather than Otsu, because boutons
    are a small fraction of the pixels and a global Otsu split lands inside the
    neuropil, flooding it. A morphological opening removes single-pixel noise.

    Args:
        measurement: Background-corrected image (original intensity scale). A
            max-projection of the corrected volume works well.
        bright_percentile: Percentile defining "bright"; 99.0 keeps the top 1%.
        min_size: Objects smaller than this (px) are dropped as noise. Keep it
            small -- boutons are tiny.

    Returns:
        Boolean mask of detected boutons.

    Raises:
        ValueError: If ``measurement`` is not 2D or the percentile is invalid.
    """
    img = np.asarray(measurement, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"detect_boutons expects a 2D image, got shape {img.shape}")
    if not 0 < bright_percentile < 100:
        raise ValueError(f"bright_percentile must be in (0, 100), got {bright_percentile}")

    mask = img > float(np.percentile(img, bright_percentile))
    mask = morphology.remove_small_objects(mask, max(1, min_size))
    mask = morphology.binary_opening(mask)
    return morphology.remove_small_objects(mask, max(1, min_size))


def detect_axons(
    measurement: np.ndarray,
    scales: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    ridge_percentile: float = 96.0,
    min_length: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect thin tube-like structures (axons) with a Sato ridge filter.

    The Sato filter (Sato et al., 1998) responds to locally tubular structure by
    Hessian eigenanalysis, so it finds dim thin fibres an intensity threshold
    misses. ``scales`` are the tube half-widths in px to look for -- keep them
    small (1-3) for fine axons; add larger values for thicker tracts.

    Args:
        measurement: Background-corrected 2D image (project a volume first).
        scales: Sato sigmas (approx. tube radii in px).
        ridge_percentile: Percentile of the ridge response taken as "on".
        min_length: Filament components smaller than this (px) are dropped;
            genuine axons are long, so this removes blob-like false positives.

    Returns:
        Tuple of (axon mask, normalised tubeness response in ``[0, 1]``).

    Raises:
        ValueError: If ``measurement`` is not 2D.
    """
    img = np.asarray(measurement, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"detect_axons expects a 2D image, got shape {img.shape}")

    tube = filters.sato(img, sigmas=scales, black_ridges=False)
    tube_max = float(tube.max())
    tube_n = tube / tube_max if tube_max > 0 else tube

    mask = tube_n > float(np.percentile(tube_n, ridge_percentile))
    mask = morphology.remove_small_objects(mask, max(1, min_length))
    mask = morphology.binary_closing(mask, morphology.disk(1))
    return mask, tube_n.astype(np.float32)


def detect_neurites(
    measurement: np.ndarray,
    bright_percentile: float = 99.0,
    bouton_min_size: int = 4,
    axon_scales: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    ridge_percentile: float = 96.0,
    axon_min_length: int = 50,
) -> NeuriteResult:
    """Detect both boutons and axons in a background-corrected image.

    Args:
        measurement: Background-corrected 2D image. For a z-stack, pass a
            max-intensity projection of the corrected volume so axons travelling
            through z stay continuous.
        bright_percentile: See :func:`detect_boutons`.
        bouton_min_size: See :func:`detect_boutons`.
        axon_scales: See :func:`detect_axons`.
        ridge_percentile: See :func:`detect_axons`.
        axon_min_length: See :func:`detect_axons`.

    Returns:
        A :class:`NeuriteResult`.

    Raises:
        ValueError: If ``measurement`` is not 2D.
    """
    boutons = detect_boutons(measurement, bright_percentile, bouton_min_size)
    axons, tubeness = detect_axons(measurement, axon_scales, ridge_percentile, axon_min_length)
    skeleton = morphology.skeletonize(axons)

    return NeuriteResult(
        bouton_mask=boutons,
        axon_mask=axons,
        skeleton=skeleton,
        n_boutons=int(measure.label(boutons).max()),
        n_axon_filaments=int(measure.label(axons).max()),
        axon_length_px=int(skeleton.sum()),
        tubeness=tubeness,
    )
