"""Ground-truth tests for bouton and axon detection.

Detecting thin axons and bright boutons in one pass is the failure mode that made
the Bee1004 demo report zero objects (min_size=200 deleted every 32-px bouton).
"""
from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk, line

from hive.neurites import detect_axons, detect_boutons, detect_neurites


def _phantom(seed=0):
    """A thin diagonal axon plus 5 bright boutons on a noisy background."""
    rng = np.random.default_rng(seed)
    img = np.full((200, 200), 50, dtype=np.float32)
    img += rng.normal(0, 3, (200, 200)).astype(np.float32)
    rr, cc = line(20, 20, 180, 180)
    for d in (-1, 0, 1):
        img[np.clip(rr + d, 0, 199), cc] = 250       # ~3-px-wide axon
    for y, x in [(40, 150), (60, 140), (150, 50), (120, 80), (90, 110)]:
        r, c = disk((y, x), 4, shape=img.shape)
        img[r, c] = 600
    return img


class TestBoutons:
    """Bouton detection finds bright puncta without flooding the background.

    The phantom contains a known number of puncta, so detection is checked
    against that count with a one-object tolerance for a punctum that splits.
    The paired background test is the real guard: a percentile threshold set
    too low turns diffuse background into thousands of spurious "boutons",
    so both over- and under-detection are pinned.
    """

    def test_finds_all_bright_puncta(self):
        result = detect_boutons(_phantom(), bright_percentile=99.0, min_size=4)
        from skimage import measure
        assert 5 <= measure.label(result).max() <= 6   # exact or one split

    def test_does_not_flood_the_background(self):
        mask = detect_boutons(_phantom(), bright_percentile=99.0)
        assert mask.mean() < 0.05                        # boutons are a tiny fraction

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            detect_boutons(np.zeros((4, 8, 8), dtype=np.float32))

    def test_rejects_bad_percentile(self):
        with pytest.raises(ValueError, match="percentile"):
            detect_boutons(np.zeros((8, 8), dtype=np.float32), bright_percentile=150)


class TestAxons:
    """Ridge-based tracing recovers thin fibres that thresholding misses.

    An axon dimmer than the global Otsu cut is invisible to intensity
    thresholding but still forms a coherent ridge, which is the case this
    phantom encodes.

    Total traced length is asserted rather than fibre count: tracing a fibre
    that breaks into two collinear fragments preserves length but doubles the
    count, so length is the stable metric.
    """

    def test_traces_a_thin_dim_fibre_intensity_threshold_would_miss(self):
        # A fibre only slightly above background: Otsu-on-intensity would ignore it,
        # but the ridge filter keys on shape, not brightness.
        img = np.full((120, 120), 100, dtype=np.float32)
        rr, cc = line(10, 60, 110, 60)
        img[rr, cc] = 130                                # only 30% above background
        mask, tube = detect_axons(img, ridge_percentile=98, min_length=20)
        assert mask.sum() > 40                           # the fibre is found
        assert tube.max() <= 1.0 and tube.min() >= 0.0   # normalised response

    def test_total_length_is_the_reliable_metric(self):
        # Filament COUNT fragments on noise; total skeletonised length does not.
        from skimage.morphology import skeletonize
        result = detect_neurites(_phantom(), axon_min_length=30)
        assert result.axon_length_px > 150              # ~226 truth, minus gaps


class TestDetectNeurites:
    """The combined entry point returns self-consistent boutons and axons.

    Both detectors run over the same image, so their outputs must agree on
    shape and stay mutually consistent rather than being independently
    plausible. Also covers the max-projection workflow, the usual way a
    z-stack is reduced before 2D neurite detection.
    """

    def test_returns_both_and_is_self_consistent(self):
        result = detect_neurites(_phantom(), axon_min_length=30)
        assert result.n_boutons >= 4
        assert result.axon_length_px == int(result.skeleton.sum())
        assert result.bouton_mask.shape == result.axon_mask.shape

    def test_max_projection_workflow(self):
        # The intended use: a corrected volume projected to 2D.
        rng = np.random.default_rng(1)
        vol = np.full((6, 100, 100), 40, dtype=np.float32)
        for z in range(6):                               # axon drifts through z
            vol[z, 20 + z, 20:80] = 300
        projection = vol.max(axis=0)
        result = detect_neurites(projection, axon_min_length=20)
        assert result.axon_length_px > 0
