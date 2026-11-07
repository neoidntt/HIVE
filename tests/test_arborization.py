"""Ground-truth tests for arborisation analysis.

Every phantom here has an analytically known answer, so a failure means the
measurement is wrong rather than merely different. The regression tests at the
bottom pin behaviours that were genuinely broken during development -- each one
failed at least once before the implementation was corrected.
"""

from __future__ import annotations

import numpy as np
import pytest

from hive.arborization import (
    analyze_arbor, crossing_number, find_branch_points, find_endpoints,
    prune_skeleton, sholl_analysis, skeleton_length, skeleton_metrics, skeletonize_mask,
)
from hive.config import ArborConfig

SIZE = 121
CENTRE = SIZE // 2


@pytest.fixture
def star() -> np.ndarray:
    """A 4-arm star: 4 arms of 40 px, one central junction, 4 tips."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[CENTRE - 1:CENTRE + 2, CENTRE - 40:CENTRE + 41] = True
    mask[CENTRE - 40:CENTRE + 41, CENTRE - 1:CENTRE + 2] = True
    return mask


@pytest.fixture
def star_with_spurs(star: np.ndarray) -> np.ndarray:
    """The 4-arm star plus two 3-px false twigs, as thinning artifacts."""
    mask = star.copy()
    mask[CENTRE - 4:CENTRE - 1, CENTRE - 20] = True
    mask[CENTRE + 2:CENTRE + 5, CENTRE + 15] = True
    return mask


class TestTopology:
    """Branch topology against phantoms with analytically known answers.

    A 4-arm star *must* report exactly 4 endpoints, 1 branch point and 4
    branches; a Y-tree must report one bifurcation. Because the right answer is
    known from the geometry rather than from a previous run, a failure here
    means the measurement is wrong, not merely different.
    """

    def test_star_has_four_tips_and_one_junction(self, star):
        metrics = skeleton_metrics(star, prune_length=3)
        assert metrics.n_endpoints == 4
        assert metrics.n_branch_points == 1
        assert metrics.n_branches == 4

    def test_star_total_length_matches_geometry(self, star):
        # 4 arms x 40 px = 160; thinning tolerance a few px.
        metrics = skeleton_metrics(star, prune_length=3)
        assert 150 < metrics.total_length < 172

    def test_ytree_bifurcation(self):
        tree = np.zeros((SIZE, SIZE), dtype=bool)
        tree[60:100, 59:62] = True
        for i in range(30):
            tree[60 - i, 59 - i:62 - i] = True
            tree[60 - i, 59 + i:62 + i] = True
        metrics = skeleton_metrics(tree, prune_length=3)
        assert metrics.n_branch_points == 1
        assert metrics.n_endpoints == 3  # trunk base + 2 tips

    def test_3d_six_arm_star(self):
        vol = np.zeros((61, 61, 61), dtype=bool)
        vol[30, 30, 10:51] = True
        vol[30, 10:51, 30] = True
        vol[10:51, 30, 30] = True
        metrics = skeleton_metrics(vol, prune_length=0)
        assert metrics.n_endpoints == 6
        assert metrics.n_branch_points == 1


class TestCrossingNumber:
    """Rutovitz crossing number classifies skeleton pixels correctly.

    Naive 8-neighbour counting misclassifies a single-pixel bump on a line as a
    junction, because that pixel has three neighbours. The crossing number
    counts 0->1 transitions around the ring instead and correctly calls it a
    tip. These tests pin that distinction, plus the invariant that a pixel in
    the middle of a path has crossing number 2. 2D only, as in Fiji's
    AnalyzeSkeleton (Arganda-Carreras et al., 2010).
    """

    def test_bump_on_a_line_is_a_tip_not_a_junction(self):
        """A pixel resting on a straight line touches 3 line pixels, so raw
        neighbour-counting calls it a junction. It is not: no branch departs.
        This misclassification made spur bases immune to pruning."""
        skel = np.zeros((9, 9), dtype=bool)
        skel[5, 1:8] = True  # a horizontal line
        skel[4, 4] = True    # one pixel resting on top of it
        assert crossing_number(skel)[4, 4] == 1       # a tip
        assert find_endpoints(skel)[4, 4]
        assert not find_branch_points(skel)[4, 4]

    def test_true_t_junction_is_a_junction(self):
        skel = np.zeros((9, 9), dtype=bool)
        skel[5, 1:8] = True
        skel[1:5, 4] = True  # a stem long enough to be a real branch
        assert crossing_number(skel)[5, 4] >= 3
        assert find_branch_points(skel)[5, 4]

    def test_path_pixel_has_crossing_number_two(self):
        skel = np.zeros((9, 9), dtype=bool)
        skel[5, 1:8] = True
        assert crossing_number(skel)[5, 4] == 2

    def test_is_2d_only(self):
        with pytest.raises(ValueError, match="2D-only"):
            crossing_number(np.zeros((4, 4, 4), dtype=bool))


class TestSkeletonLength:
    """Skeleton length is measured geometrically, not by counting pixels.

    A diagonal step spans sqrt(2), not 1. Counting pixels therefore
    underestimates any diagonal neurite by ~41%, which is why these tests pin
    the exact value for both orthogonal and diagonal lines.

    Anisotropic spacing is checked separately: confocal z-steps are routinely
    3-10x the xy pixel size, so applying the wrong per-axis scale silently
    distorts every reported length.
    """

    def test_orthogonal_line_length_is_exact(self):
        skel = np.zeros((50, 50), dtype=bool)
        skel[25, 10:41] = True  # 31 px -> 30 steps
        assert skeleton_length(skel) == pytest.approx(30.0)

    def test_diagonal_uses_sqrt2_not_pixel_count(self):
        """Counting pixels underestimates a diagonal by ~29%."""
        skel = np.zeros((50, 50), dtype=bool)
        for i in range(30):
            skel[10 + i, 10 + i] = True  # 30 px -> 29 steps
        assert skeleton_length(skel) == pytest.approx(29 * np.sqrt(2))
        assert skel.sum() == 30  # the naive answer, ~27% low

    def test_scalar_spacing_converts_to_physical_units(self):
        skel = np.zeros((50, 50), dtype=bool)
        skel[25, 10:41] = True
        assert skeleton_length(skel, spacing=0.5) == pytest.approx(15.0)

    def test_anisotropic_spacing_uses_the_right_axis(self):
        vol = np.zeros((20, 40, 40), dtype=bool)
        vol[5:15, 20, 20] = True  # 10 voxels along z -> 9 steps
        assert skeleton_length(vol, spacing=(2.0, 0.5, 0.5)) == pytest.approx(18.0)


class TestPruning:
    """Spur pruning restores ground truth without eroding real branches.

    Skeletonising a noisy mask sprouts short false twigs, and every twig adds a
    spurious endpoint and branch point. The paired tests here show the failure
    and the fix: an unpruned spurred star reports inflated counts, while the
    pruned one recovers the exact 4 tips / 1 junction.

    The counterweight tests matter just as much -- pruning must not shorten a
    genuine branch, and ``prune_length=0`` must be a true no-op.
    """

    def test_removes_spurs_and_recovers_ground_truth(self, star_with_spurs):
        """Regression: this returned 3 junctions and 6 tips for a 1-junction,
        4-tip star. Two earlier designs failed here -- segment-based pruning
        left the spur's base behind because that base belonged to the junction
        cut set and so was in no prunable segment."""
        metrics = skeleton_metrics(star_with_spurs, prune_length=5)
        assert metrics.pruned_spurs == 2
        assert metrics.n_endpoints == 4
        assert metrics.n_branch_points == 1

    def test_unpruned_spurs_inflate_counts(self, star_with_spurs):
        metrics = skeleton_metrics(star_with_spurs, prune_length=0)
        assert metrics.n_endpoints == 6
        assert metrics.n_branch_points == 3

    def test_does_not_shorten_real_branches(self, star):
        """Endpoint erosion would eat real neurites; the walk must not."""
        unpruned = skeleton_metrics(star, prune_length=0)
        pruned = skeleton_metrics(star, prune_length=5)
        assert pruned.total_length == pytest.approx(unpruned.total_length)
        assert pruned.pruned_spurs == 0

    def test_prune_length_zero_is_a_noop(self, star_with_spurs):
        skel = skeletonize_mask(star_with_spurs)
        out, removed = prune_skeleton(skel, min_length=0)
        assert removed == 0
        assert np.array_equal(out, skel)


class TestSholl:
    """Sholl analysis (Sholl, 1953) against known arbor geometry.

    A 4-arm star crosses every concentric shell exactly 4 times until the
    shells outgrow the arms, so both the intersection profile and the
    enclosing radius are known in advance. 3D uses spherical shells rather
    than circles.

    The semi-log regression coefficient (Ferreira et al., 2014) must return
    NaN when too few shells are available to fit -- reporting a slope from two
    points would be a fabricated number.
    """

    def test_star_gives_four_intersections_at_every_shell(self, star):
        metrics = skeleton_metrics(star, prune_length=3)
        result = sholl_analysis(metrics.skeleton, (CENTRE, CENTRE), step=5.0)
        assert result.n_primary == 4
        assert result.max_intersections == 4
        assert result.ramification_index == pytest.approx(1.0)
        assert np.all(result.intersections[:7] == 4)

    def test_enclosing_radius_matches_arm_length(self, star):
        metrics = skeleton_metrics(star, prune_length=3)
        result = sholl_analysis(metrics.skeleton, (CENTRE, CENTRE), step=5.0)
        assert 35 <= result.enclosing_radius <= 42  # arms are 40 px

    def test_ramification_index_detects_branching_gain(self):
        """A trunk that forks into 2: 1 primary, max 2 -> index 2.0."""
        tree = np.zeros((SIZE, SIZE), dtype=bool)
        tree[60:100, 59:62] = True
        for i in range(30):
            tree[60 - i, 59 - i:62 - i] = True
            tree[60 - i, 59 + i:62 + i] = True
        metrics = skeleton_metrics(tree, prune_length=3)
        result = sholl_analysis(metrics.skeleton, (100, 60), step=5.0)
        assert result.n_primary == 1
        assert result.max_intersections == 2
        assert result.ramification_index == pytest.approx(2.0)

    def test_3d_uses_spherical_shells(self):
        vol = np.zeros((61, 61, 61), dtype=bool)
        vol[30, 30, 10:51] = True
        vol[30, 10:51, 30] = True
        vol[10:51, 30, 30] = True
        metrics = skeleton_metrics(vol, prune_length=0)
        result = sholl_analysis(metrics.skeleton, (30, 30, 30), step=5.0)
        assert result.n_primary == 6

    def test_regression_coefficient_is_nan_with_too_few_shells(self):
        skel = np.zeros((20, 20), dtype=bool)
        skel[10, 9:12] = True
        result = sholl_analysis(skel, (10, 10), step=5.0)
        assert np.isnan(result.regression_coefficient)

    def test_rejects_mismatched_centre_dimensionality(self, star):
        with pytest.raises(ValueError, match="coordinates"):
            sholl_analysis(skeletonize_mask(star), (1, 2, 3), step=5.0)

    def test_rejects_non_positive_step(self, star):
        with pytest.raises(ValueError, match="step must be positive"):
            sholl_analysis(skeletonize_mask(star), (CENTRE, CENTRE), step=0)


class TestDegenerateInputs:
    """Degenerate masks return empty results instead of crashing.

    Empty masks, single pixels and skeletons with no branches occur routinely
    in real batches -- a debris speck that survives filtering is a one-pixel
    "arbor". These must produce empty or trivial output so one bad object
    cannot abort a run. Genuinely unsupported input (4D) still raises.
    """

    @pytest.mark.parametrize("mask", [
        np.zeros((50, 50), dtype=bool),
        np.ones((30, 30), dtype=bool),
    ], ids=["empty", "full_frame"])
    def test_does_not_crash(self, mask):
        metrics = skeleton_metrics(mask, prune_length=3)
        assert metrics.total_length >= 0

    def test_single_pixel(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True
        metrics = skeleton_metrics(mask, prune_length=3)
        assert metrics.total_length == 0.0

    def test_empty_skeleton_sholl_is_empty(self):
        result = sholl_analysis(np.zeros((20, 20), dtype=bool), (10, 10))
        assert result.radii.size == 0
        assert result.max_intersections == 0

    def test_rejects_4d(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            skeletonize_mask(np.zeros((2, 2, 2, 2), dtype=bool))


class TestAnalyzeArbor:
    """The public entry point emits the full documented column set.

    Downstream reporting and the README's metric list both depend on these
    exact column names, so a rename would silently break exports. Zero objects
    must still yield a correctly-shaped empty frame.
    """

    def test_produces_expected_columns(self, star):
        labels = star.astype(np.int32)
        df, curves = analyze_arbor(labels, ArborConfig(sholl_step=5.0))
        for column in ("neurite_length_px", "n_branch_points", "n_endpoints",
                       "sholl_ramification_index", "sholl_regression_coeff_k"):
            assert column in df.columns
        assert len(df) == 1
        assert curves[1].n_primary == 4

    def test_empty_labels_give_empty_frame(self):
        df, curves = analyze_arbor(np.zeros((20, 20), dtype=np.int32), ArborConfig())
        assert df.empty
        assert curves == {}
