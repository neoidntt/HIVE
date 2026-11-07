"""Background preview: shows the methods, changes nothing."""

from __future__ import annotations

import numpy as np
import pytest

from hive.baseline_preview import METHODS, center_crop, preview_figure, preview_methods
from hive.config import PreprocessConfig
from hive.synthetic import make_synthetic_2d


def _image():
    return make_synthetic_2d(0)[1][1]


def test_every_method_is_previewed_and_the_input_is_untouched():
    image = _image()
    before = image.copy()
    previews = preview_methods(image, PreprocessConfig(tophat_radius=15.0))
    assert [p.method for p in previews] == list(METHODS)
    assert np.array_equal(image, before)


def test_none_changes_nothing_and_the_others_remove_background():
    by_method = {p.method: p for p in preview_methods(_image(), PreprocessConfig(tophat_radius=15.0))}
    assert by_method["none"].removed_fraction == 0 and by_method["none"].background is None
    assert np.array_equal(by_method["none"].corrected, center_crop(_image()))
    for method in ("percentile", "smooth", "tophat"):
        assert by_method[method].removed_fraction > 0
        assert by_method[method].corrected.mean() < by_method["none"].corrected.mean()


def test_too_small_a_radius_is_visible_as_signal_at_zero():
    image = np.full((80, 80), 20.0, np.float32)
    image[20:60, 20:60] += 300                       # a 40-px object
    small = {p.method: p for p in preview_methods(image, PreprocessConfig(tophat_radius=3.0))}
    large = {p.method: p for p in preview_methods(image, PreprocessConfig(tophat_radius=40.0))}
    assert small["tophat"].corrected.max() < large["tophat"].corrected.max()


def test_preview_works_on_a_crop_of_a_large_image():
    big = np.random.default_rng(0).uniform(10, 20, (900, 1100)).astype(np.float32)
    previews = preview_methods(big, PreprocessConfig(), max_side=256)
    assert previews[0].corrected.shape == (256, 256)


def test_per_axis_radii_use_the_in_plane_value():
    preview_methods(_image(), PreprocessConfig(tophat_radius=(2.0, 12.0, 12.0),
                                               baseline_sigma=(2.0, 8.0, 8.0)))


def test_the_figure_has_a_row_per_method():
    image = _image()
    fig = preview_figure(center_crop(image), preview_methods(image, PreprocessConfig(tophat_radius=15.0)))
    assert len(fig.axes) == 3 * len(METHODS)


def test_only_a_single_plane_is_accepted():
    with pytest.raises(ValueError, match="2D"):
        preview_methods(np.zeros((3, 8, 8)), PreprocessConfig())
