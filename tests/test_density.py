"""Check density regions against exact one-dimensional masses."""

import numpy as np
import pytest

from yemale import ot


def test_smooth_density_region_matches_the_two_site_formula():
    smooth = ot.fit([[0.0]]).smooth(0.4)
    points = np.array([[-0.7], [0.0], [0.9], [30.0]])
    density = 0.3125 / np.cosh(points[:, 0] / 0.8) ** 2
    np.testing.assert_allclose(smooth.density(points), density, rtol=1e-12)
    np.testing.assert_allclose(smooth.log_density(points), np.log(density))

    source = np.array([[-1.0], [1.0]])
    tiny = ot.fit(source * 1e-308).smooth()
    np.testing.assert_allclose(
        tiny.log_density([1e-308]),
        ot.fit(source).smooth().log_density([1.0]) - np.log(1e-308),
    )

    region = smooth.density_region(mass=0.4)
    assert region.threshold == pytest.approx(0.1125, abs=0.002)
    assert region.mass == pytest.approx(0.4, abs=0.004)
    halfwidth = 0.8 * np.arctanh(0.8)
    np.testing.assert_array_equal(
        region.contains([0, 0.98 * halfwidth, 1.02 * halfwidth]),
        [True, True, False],
    )
    with pytest.raises(ValueError, match=r"total mass.*0\.5"):
        smooth.density_region(mass=0.6)


def test_reference_and_weighted_regions_include_entire_density_ties():
    reference = ot.reference(3, 1)
    region = reference.density_region(mass=0.5)
    assert region.threshold == pytest.approx(0.5)
    assert region.mass == pytest.approx(1.0)
    np.testing.assert_array_equal(region.contains([-1, 0, 1, 2]), [True] * 3 + [False])

    law = ot.Law(reference, cells=[1, 3], weights=[0.25, 0.75])
    region = law.density_region(mass=0.5)
    assert region.threshold == pytest.approx(1.5)
    assert region.mass == pytest.approx(0.75)
    np.testing.assert_array_equal(
        region.contains(reference.centers), [False, False, False, True]
    )


def test_two_dimensional_regions_keep_the_missing_reference_mass():
    smooth = ot.fit(np.random.default_rng(0).normal(size=(99, 2))).smooth()
    with pytest.raises(ValueError, match=r"total mass.*0\.90370"):
        smooth.density_region(mass=0.95)
    region = smooth.density_region(mass=0.5, n_integration_points=16)
    assert 0.5 <= region.mass <= 0.903705
    points = np.array([[[0.0, 0.0], [0.2, 0.1]], [[10.0, 10.0], [-10.0, -10.0]]])
    np.testing.assert_array_equal(
        region.contains(points), smooth.density(points) >= region.threshold
    )
