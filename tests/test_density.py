"""Check forward densities and density regions against exact formulas."""

import numpy as np
import pytest

from yemale import ot


def test_smooth_forward_density_matches_the_two_site_formula():
    smooth = ot.fit([[0.0]]).smooth(0.4)
    points = np.array([[-0.7], [0.0], [0.9], [30.0]])
    density = 0.3125 / np.cosh(points[:, 0] / 0.8) ** 2
    np.testing.assert_allclose(smooth.density(points), density, rtol=1e-12)
    np.testing.assert_allclose(smooth.log_density(points), np.log(density))

    source = np.array([[-1.0], [1.0]])
    tiny = ot.fit(source * 1e-308).smooth(0.4e-308)
    np.testing.assert_allclose(
        tiny.log_density([1e-308]),
        ot.fit(source).smooth(0.4).log_density([1.0]) - np.log(1e-308),
    )


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
