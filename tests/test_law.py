"""Check sampling, density, and averages for reference and mapped distributions."""

import math

import numpy as np
import pytest

from yemale import ot


@pytest.mark.parametrize("dimension", [1, 2, 5])
def test_reference_law(dimension):
    reference = ot.reference(39, dimension)
    labels = np.repeat(np.arange(reference.n + 1), 2048)
    sample = reference._sample(labels, np.random.default_rng(12))
    assert np.array_equal(reference.locate(sample), labels)
    np.testing.assert_allclose(
        sample.reshape(reference.n + 1, -1, dimension).mean(axis=1),
        reference.centers,
        atol=0.025,
    )
    np.testing.assert_allclose(
        (sample * sample).mean(axis=0), 1 / (3 * dimension), atol=0.004
    )
    np.testing.assert_allclose(
        reference.expect(lambda u: u * u, n_integration_points=256),
        1 / (3 * dimension),
        atol=0.003,
    )
    log_area = np.log(2 * np.pi ** (dimension / 2) / math.gamma(dimension / 2))
    np.testing.assert_allclose(
        reference.logpdf(sample),
        -log_area - (dimension - 1) * np.log(np.linalg.norm(sample, axis=1)),
    )
    assert np.isneginf(reference.logpdf(np.full(dimension, 2.0)))
    np.testing.assert_allclose(reference.mean(), 0, atol=1e-15)
    np.testing.assert_allclose(
        reference.cov(), np.eye(dimension) / (3 * dimension), atol=1e-15
    )
    order = (2,) + (0,) * (dimension - 1)
    np.testing.assert_allclose(reference.moment(powers=order), 1 / (3 * dimension))
    np.testing.assert_allclose(reference.entropy(), log_area - dimension + 1)


def test_reference_lookup():
    reference = ot.reference(7, 2)
    # Two shells meet at radius 1/2. Each has four counterclockwise quadrants:
    # labels 0..3 in the inner shell and 4..7 in the outer shell.
    # A shared radial or angular boundary belongs to the smallest label.
    points = np.array(
        [
            [0, 0],
            [0.5, 0],
            [0, 0.5],
            [-0.5, 0],
            [0, -0.5],
            [0.75, 0],
            [0, 0.75],
            [-0.75, 0],
            [0, -0.75],
            [2, 0],
            [np.nan, 0],
            [np.inf, 0],
        ]
    ).reshape(3, 4, 2)
    np.testing.assert_array_equal(
        reference.locate(points), [[0, 0, 0, 1], [2, 4, 4, 5], [6, -1, -1, -1]]
    )
    assert reference.locate([0, 0]) == 0
    assert reference.locate(np.empty((0, 2))).shape == (0,)


def test_reference_extreme_radii():
    reference = ot.reference(1, 2)
    with np.errstate(over="raise", invalid="raise", under="raise"):
        assert reference.locate([0.0, -1e-200]) == 1
        expected = -np.log(2 * np.pi) - np.log(1e-200)
        np.testing.assert_allclose(reference.logpdf([0.0, -1e-200]), expected)
        np.testing.assert_allclose(
            ot.Law(reference, cells=[1]).logpdf([0.0, -1e-200]), expected + np.log(2)
        )
        assert np.isneginf(reference.logpdf([1e308, 0.0]))


def test_known_moments():
    reference = ot.reference(1, 10)
    law = ot.Law(reference)
    np.testing.assert_allclose(law.cov(), reference.cov(), rtol=1e-13, atol=1e-14)
    # A high-order moment exercises the stable Beta integrals. Its small value
    # must be accurate relatively; a nonzero absolute tolerance could hide loss.
    powers = (0, 34) + (0,) * 8
    np.testing.assert_allclose(
        law.moment(powers), reference.moment(powers), rtol=1e-11, atol=0
    )
    cell = ot.Law(ot.reference(1, 3), cells=[0])
    # On this hemisphere, T = Theta[0] is uniform on [-1, 0].
    # Independence gives E[(R*T)**40] = (1/41) * (1/41).
    np.testing.assert_allclose(cell.moment(powers=(40, 0, 0)), 1 / 41**2)
    # E[R**4] = 1/5 and E[Theta[0]**2 * Theta[1]**2] = 1/15.
    np.testing.assert_allclose(ot.reference(7, 3).moment(powers=(2, 2, 0)), 1 / 75)


def test_predictive_uniform_gaps():
    transport = ot.fit([[0.0]])

    def shift(labels):
        return np.where(labels == 0, -1.0, 1.0)[:, None]

    def map_from_reference(points, labels):
        return 2 * points + shift(labels)

    # Equal mass is uniform on [-3, -1] and [1, 3]: density 1/4,
    # second moment 13/3, and differential entropy log(4).
    law = transport.predictive_distribution(
        map_from_reference=map_from_reference,
        inverse=lambda z, labels: (z - shift(labels)) / 2,
        inverse_logabsdet=lambda z, labels: np.full_like(z, -np.log(2.0)),
    )
    direct = ot.Law(transport.reference, forward=map_from_reference)
    np.testing.assert_array_equal(direct.sample(8, rng=7), law.sample(8, rng=7))
    np.testing.assert_allclose(law.pdf([-2, -0.5, 0.5, 2]), [0.25, 0, 0, 0.25])
    sample = law.sample(10000, rng=7)
    assert np.all((np.abs(sample) >= 1) & (np.abs(sample) <= 3))
    np.testing.assert_allclose(np.exp(law.logpdf(sample)), 0.25)
    assert (sample < 0).mean() == pytest.approx(0.5, abs=0.02)
    np.testing.assert_allclose(law.mean(), [0], atol=1e-15)
    np.testing.assert_allclose(
        law.moment(powers=2, n_integration_points=512), 13 / 3, atol=2e-6
    )
    np.testing.assert_allclose(law.cov(n_integration_points=512), [[13 / 3]], atol=2e-6)
    np.testing.assert_allclose(law.entropy(), np.log(4))
    with pytest.raises(TypeError):
        direct.logpdf(0)
    with pytest.raises(TypeError):
        direct.entropy()


def test_inverse_jacobians_compose():
    law = ot.Law(
        ot.reference(1, 1),
        forward=lambda u, labels: 2 * u,
        backward=lambda z: (z / 2, np.full_like(z, -np.log(2))),
    )
    region = law.density_region(0.5, n_integration_points=8)
    mapped = law._map(
        lambda z: 3 * z,
        lambda z: (z / 3, np.full(len(z), -np.log(3))),
    )
    assert mapped.cells is law.cells
    assert mapped.weights is law.weights
    assert mapped.density_region(
        0.5, n_integration_points=8
    ).threshold == pytest.approx(region.threshold / 3)
    points = np.array([[-1.0], [0.0], [1.0]])
    np.testing.assert_allclose(np.exp(law.logpdf(points)), 0.25)
    np.testing.assert_allclose(np.exp(mapped.logpdf(points)), 1 / 12)
    np.testing.assert_allclose(mapped.sample(8, rng=7), 3 * law.sample(8, rng=7))
    bad = law._map(lambda z: z, lambda z: (z, np.zeros(2)))
    with pytest.raises(ValueError):
        bad.logpdf(points)


def test_weighted_mixture():
    reference = ot.reference(3, 1)
    weights = np.array([0.25, 0.75])
    law = ot.Law(reference, cells=[1, 3], weights=weights)
    weights[:] = 0  # The law must own its probability data.
    np.testing.assert_allclose(np.exp(law.logpdf(reference.centers)), [0, 0.5, 0, 1.5])
    labels = reference.locate(law.sample(10000, rng=3))
    assert np.isin(labels, [1, 3]).all()
    assert (labels == 3).mean() == pytest.approx(0.75, abs=0.02)
    np.testing.assert_allclose(law.mean(), [0.5])
    assert law.expect(lambda z: np.full(len(z), 7)) == 7
    assert law.expect(
        lambda z: z[:, 0] > 0,
        n_integration_points=8,
        rng=np.random.default_rng(1),
    ) == pytest.approx(0.75)


def test_invalid_law_inputs():
    reference = ot.reference(3, 1)
    with pytest.raises(TypeError):
        ot.Law(reference, forward=0)
    with pytest.raises(ValueError):
        ot.Law(reference, cells=[0, 0, 1])
    for invalid in ([1.0], [1, -1], [0.2, 0.2]):
        with pytest.raises(ValueError):
            ot.Law(reference, cells=[1, 3], weights=invalid)
    for invalid in ((1, 2), (1.5, 0, 0)):
        with pytest.raises(ValueError):
            ot.reference(1, 3).moment(powers=invalid)


def test_expect_shapes():
    reference = ot.reference(7, 2)
    for distribution in (reference, ot.Law(reference)):
        for function in (np.linalg.norm, lambda x: np.ones(3)):
            with pytest.raises(ValueError):
                distribution.expect(function)
        np.testing.assert_allclose(
            distribution.expect(lambda x: np.linalg.norm(x, axis=-1)), 0.5, atol=0.005
        )
        np.testing.assert_allclose(
            distribution.expect(lambda x: x * x),
            distribution.cov().diagonal(),
            atol=0.005,
        )


def test_zero_mass_cells():
    reference = ot.reference(10, 2)
    assert np.isneginf(ot.Law(reference, cells=[reference.n]).logpdf([0, 0]))
    transport = ot.fit([[-1.0], [0.0], [1.0]])
    inner = ot.Law(transport.reference, weights=[0, 0.5, 0.5, 0])
    pullback = transport.smooth().pullback(inner, candidate=0.0)
    np.testing.assert_allclose(pullback.mean(), [0], atol=1e-8)


def test_reference_pole_ties():
    reference = ot.reference(7, 4)
    np.testing.assert_array_equal(
        reference.locate([[-0.1, 0, 0, 0], [0.1, 0, 0, 0]]), [0, 4]
    )
