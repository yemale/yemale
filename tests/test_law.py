"""Check sampling, density, and averages for reference and mapped distributions."""

import math

import numpy as np
import pytest

from yemale import ot


@pytest.mark.parametrize("dimension", [1, 2, 5])
def test_reference_cells_recover_the_spherical_law(dimension):
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
    np.testing.assert_allclose(reference.pdf(sample), np.exp(reference.logpdf(sample)))
    assert np.isneginf(reference.logpdf(np.full(dimension, 2.0)))
    np.testing.assert_allclose(reference.mean(), 0, atol=1e-15)
    np.testing.assert_allclose(
        reference.covariance(), np.eye(dimension) / (3 * dimension), atol=1e-15
    )
    order = (2,) + (0,) * (dimension - 1)
    np.testing.assert_allclose(reference.moment(powers=order), 1 / (3 * dimension))
    np.testing.assert_allclose(reference.entropy(), log_area - dimension + 1)
    assert reference is ot.reference(39, dimension)
    assert not reference.centers.flags.writeable


def test_reference_lookup_boundaries_support_and_batch_axes():
    reference = ot.reference(7, 2)
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


def test_tiny_reference_points_keep_their_direction_and_finite_log_density():
    reference = ot.reference(1, 2)
    with np.errstate(over="raise", invalid="raise", under="raise"):
        assert reference.locate([0.0, -1e-200]) == 1
        expected = -np.log(2 * np.pi) - np.log(1e-200)
        np.testing.assert_allclose(reference.logpdf([0.0, -1e-200]), expected)
        np.testing.assert_allclose(
            ot.Law(reference, cells=[1]).logpdf([0.0, -1e-200]), expected + np.log(2)
        )
        assert np.isneginf(reference.logpdf([1e308, 0.0]))


def test_cell_moments_recover_full_reference_moments():
    reference = ot.reference(1, 10)
    law = ot.Law(reference)
    np.testing.assert_allclose(
        law.covariance(), reference.covariance(), rtol=1e-13, atol=1e-14
    )
    powers = (0, 34) + (0,) * 8
    np.testing.assert_allclose(
        law.moment(powers), reference.moment(powers), rtol=1e-11, atol=0
    )


def test_predictive_distribution_preserves_mass_support_and_inverse_jacobian(
    monkeypatch,
):
    transport = ot.fit([[0.0]])

    def shift(labels):
        return np.where(labels == 0, -1.0, 1.0)[:, None]

    def map_from_reference(points, labels):
        return 2 * points + shift(labels)

    law = transport.predictive_distribution(
        map_from_reference=map_from_reference,
        inverse=lambda z, labels: (z - shift(labels)) / 2,
        inverse_logabsdet=lambda z, labels: np.full_like(z, -np.log(2.0)),
    )
    locate = ot.Reference.locate
    calls = []

    def counted_locate(reference, points):
        calls.append(len(points))
        return locate(reference, points)

    monkeypatch.setattr(ot.Reference, "locate", counted_locate)
    np.testing.assert_allclose(
        np.exp(law.logpdf([-2, -0.5, 0.5, 2])), [0.25, 0, 0, 0.25]
    )
    assert calls == [4]
    np.testing.assert_allclose(law.pdf([-2, -0.5, 0.5, 2]), [0.25, 0, 0, 0.25])
    sample = law.sample(10000, rng=7)
    assert np.all((np.abs(sample) >= 1) & (np.abs(sample) <= 3))
    np.testing.assert_allclose(np.exp(law.logpdf(sample)), 0.25)
    assert (sample < 0).mean() == pytest.approx(0.5, abs=0.02)
    np.testing.assert_allclose(law.mean(), [0], atol=1e-15)
    np.testing.assert_allclose(
        law.moment(powers=2, n_integration_points=512), 13 / 3, atol=2e-6
    )
    np.testing.assert_allclose(
        law.covariance(n_integration_points=512), [[13 / 3]], atol=2e-6
    )
    np.testing.assert_allclose(law.entropy(), np.log(4))
    assert law.dimension == 1
    np.testing.assert_allclose(
        transport.reference_distribution(-0.2).mean(), transport(-0.2)
    )
    with pytest.raises(TypeError, match="density requires"):
        transport.predictive_distribution(map_from_reference).logpdf(0)
    with pytest.raises(TypeError, match="entropy requires"):
        transport.predictive_distribution(map_from_reference).entropy()


def test_inverse_log_jacobians_compose_one_scalar_per_row():
    law = ot.Law(
        ot.reference(1, 1),
        forward=lambda labels, u: 2 * u,
        backward=lambda z: (z / 2, np.full_like(z, -np.log(2))),
    )
    mapped = law._map(
        lambda z: 3 * z,
        lambda z: (z / 3, np.full(len(z), -np.log(3))),
    )
    points = np.array([[-1.0], [0.0], [1.0]])
    np.testing.assert_allclose(np.exp(law.logpdf(points)), 0.25)
    np.testing.assert_allclose(np.exp(mapped.logpdf(points)), 1 / 12)
    bad = law._map(lambda z: z, lambda z: (z, np.zeros(2)))
    with pytest.raises(ValueError, match="2 values for 3 points"):
        bad.logpdf(points)


def test_mixture_density_sampling_and_expectation_agree():
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
    for rng in (0, np.random.default_rng(1)):
        assert law.expect(
            lambda z: z[:, 0] > 0,
            n_integration_points=8,
            rng=rng,
        ) == pytest.approx(0.75)
    first = law.expect(
        lambda z: z[:, 0] ** 2,
        n_integration_points=8,
        rng=3,
    )
    second = law.expect(
        lambda z: z[:, 0] ** 2,
        n_integration_points=8,
        rng=3,
    )
    assert first == second
    cell = ot.Law(ot.reference(1, 3), cells=[0])
    np.testing.assert_allclose(cell.moment(powers=(40, 0, 0)), 1 / 41**2)
    np.testing.assert_allclose(ot.reference(7, 3).moment(powers=(2, 2, 0)), 1 / 75)
    for invalid in ((1, 2), (1, -1, 0), (1.5, 0, 0)):
        with pytest.raises(ValueError, match="nonnegative integers"):
            ot.reference(1, 3).moment(powers=invalid)
    nodes = reference._points(law.cells, 64)
    assert nodes is reference._points(law.cells, 64) and not nodes.flags.writeable
    with pytest.raises(ValueError, match="distinct"):
        ot.Law(reference, cells=[0, 0, 1])
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        ot.Law(reference, cells=[4])
    with pytest.raises(ValueError, match="one per selected cell"):
        ot.Law(reference, cells=[0, 1], weights=[1.0])
    for invalid in ([1, -1], [0.2, 0.2]):
        with pytest.raises(ValueError, match="weights"):
            ot.Law(reference, cells=[1, 3], weights=invalid)


def test_expect_requires_one_result_per_point():
    reference = ot.reference(7, 2)
    for distribution in (reference, ot.Law(reference)):
        for function in (np.linalg.norm, lambda x: np.ones(3)):
            with pytest.raises(ValueError, match="one result per point"):
                distribution.expect(function)
        np.testing.assert_allclose(
            distribution.expect(lambda x: np.linalg.norm(x, axis=-1)), 0.5, atol=0.005
        )
        np.testing.assert_allclose(
            distribution.expect(lambda x: x * x),
            distribution.covariance().diagonal(),
            atol=0.005,
        )


def test_zero_mass_cells_contribute_neither_density_nor_integrals():
    reference = ot.reference(10, 2)
    assert np.isneginf(ot.Law(reference, cells=[reference.n]).logpdf([0, 0]))
    transport = ot.fit([[-1.0], [0.0], [1.0]])
    inner = ot.Law(transport.reference, weights=[0, 0.5, 0.5, 0])
    # Excluded outer cells are outside the inverse domain and must not be evaluated.
    pullback = transport.smooth().pullback(inner)
    np.testing.assert_allclose(pullback.mean(), [0], atol=1e-8)
