"""Reference and source-law mathematical contracts."""

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
        ot.expect(reference, lambda u: u * u, size=256), 1 / (3 * dimension), atol=0.003
    )
    log_area = np.log(2 * np.pi ** (dimension / 2) / math.gamma(dimension / 2))
    np.testing.assert_allclose(
        reference.logpdf(sample),
        -log_area - (dimension - 1) * np.log(np.linalg.norm(sample, axis=1)),
    )
    assert np.isneginf(reference.logpdf(np.full(dimension, 2.0)))
    assert reference is ot.reference(39, dimension)
    assert not reference.centers.flags.writeable


def test_filling_preserves_mass_support_and_inverse_jacobian():
    transport = ot.fit([[0.0]])

    def shift(labels):
        return np.where(labels == 0, -1.0, 1.0)[:, None]

    def filling(labels, points):
        return 2 * points + shift(labels)

    law = transport.extend(
        filling,
        inverse=lambda labels, z: (z - shift(labels)) / 2,
        inverse_logabsdet=lambda labels, z: np.full_like(z, -np.log(2.0)),
    )
    np.testing.assert_allclose(
        np.exp(law.logpdf([-2, -0.5, 0.5, 2])), [0.25, 0, 0, 0.25]
    )
    sample = law.sample(10000, rng=7)
    assert np.all((np.abs(sample) >= 1) & (np.abs(sample) <= 3))
    np.testing.assert_allclose(np.exp(law.logpdf(sample)), 0.25)
    assert (sample < 0).mean() == pytest.approx(0.5, abs=0.02)
    np.testing.assert_allclose(ot.expect(law, lambda z: z), [0], atol=1e-15)
    np.testing.assert_allclose(
        ot.expect(transport.law(-0.2), lambda u: u), transport(-0.2)
    )
    with pytest.raises(TypeError, match="density requires"):
        transport.extend(filling).logpdf(0)


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
    np.testing.assert_allclose(ot.expect(law, lambda z: z), [0.5])
    assert ot.expect(law, lambda z: 7) == 7
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


def test_zero_mass_cells_contribute_neither_density_nor_integrals():
    reference = ot.reference(10, 2)
    assert np.isneginf(ot.Law(reference, cells=[reference.n]).logpdf([0, 0]))
    transport = ot.fit([[-1.0], [0.0], [1.0]])
    inner = ot.Law(transport.reference, weights=[0, 0.5, 0.5, 0])
    # Excluded outer cells are outside the inverse domain and must not be evaluated.
    pullback = transport.smooth().pullback(inner)
    np.testing.assert_allclose(ot.expect(pullback, lambda z: z), [0], atol=1e-8)
