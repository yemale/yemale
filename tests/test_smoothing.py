"""Check smooth maps, their derivatives, and augmented reverse maps."""

import numpy as np
from scipy.special import expit, softmax

from yemale import ot


def test_default_temperature_uses_assignment_margins():
    source = np.random.default_rng(8).normal(size=(39, 1))
    transport = ot.fit(source)
    generator = np.random.default_rng(0)
    probes = source[generator.integers(0, len(source), size=4000)]
    probes += 0.02 * generator.multivariate_normal(
        [0.0], np.atleast_2d(np.cov(source, rowvar=False)) + 1e-9 * source.var(), 4000
    )
    logits = probes @ transport.reference.centers.T - transport.phi
    radii = transport.reference.ranks
    winners = logits.argmax(axis=1)
    best = logits[np.arange(len(logits)), winners]
    logits[
        np.isclose(radii[None, :], radii[winners, None], rtol=0, atol=4e-15)
    ] = -np.inf
    gaps = best - logits.max(axis=1)
    temperature = 100 * np.median(gaps[gaps > 1e-10])
    np.testing.assert_allclose(
        transport.smooth()(source), transport.smooth(temperature)(source), atol=1e-12
    )
    # A single radial level has no comparison margin.
    single = ot.fit([0.0])
    np.testing.assert_array_equal(single.smooth()([0.2]), single.smooth(0.05)([0.2]))


def test_two_site_map_matches_tanh():
    # Sites are +/-0.5, giving T(z) = 0.5*tanh(z/0.8) at temperature 0.4.
    transport = ot.fit([[0.0]])
    smooth = transport.smooth(0.4)
    points = np.array([[-0.7], [0.0], [0.9], [15.0]])
    derivative = 0.625 / np.cosh(points[:, 0] / 0.8) ** 2
    np.testing.assert_allclose(smooth(points)[:, 0], 0.5 * np.tanh(points[:, 0] / 0.8))
    # Check relative error so a small, nonzero derivative cannot pass as zero.
    np.testing.assert_allclose(
        smooth.jacobian(points)[:, 0, 0], derivative, rtol=1e-12, atol=0
    )
    # Exercise both kernel dispatches and preservation of two batch axes.
    for batch in (points, np.tile(points, (8192, 1)).reshape(2, 16384, 1)):
        mapped, jacobian = smooth.map_jacobian(batch)
        np.testing.assert_allclose(mapped, 0.5 * np.tanh(batch / 0.8))
        np.testing.assert_allclose(
            jacobian[..., 0, 0],
            0.625 / np.cosh(batch[..., 0] / 0.8) ** 2,
            rtol=1e-12,
            atol=0,
        )
        np.testing.assert_allclose(
            smooth.potential(batch), 0.4 * np.log(2 * np.cosh(batch[..., 0] / 0.8))
        )


def test_derivatives_in_source_coordinates():
    rng = np.random.default_rng(7)
    transport = ot.fit(4 * rng.normal(size=(50, 3)) + 3)
    smooth = transport.smooth(0.6)
    point = 4 * rng.normal(size=3) + 3
    delta = 1e-5 * np.eye(3)
    gradient = (
        smooth.potential(point + delta) - smooth.potential(point - delta)
    ) / 2e-5
    jacobian = ((smooth(point + delta) - smooth(point - delta)) / 2e-5).T
    np.testing.assert_allclose(gradient, smooth(point), atol=2e-8)
    np.testing.assert_allclose(jacobian, smooth.jacobian(point), atol=2e-8)
    np.testing.assert_allclose(
        smooth.reference_distribution(point=point).mean(),
        smooth(point=point),
        atol=1e-14,
    )


def test_target_affine_transformation():
    source = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5]])
    target = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    points = np.array([[0.2, 0.4], [-0.1, 0.3]])
    smooth = ot.fit(source, target=target).smooth(0.4)
    for scale, shift in ((2.0, 20.0), (1e-150, 0.0), (1e150, 0.0)):
        transformed = ot.fit(source, target=scale * target + shift).smooth(scale * 0.4)
        np.testing.assert_allclose(
            (transformed(points) - shift) / scale, smooth(points), atol=1e-12
        )
        np.testing.assert_allclose(
            transformed.jacobian(points) / scale, smooth.jacobian(points), atol=1e-12
        )
        # The potential gauge is anchored at the source mean.
        linear = ((points - source.mean(axis=0)) * shift).sum(axis=1)
        np.testing.assert_allclose(
            (transformed.potential(points) - linear) / scale,
            smooth.potential(points),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transformed.inverse(transformed(points), candidate=points[0]),
            smooth.inverse(smooth(points), candidate=points[0]),
            atol=1e-12,
        )


def test_augmented_inverse_matches_logistic():
    transport = ot.fit([[0.0]])
    smooth = transport.smooth(0.4)
    candidate = np.array([2.0])

    # Phi(0) = 0 and Phi(2) = 1, so both augmented anchors enter this logistic.
    def reverse(targets):
        return 2 * expit((2 * targets - 1) / 0.4)

    targets = np.array([[-2.0], [-0.5], [0.0], [0.5], [2.0]])
    np.testing.assert_allclose(
        smooth.inverse(targets, candidate=candidate), reverse(targets)
    )

    law = smooth.pullback(transport.reference, candidate=candidate)
    shifted = ot.Law(
        transport.reference, forward=lambda targets, labels: targets + 0.25
    )
    shifted_pullback = smooth.pullback(shifted, candidate=candidate)
    np.testing.assert_allclose(
        law.sample(16, rng=4), reverse(ot.Law(transport.reference).sample(16, rng=4))
    )
    np.testing.assert_allclose(
        shifted_pullback.sample(16, rng=4), reverse(shifted.sample(16, rng=4))
    )
    expected_mean = 0.2 * (np.logaddexp(0, 2.5) - np.logaddexp(0, -7.5))
    np.testing.assert_allclose(
        law.mean(n_integration_points=512), [expected_mean], atol=1e-7
    )


def test_augmented_inverse_uses_all_source_anchors():
    source = np.array([[-4.0, -4.0], [3.0, 3.0], [10.0, -7.5]])
    # Collinear target sites are allowed, including queries outside their hull.
    sites = np.array([[-1.0, -0.5], [0.0, 0.0], [0.7, 0.35], [1.3, 0.65]]) + [20, -15]
    candidate = np.array([0.4, -0.2])
    targets = np.array([[[-2.0, 1.0], [0.3, 0.2]], [[1.0, 2.0], [0.5, -0.8]]])
    targets += [20, -15]
    transport = ot.fit(source, target=sites)
    anchors = np.vstack((source, candidate))
    logits = (targets @ anchors.T - transport.potential(anchors)) / 0.4
    expected = softmax(logits, axis=-1) @ anchors
    actual = transport.smooth(0.4).inverse(targets, candidate=candidate)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
