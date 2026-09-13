"""Check smooth maps, their derivatives, and their inverses."""

import numpy as np
import pytest

from yemale import ot


def test_two_site_map_inverse_and_pullback_density_have_closed_forms():
    transport = ot.fit([[0.0]])
    smooth = transport.smooth(0.4)
    points = np.array([[-0.7], [0.0], [0.9], [15.0]])
    derivative = 0.625 / np.cosh(points[:, 0] / 0.8) ** 2
    np.testing.assert_allclose(smooth(points)[:, 0], 0.5 * np.tanh(points[:, 0] / 0.8))
    # Check relative error so a small, nonzero derivative cannot pass as zero.
    np.testing.assert_allclose(
        smooth.jacobian(points)[:, 0, 0], derivative, rtol=1e-12, atol=0
    )
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
    targets = np.array([-0.2, 0.0, 0.25])
    np.testing.assert_allclose(
        smooth.inverse(targets)[:, 0], 0.8 * np.arctanh(2 * targets), atol=1e-8
    )
    base = ot.Law(transport.reference)._map(
        lambda u: 0.5 * u,
        lambda u: (2 * u, np.log(2.0)),
    )
    pullback = smooth.pullback(base)
    np.testing.assert_allclose(pullback.logpdf(points), np.log(derivative), rtol=1e-12)
    np.testing.assert_allclose(
        smooth(pullback.sample(4, rng=0)), base.sample(4, rng=0), rtol=0, atol=1e-9
    )
    with pytest.raises(ValueError, match="full reference"):
        smooth.pullback(transport.reference)
    for target in [0.5, 0.5000000001]:
        with pytest.raises(ValueError, match="strictly inside"):
            smooth.inverse(target)


def test_smooth_derivatives_and_inverse_use_original_source_coordinates():
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
    mapped, paired_jacobian = smooth.map_jacobian(point)
    np.testing.assert_allclose(mapped, smooth(point), atol=2e-8)
    np.testing.assert_array_equal(paired_jacobian, smooth.jacobian(point))
    np.testing.assert_allclose(smooth.inverse(smooth(point)), point, rtol=0, atol=5e-8)
    np.testing.assert_allclose(
        smooth.reference_distribution(point=point).mean(),
        smooth(point=point),
        atol=1e-14,
    )


def test_small_temperature_inverse_converges_when_direct_newton_stalls():
    transport = ot.fit(np.random.default_rng(0).normal(size=(50, 16)))
    smooth = transport.smooth(1e-6)
    sites = transport.reference.centers
    targets = 0.05 * sites.mean(axis=0) + 0.95 * sites[::10]
    np.testing.assert_allclose(
        smooth(smooth.inverse(targets)), targets, rtol=0, atol=1e-9
    )


def test_smooth_target_coordinates_preserve_potential_jacobian_and_inverse():
    source = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5]])
    target = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    points = np.array([[0.2, 0.4], [-0.1, 0.3]])
    smooth = ot.fit(source, target=target).smooth(0.4)
    for scale, shift in ((1.0, 20.0), (1e-150, 0.0), (1e150, 0.0)):
        transformed = ot.fit(source, target=scale * target + shift).smooth(scale * 0.4)
        np.testing.assert_allclose(
            (transformed(points) - shift) / scale, smooth(points), atol=1e-12
        )
        np.testing.assert_allclose(
            transformed.jacobian(points) / scale, smooth.jacobian(points), atol=1e-12
        )
        linear = ((points - source.mean(axis=0)) * shift).sum(axis=1)
        np.testing.assert_allclose(
            (transformed.potential(points) - linear) / scale,
            smooth.potential(points),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transformed.inverse(transformed(points)), points, atol=1e-8
        )


def test_inverse_requires_full_rank_and_strict_hull_interior():
    deficient = ot.fit([[0.0, 0.0]]).smooth()
    with pytest.raises(ValueError, match="affinely span"):
        deficient.inverse(deficient([0.0, 0.0]))
    smooth = ot.fit(
        [[0.0, 0.0], [1.0, 1.0]],
        target=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    ).smooth(0.4)
    np.testing.assert_allclose(
        smooth(smooth.inverse([0.2, 0.2])), [0.2, 0.2], rtol=0, atol=1e-9
    )
    for target in [[0.5, 0.5], [0.6, 0.6]]:
        with pytest.raises(ValueError, match="strictly inside"):
            smooth.inverse(target)
