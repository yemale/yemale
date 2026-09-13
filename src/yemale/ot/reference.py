"""A reference distribution split into cells of equal probability."""

import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from operator import index

import numpy as np

from yemale._array import count, readonly, restore, rows

from ._core import beta


@dataclass(frozen=True, eq=False)
class Reference:
    """A distribution on the unit ball, split into ``n + 1`` equal-probability cells.

    The direction is uniform on the sphere, independently of a radius uniform
    on ``[0, 1]``. In one dimension, this is uniform on ``[-1, 1]``.
    ``n`` is the calibration size; the extra cell accounts for the candidate.
    """

    n: int
    dimension: int

    def __post_init__(self):
        object.__setattr__(self, "n", count(self.n, "n", minimum=0))
        object.__setattr__(self, "dimension", count(self.dimension, "dimension"))

    @cached_property
    def _partition(self):
        if self.dimension == 1:
            edges = np.linspace(-1.0, 1.0, self.n + 2)
            return (
                readonly(edges[:-1]),
                readonly(edges[1:]),
                readonly(np.empty((self.n + 1, 0, 2))),
            )

        shell_count = 1
        if self.n + 1 > self.dimension:
            shell_count = min(
                int(np.ceil((self.n + 1) ** (1.0 / self.dimension))),
                (self.n + 1) // (self.dimension + 1),
            )
        base, extra = divmod(self.n + 1, shell_count)
        cell_counts = np.full(shell_count, base, dtype=np.int64)
        cell_counts[:extra] += 1
        radial_edges = np.r_[0.0, np.cumsum(cell_counts) / (self.n + 1)]
        directional = {
            int(k): beta._quantile_cells(self.dimension, int(k))
            for k in np.unique(cell_counts)
        }
        bounds = np.concatenate([directional[k] for k in cell_counts])
        return (
            readonly(np.repeat(radial_edges[:-1], cell_counts)),
            readonly(np.repeat(radial_edges[1:], cell_counts)),
            readonly(bounds),
        )

    @cached_property
    def centers(self):
        """Mean reference point in each cell, with shape ``(n + 1, dimension)``."""
        lower, upper, bounds = self._partition
        radial_mean = (lower + upper) / 2.0
        if self.dimension == 1:
            return readonly(radial_mean[:, None])
        direction = beta.directional_barycentres(
            self.dimension,
            self.n + 1,
            bounds=bounds,
            beta_bounds=self._beta_bounds,
        )
        return readonly(radial_mean[:, None] * direction)

    @cached_property
    def _beta_bounds(self):
        return readonly(beta._beta_quantile_bounds(self.dimension, self._partition[2]))

    @cached_property
    def _log_area(self):
        return (
            math.log(2.0)
            + self.dimension / 2.0 * math.log(math.pi)
            - math.lgamma(self.dimension / 2.0)
        )

    @cached_property
    def ranks(self):
        """Distance of each cell centre from the origin, with shape ``(n + 1,)``."""
        return readonly(np.linalg.norm(self.centers, axis=1))

    @cached_property
    def signs(self):
        """Unit directions of ``centers``, with the same shape; zero stays zero."""
        return readonly(
            np.divide(
                self.centers,
                self.ranks[:, None],
                out=np.zeros_like(self.centers),
                where=self.ranks[:, None] > 0.0,
            )
        )

    @cached_property
    def _law(self):
        from .law import Law

        return Law(self)

    @cached_property
    def _cell_entropies(self):
        return readonly(_conditional_entropies(self))

    def _cell_moments(self, order):
        return _cell_moments(self, order)

    def sample(self, size=1, *, rng=None):
        """Draw ``(size, dimension)`` samples; radius is uniform, not volume.

        ``rng`` is a seed or NumPy generator; omit it for fresh randomness.
        """
        generator = np.random.default_rng(rng)
        labels = generator.integers(self.n + 1, size=count(size), dtype=np.int64)
        return self._sample(labels, generator)

    def _sample(self, labels, generator):
        labels = np.asarray(labels, dtype=np.int64)
        coordinates = generator.random((labels.size, self.dimension))
        return self._map(labels.reshape(-1), coordinates).reshape(
            labels.shape + (self.dimension,)
        )

    def _points(self, labels, n_integration_points):
        """Return points used to approximate averages within each selected cell."""
        return _quadrature(
            self,
            tuple(np.asarray(labels).reshape(-1)),
            count(n_integration_points, "n_integration_points"),
        )

    def _map(self, labels, coordinates):
        lower, upper, bounds = self._partition
        radius = lower[labels] + coordinates[:, 0] * (upper - lower)[labels]
        if self.dimension == 1:
            return radius[:, None]
        direction = beta.directions_from_coords(
            self.dimension, bounds[labels], coordinates[:, 1:]
        )
        return radius[:, None] * direction

    def locate(self, value):
        """Return cell labels, choosing the smallest on shared boundaries.

        Points outside the unit ball return -1.
        """
        points, shape = rows(value, self.dimension, "target")
        lower, upper, bounds = self._partition
        if self.dimension == 1:
            labels = np.searchsorted(upper, points[:, 0], side="left").astype(np.int64)
            support = (
                np.isfinite(points[:, 0])
                & (points[:, 0] >= -1.0)
                & (points[:, 0] <= 1.0)
            )
            labels[~support] = -1
        else:
            labels = np.full(len(points), -1, dtype=np.int64)
            # Keep each query-by-cell temporary array near 1 MiB or smaller.
            block_size = max(1, (1 << 20) // (self.n + 1))
            for start in range(0, len(points), block_size):
                block = points[start : start + block_size]
                radius, support = _radius_support(block)
                within = (radius[:, None] >= lower) & (radius[:, None] <= upper)
                oriented = support & (radius > 0.0)
                unit = np.zeros_like(block)
                unit[oriented] = block[oriented] / radius[oriented, None]
                coordinates = beta.direction_cdf(unit)
                for level in range(self.dimension - 1):
                    band = (coordinates[:, level, None] >= bounds[:, level, 0]) & (
                        coordinates[:, level, None] <= bounds[:, level, 1]
                    )
                    band[~oriented] = True
                    within &= band
                labels[start : start + len(block)] = np.where(
                    support & within.any(axis=1), within.argmax(axis=1), -1
                )
        return restore(labels, shape)

    def logpdf(self, value):
        """Return log density with respect to volume, not cell mass.

        Accept a point ``(d,)`` or batch ``(..., d)``. Return ``-inf`` outside
        the unit ball and ``+inf`` at the origin when ``dimension > 1``.
        """
        points, shape = rows(value, self.dimension, "target")
        radius, support = _radius_support(points)
        density = np.full(len(points), -self._log_area)
        if self.dimension > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                density -= (self.dimension - 1) * np.log(radius)
        density[~support] = -np.inf
        return restore(density, shape)

    def pdf(self, value):
        """Return the density at a point ``(d,)`` or batch ``(..., d)``."""
        return np.exp(self.logpdf(value))

    def expect(self, function, *, n_integration_points=64, rng=None):
        """Approximate ``E[function(X)]``.

        ``function`` receives points ``(q, d)`` and returns values ``(q, ...)``.
        The result averages over points and keeps the remaining axes.
        ``n_integration_points`` is the number of points per cell.
        ``rng=None`` uses fixed points; a seed or NumPy generator draws random
        points independently within each cell.
        """
        return self._law.expect(
            function,
            n_integration_points=n_integration_points,
            rng=rng,
        )

    def moment(self, powers):
        """Return the exact raw moment ``E[prod_j X[j] ** powers[j]]``.

        ``powers`` contains one nonnegative integer per coordinate.
        """
        return _reference_moment(self.dimension, _multi_index(powers, self.dimension))

    def mean(self):
        """Return ``E[X]`` as a vector of length ``dimension``."""
        return np.zeros(self.dimension)

    def covariance(self):
        """Return ``E[(X - E[X]) (X - E[X]).T]`` as a square matrix."""
        return np.eye(self.dimension) / (3 * self.dimension)

    def entropy(self):
        """Return differential entropy ``-E[log(pdf(X))]``, in nats."""
        return self._log_area - self.dimension + 1


@lru_cache(maxsize=32, typed=True)
def reference(n, dimension):
    """Return a shared Reference with ``n + 1`` cells in ``dimension`` dimensions.

    ``n`` is the number of calibration observations, not the number of cells.
    """
    return Reference(n, dimension)


def _radius_support(points):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        radius = np.linalg.norm(points, axis=1)
    # Keep tiny nonzero points distinct from the origin.
    small = radius < np.sqrt(np.finfo(float).tiny)
    radius[small] = np.hypot.reduce(points[small], axis=1)
    return radius, np.isfinite(points).all(axis=1) & (radius <= 1.0)


def _multi_index(powers, dimension):
    values = ()
    try:
        values = np.asarray(powers, dtype=object).reshape(-1)
        result = tuple(index(value) for value in values)
    except (TypeError, ValueError):
        result = ()
    if (
        len(result) != dimension
        or any(isinstance(value, (bool, np.bool_)) for value in values)
        or any(value < 0 for value in result)
    ):
        raise ValueError(
            f"powers must contain {dimension} nonnegative integers; got {powers!r}"
        )
    return result


def _reference_moment(dimension, order):
    if any(power % 2 for power in order):
        return 0.0
    half_order = np.asarray(order) / 2.0
    log_direction = (
        math.lgamma(dimension / 2.0)
        - math.lgamma(half_order.sum() + dimension / 2.0)
        + sum(math.lgamma(power + 0.5) - math.lgamma(0.5) for power in half_order)
    )
    return math.exp(log_direction) / (sum(order) + 1)


@lru_cache(maxsize=128)
def _cell_moments(reference, order):
    degree = sum(order)
    if degree == 0:
        return readonly(np.ones(reference.n + 1))
    if degree == 1:
        coordinate = order.index(1)
        return readonly(reference.centers[:, coordinate])

    lower, upper, bounds = reference._partition
    if reference.dimension == 1:
        return readonly(_signed_uniform_moment(lower, upper, degree))
    radial = _positive_uniform_moment(lower, upper, degree)
    angular = beta.angular_moments(
        reference.dimension, bounds, order, beta_bounds=reference._beta_bounds
    )
    return readonly(radial * angular)


def _positive_power_integral(lower, upper, power):
    value = np.zeros_like(lower)
    active = upper > lower
    if power == 0:
        value[active] = upper[active] - lower[active]
        return value
    ratio = lower[active] / upper[active]
    with np.errstate(divide="ignore"):
        difference = -np.expm1((power + 1) * np.log(ratio))
    value[active] = upper[active] ** (power + 1) * difference / (power + 1)
    return value


def _positive_uniform_moment(lower, upper, power):
    return _positive_power_integral(lower, upper, power) / (upper - lower)


def _signed_uniform_moment(lower, upper, power):
    value = np.zeros_like(lower)
    negative = lower < 0.0
    if np.any(negative):
        value[negative] = (-1.0) ** power * _positive_power_integral(
            -np.minimum(upper[negative], 0.0), -lower[negative], power
        )
    positive = upper > 0.0
    if np.any(positive):
        value[positive] += _positive_power_integral(
            np.maximum(lower[positive], 0.0), upper[positive], power
        )
    return value / (upper - lower)


def _conditional_entropies(reference):
    lower, upper, _ = reference._partition
    value = np.full(reference.n + 1, reference._log_area - math.log(reference.n + 1))
    if reference.dimension > 1:
        width = (upper - lower) / upper
        correction = np.zeros_like(lower)
        positive = lower > 0.0
        # This form stays accurate when a radial interval is narrow.
        correction[positive] = (
            (1.0 - width[positive]) * np.log1p(-width[positive]) / width[positive]
        )
        expected_log_radius = np.log(upper) - 1.0 - correction
        value += (reference.dimension - 1) * expected_log_radius
    return value


@lru_cache(maxsize=8)
def _quadrature(reference, labels, n_integration_points):
    if reference.dimension == 1:
        coordinates = ((np.arange(n_integration_points) + 0.5) / n_integration_points)[
            :, None
        ]
    else:
        from scipy.stats import qmc

        coordinates = qmc.Halton(reference.dimension, scramble=False).random(
            n_integration_points + 1
        )[1:]
    points = reference._map(
        np.repeat(labels, n_integration_points),
        np.tile(coordinates, (len(labels), 1)),
    )
    return readonly(
        points.reshape(len(labels), n_integration_points, reference.dimension)
    )
