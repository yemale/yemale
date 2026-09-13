"""Equal-mass spherical reference cells and their conditional laws."""

import math
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np

from yemale._array import count, readonly, restore, rows

from ._core import beta


@dataclass(frozen=True, eq=False)
class Reference:
    """The spherical-uniform law, partitioned into ``n + 1`` equal-mass cells."""

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
        lower, upper, bounds = self._partition
        radial_mean = (lower + upper) / 2.0
        if self.dimension == 1:
            return readonly(radial_mean[:, None])
        direction = beta.directional_barycentres(
            self.dimension, self.n + 1, bounds=bounds
        )
        return readonly(radial_mean[:, None] * direction)

    @cached_property
    def ranks(self):
        return readonly(np.linalg.norm(self.centers, axis=1))

    @cached_property
    def signs(self):
        return readonly(
            np.divide(
                self.centers,
                self.ranks[:, None],
                out=np.zeros_like(self.centers),
                where=self.ranks[:, None] > 0.0,
            )
        )

    def sample(self, size=1, *, rng=None):
        """Draw rows from the reference law; radius is uniform, not volume."""
        generator = np.random.default_rng(rng)
        labels = generator.integers(self.n + 1, size=count(size), dtype=np.int64)
        return self._sample(labels, generator)

    def _sample(self, labels, generator):
        labels = np.asarray(labels, dtype=np.int64)
        coordinates = generator.random((labels.size, self.dimension))
        return self._map(labels.reshape(-1), coordinates).reshape(
            labels.shape + (self.dimension,)
        )

    def _points(self, labels, size):
        """Cached equal-weight cell cubature; arbitrary integrals are approximate."""
        return _quadrature(self, tuple(np.asarray(labels).reshape(-1)), count(size))

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
        """Return cell labels, using the first cell on boundaries and -1 outside support."""
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
            radius, support = _radius_support(points)
            within = (radius[:, None] >= lower) & (radius[:, None] <= upper)
            oriented = support & (radius > 0.0)
            unit = np.zeros_like(points)
            unit[oriented] = points[oriented] / radius[oriented, None]
            coordinates = beta.direction_cdf(unit)
            for level in range(self.dimension - 1):
                band = (coordinates[:, level, None] >= bounds[:, level, 0]) & (
                    coordinates[:, level, None] <= bounds[:, level, 1]
                )
                within[oriented] &= band[oriented]
            labels = np.where(support & within.any(axis=1), within.argmax(axis=1), -1)
        return restore(labels, shape)

    def logpdf(self, value):
        """Reference log density with respect to Lebesgue measure."""
        points, shape = rows(value, self.dimension, "target")
        radius, support = _radius_support(points)
        log_area = (
            math.log(2.0)
            + self.dimension / 2.0 * math.log(math.pi)
            - math.lgamma(self.dimension / 2.0)
        )
        density = np.full(len(points), -log_area)
        if self.dimension > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                density -= (self.dimension - 1) * np.log(radius)
        density[~support] = -np.inf
        return restore(density, shape)


@lru_cache(maxsize=32, typed=True)
def reference(n, dimension):
    """Return the canonical ``n + 1`` reference cells for calibration size ``n``."""
    return Reference(n, dimension)


def _radius_support(points):
    radius = np.linalg.norm(points, axis=1)
    return radius, np.isfinite(points).all(axis=1) & (radius <= 1.0)


@lru_cache(maxsize=8)
def _quadrature(reference, labels, size):
    if reference.dimension == 1:
        coordinates = ((np.arange(size) + 0.5) / size)[:, None]
    else:
        from scipy.stats import qmc

        coordinates = qmc.Halton(reference.dimension, scramble=False).random(size + 1)[
            1:
        ]
    points = reference._map(
        np.repeat(labels, size), np.tile(coordinates, (len(labels), 1))
    )
    return readonly(points.reshape(len(labels), size, reference.dimension))
