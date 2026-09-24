"""Density superlevel sets selected by integrated mass."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from yemale._array import count, readonly


@dataclass(frozen=True, eq=False, repr=False)
class DensityRegion:
    """Points whose density reaches a cutoff, including ties.

    Create with ``law.density_region(mass)`` or ``reference.density_region(mass)``.
    ``mass`` estimates the included density integral, not the coverage of future
    observations.

    Attributes:
        mass: Estimated density integral over the region, including cutoff ties.
    """

    _log_density: Callable
    _log_threshold: float
    mass: float

    def __repr__(self):
        return f"DensityRegion(mass={self.mass:.6g}, threshold={self.threshold:.6g})"

    @property
    def threshold(self):
        """Density cutoff; membership is compared in log space."""
        with np.errstate(over="ignore", under="ignore"):
            return float(np.exp(self._log_threshold))

    def contains(self, points):
        """Return membership for one point or a batch, preserving batch axes."""
        return self._log_density(points) >= self._log_threshold


def _density_region(owner, log_density, prepare, mass, n_integration_points):
    """Select a cutoff from quadrature points and their mass under the density.

    prepare(size) returns those points and mass weights; scores only order them.
    Cached callbacks must remain unchanged.
    """
    mass = float(mass)
    if not np.isfinite(mass) or not 0 < mass <= 1:
        raise ValueError("mass must be finite and in (0, 1]")
    size = count(n_integration_points, "n_integration_points")
    cached = getattr(owner, "_density_cache", None)
    if cached is None or cached[0] != size:
        points, weights = prepare(size)
        scores = np.asarray(log_density(points)).reshape(-1)
        if np.isnan(scores).any():
            raise RuntimeError("density is undefined at an integration point")
        order = np.argsort(-scores)
        scores = readonly(scores[order])
        cumulative = np.cumsum(weights[order])
        if len(cumulative):
            # Match the quadrature total despite cumulative rounding.
            cumulative[-1] = weights.sum()
        cached = (size, scores, readonly(cumulative))
        object.__setattr__(owner, "_density_cache", cached)
    _, scores, cumulative = cached
    if not len(cumulative) or mass > cumulative[-1] + 1e-12:
        raise ValueError(
            "integration does not resolve the requested mass; "
            "increase n_integration_points"
        )
    index = min(np.searchsorted(cumulative, mass), len(scores) - 1)
    threshold = float(scores[index])
    end = np.searchsorted(-scores, -threshold, side="right")
    return DensityRegion(log_density, threshold, float(cumulative[end - 1]))
