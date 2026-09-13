"""Reference-cell mixtures and their images."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from yemale._array import count, readonly, restore, rows, scalars

from .reference import Reference


@dataclass(frozen=True, eq=False, init=False, repr=False)
class Law:
    """A probability mixture of reference cells, optionally mapped into source space.

    ``forward(labels, points)`` maps ``(q,)``, ``(q, d)`` to ``(q, d)``.
    ``backward(points)`` returns inverse points ``(q, d)`` and the inverse
    log-absolute determinant: a scalar constant, ``(q,)``, or ``(q, 1)``.
    """

    reference: Reference
    cells: np.ndarray
    weights: np.ndarray  # Aligned with cells.
    _forward: Callable | None
    _backward: Callable | None

    def __repr__(self):
        return (
            f"Law(reference={self.reference!r}, cell_count={len(self.cells)}, "
            f"mapped={self._forward is not None})"
        )

    def __init__(
        self, reference, cells=None, weights=None, forward=None, backward=None
    ):
        if not isinstance(reference, Reference):
            raise TypeError("a Law requires a Reference")
        selected = np.arange(reference.n + 1) if cells is None else np.atleast_1d(cells)
        if (
            selected.ndim != 1
            or not np.issubdtype(selected.dtype, np.integer)
            or not len(selected)
            or np.any(selected < 0)
            or np.any(selected > reference.n)
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError(
                f"cells must be nonempty, distinct integer labels in [0, {reference.n}]; "
                f"got {selected}"
            )
        probability = (
            np.full(len(selected), 1 / len(selected))
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if probability.shape != selected.shape:
            raise ValueError(
                f"weights must have shape {selected.shape}, one per selected cell; "
                f"got {probability.shape}"
            )
        if (
            not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.isclose(probability.sum(), 1, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("weights must be finite, nonnegative and sum to one")
        if forward is None and backward is not None:
            raise ValueError("an inverse requires a forward map")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "cells", readonly(selected, dtype=np.int64))
        object.__setattr__(self, "weights", readonly(probability / probability.sum()))
        object.__setattr__(self, "_forward", forward)
        object.__setattr__(self, "_backward", backward)

    @cached_property
    def _probability(self):
        value = np.zeros(self.reference.n + 1)
        value[self.cells] = self.weights
        return readonly(value)

    def _push(self, labels, points):
        if self._forward is None:
            return points
        mapped, _ = rows(
            self._forward(labels, points), self.reference.dimension, "filling"
        )
        if len(mapped) != len(points) or not np.isfinite(mapped).all():
            raise ValueError(
                f"filling must return {len(points)} finite points; got shape {mapped.shape}"
            )
        return mapped

    def sample(self, size=1, *, rng=None):
        """Draw ``(size, dimension)`` samples; rng is a seed or NumPy generator."""
        generator = np.random.default_rng(rng)
        labels = generator.choice(self.cells, size=count(size), p=self.weights)
        return self._push(labels, self.reference._sample(labels, generator))

    def logpdf(self, value):
        """Log Lebesgue density; a mapped law needs its branch inverse and Jacobian."""
        points, shape = rows(value, self.reference.dimension, "value")
        target, jacobian = points, np.zeros(len(points))
        if self._forward is not None:
            if self._backward is None:
                raise TypeError(
                    "density requires a branch inverse and its log-Jacobian"
                )
            target, jacobian = self._backward(points)
            target, _ = rows(target, self.reference.dimension, "inverse")
            if len(target) != len(points):
                raise ValueError(
                    f"inverse must return {len(points)} points; got {len(target)}"
                )
            jacobian = scalars(jacobian, len(points), "inverse log-Jacobian")
        cells = np.asarray(self.reference.locate(target)).reshape(-1)
        mass = np.zeros(len(points))
        inside = cells >= 0
        mass[inside] = self._probability[cells[inside]]
        valid = (mass > 0) & (jacobian != -np.inf)
        result = np.full(len(points), -np.inf)
        # Mask zero mass first: the spherical density is infinite at the origin.
        result[valid] = (
            self.reference.logpdf(target[valid])
            + np.log((self.reference.n + 1) * mass[valid])
            + jacobian[valid]
        )
        return restore(result, shape)

    def _map(self, forward, backward=None):
        """Compose ``forward(points)``; ``backward`` follows the Law contract."""

        def mapped(labels, points):
            return forward(self._push(labels, points))

        inverse = None
        if backward is not None and (
            self._forward is None or self._backward is not None
        ):

            def inverse(value):
                target, jacobian = backward(value)
                jacobian = scalars(jacobian, len(value), "inverse log-Jacobian")
                if self._backward is not None:
                    target, previous = self._backward(target)
                    previous = scalars(previous, len(value), "inverse log-Jacobian")
                    jacobian = jacobian + previous
                return target, jacobian

        return Law(self.reference, self.cells, self.weights, mapped, inverse)


def expect(law, function, *, size=64):
    """Numerically integrate a vectorized function, using size nodes per cell.

    ``function`` maps ``(q, d)`` to ``(q, ...)`` or a scalar constant.
    Cached reference nodes are read-only; callbacks must not mutate input rows.
    """
    law = law if isinstance(law, Law) else Law(law)
    active = law.weights > 0
    cells, weights = law.cells[active], law.weights[active]
    points = law.reference._points(cells, size)
    labels = np.repeat(cells, points.shape[1])
    mapped = law._push(labels, points.reshape(-1, law.reference.dimension))
    values = np.asarray(function(mapped))
    if values.ndim == 0:
        return values.item()
    if len(values) != len(mapped):
        raise ValueError(
            "function must return one value per point or a scalar constant"
        )
    means = values.reshape(len(cells), points.shape[1], *values.shape[1:]).mean(axis=1)
    return np.average(means, axis=0, weights=weights)
