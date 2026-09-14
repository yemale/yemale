"""Distributions built from reference cells and optional maps."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from yemale._array import count, readonly, restore, rows, scalars

from .density import _density_region
from .reference import Reference, _multi_index


@dataclass(frozen=True, eq=False, init=False, repr=False)
class Law:
    """A probability distribution built from cells of a Reference.

    Sampling chooses a cell using its weight, then draws from the reference
    distribution within that cell. An optional map transforms the samples.

    Attributes:
        reference: Reference distribution containing the selected cells.
        cells: Distinct zero-based reference-cell labels.
        weights: Cell probabilities, aligned with cells and summing to one.
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

    @property
    def dimension(self):
        """Number of coordinates in each sample."""
        return self.reference.dimension

    def __init__(
        self, reference, cells=None, weights=None, forward=None, backward=None
    ):
        """Choose reference cells, their probabilities, and optional maps.

        Args:
            reference: Reference containing the cells.
            cells: Distinct integer labels in ``[0, reference.n]``. Defaults to all.
            weights: Nonnegative probabilities, one per selected cell, summing to one.
                Defaults to equal weights over the selected cells.
            forward: Optional ``forward(points, labels)`` taking ``(q, d)``
                reference points and ``(q,)`` labels. Returns mapped points ``(q, d)``.
                For density evaluation, it must be one-to-one across selected cells.
            backward: Optional ``backward(points)`` taking mapped points ``(q, d)``.
                Returns reference points ``(q, d)`` and the inverse Jacobian's log
                absolute determinant, as a scalar, ``(q,)``, or ``(q, 1)``.
                Required for density and entropy when ``forward`` is set.

        Notes:
            Callbacks must treat input points as read-only.
            A ``-inf`` log determinant can mark points outside the mapped support.
        """
        if not isinstance(reference, Reference):
            raise TypeError("a Law requires a Reference")
        if forward is not None and not callable(forward):
            raise TypeError("forward must be callable as (points, labels)")
        if backward is not None and not callable(backward):
            raise TypeError("backward must be callable as (points)")
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
            self._forward(points, labels), self.reference.dimension, "forward map"
        )
        if len(mapped) != len(points) or not np.isfinite(mapped).all():
            raise ValueError(
                f"forward map must return {len(points)} finite points; "
                f"got shape {mapped.shape}"
            )
        return mapped

    def _integration_points(self, n_integration_points, rng=None):
        n_integration_points = count(n_integration_points, "n_integration_points")
        active = self.weights > 0
        cells, weights = self.cells[active], self.weights[active]
        labels = np.repeat(cells, n_integration_points)
        if rng is None:
            points = self.reference._points(cells, n_integration_points).reshape(
                -1, self.dimension
            )
        else:
            generator = np.random.default_rng(rng)
            points = self.reference._sample(labels, generator)
        mapped = self._push(labels, points)
        return mapped, np.repeat(weights / n_integration_points, n_integration_points)

    def sample(self, size=1, *, rng=None):
        """Draw ``(size, dimension)`` samples; rng is a seed or NumPy generator."""
        generator = np.random.default_rng(rng)
        labels = generator.choice(self.cells, size=count(size), p=self.weights)
        return self._push(labels, self.reference._sample(labels, generator))

    def logpdf(self, value):
        """Return the natural logarithm of ``pdf(value)``.

        Accept a point ``(d,)`` or batch ``(..., d)``; return one value per point.
        A mapped law requires ``backward``; outside its support, return ``-inf``.
        """
        points, shape = rows(value, self.reference.dimension, "value")
        target, jacobian = points, np.zeros(len(points))
        if self._forward is not None:
            if self._backward is None:
                raise TypeError(
                    "density requires an inverse map and its log-Jacobian; "
                    "supply backward to Law, or inverse and inverse_logabsdet "
                    "to predictive_distribution"
                )
            target, jacobian = self._backward(points)
            target, _ = rows(target, self.reference.dimension, "inverse")
            if len(target) != len(points):
                raise ValueError(
                    f"inverse must return {len(points)} points; got {len(target)}"
                )
            jacobian = scalars(jacobian, len(points), "inverse log-Jacobian")
        if np.all(self._probability == self._probability[0]):
            # Equal weights over all cells recover the full reference density.
            result = np.asarray(self.reference.logpdf(target)).reshape(-1)
            valid = (result != -np.inf) & (jacobian != -np.inf)
            result[~valid] = -np.inf
            result[valid] += jacobian[valid]
            return restore(result, shape)
        cells = np.asarray(self.reference.locate(target)).reshape(-1)
        mass = np.zeros(len(points))
        inside = cells >= 0
        mass[inside] = self._probability[cells[inside]]
        valid = (mass > 0) & (jacobian != -np.inf)
        result = np.full(len(points), -np.inf)
        # Skip cells with zero probability before evaluating a density that may be infinite.
        result[valid] = (
            self.reference.logpdf(target[valid])
            + np.log((self.reference.n + 1) * mass[valid])
            + jacobian[valid]
        )
        return restore(result, shape)

    def pdf(self, value):
        r"""Return density per unit volume, not the probability of a cell.

        Accept ``(d,)`` or ``(..., d)``; return one value per point, zero outside
        the support. For reference law nu, n = reference.n, cell weight w_j,
        and inverse map q_j,

        .. math::

            p(x)=(n+1)w_j p_\nu(q_j(x))|\det Dq_j(x)|.

        The formula holds almost everywhere on the image of cell j. Mapped laws
        require ``backward`` and a differentiable map with nonsingular Jacobian,
        one-to-one across selected cells. Without a map, q_j is the identity.
        """
        return np.exp(self.logpdf(value))

    def density_region(self, mass, *, n_integration_points=256):
        r"""Select a density superlevel set with approximately the requested mass.

        .. math:: A_t=\{x:p(x)\ge t\},\qquad \int_{A_t}p(x)\,dx\approx\mathrm{mass}.

        ``mass`` is in (0, 1]. Include all ties at the cutoff; the returned
        ``mass`` may therefore be larger. This is not a future-data coverage
        guarantee. ``n_integration_points`` sets the points per selected cell;
        the prepared density values are reused for subsequent mass requests.
        Mapped laws require the inverse used by ``logpdf``.
        """
        return _density_region(
            self, self.logpdf, self._integration_points, mass, n_integration_points
        )

    def expect(self, function, *, n_integration_points=64, rng=None):
        r"""Approximate ``E[function(X)]`` under this distribution.

        .. math::

            \mathbb E[f(X)]
            =\sum_j w_j\,\mathbb E_\nu[f(Q_j(U))\mid U\in L_j].

        Here nu is the reference law, w_j the cell weight, and Q_j the forward
        map in cell L_j (the identity when no map is supplied).

        ``function`` receives points ``(q, d)`` and returns values ``(q, ...)``.
        The result averages over points and keeps the remaining axes.
        ``n_integration_points`` is the number of points per selected cell.
        ``rng=None`` uses fixed points (interval midpoints in 1-D, Halton points
        otherwise). A seed or NumPy generator draws random points within each cell.
        """
        mapped, weights = self._integration_points(n_integration_points, rng)
        values = np.asarray(function(mapped))
        if values.ndim == 0 or len(values) != len(mapped):
            raise ValueError(
                f"function must return one result per point with shape ({len(mapped)}, ...); "
                f"got {values.shape}. For coordinate sums or norms, use axis=-1."
            )
        return np.average(values, axis=0, weights=weights)

    def moment(self, powers, *, n_integration_points=64):
        r"""Return a raw moment, with one power per coordinate.

        ``powers`` gives the nonnegative integers in alpha.
        For example, ``moment((2, 0))`` means ``E[X[0]**2]``.

        .. math::

            \mathrm{moment}(\alpha)
            =\mathbb E\!\left[\prod_{r=1}^d X_r^{\alpha_r}\right].

        Reference-cell moments are exact. Mapped moments use fixed integration
        points; ``n_integration_points`` sets their number per cell.
        """
        powers = _multi_index(powers, self.dimension)
        if self._forward is None:
            return self._reference_moment(powers)
        powers = np.asarray(powers)
        return self.expect(
            lambda points: np.prod(points**powers, axis=1),
            n_integration_points=n_integration_points,
        )

    def mean(self, *, n_integration_points=64):
        r"""Return the mean ``E[X]`` as a vector of length ``dimension``.

        .. math::

            \mathbb E[X]=(\mathbb E[X_1],\ldots,\mathbb E[X_d])^\top.

        Reference-cell means are exact. Mapped means use fixed points;
        ``n_integration_points`` sets their number per cell.
        """
        if self._forward is None:
            return np.average(
                self.reference.centers[self.cells], axis=0, weights=self.weights
            )
        points, weights = self._integration_points(n_integration_points)
        return np.average(points, axis=0, weights=weights)

    def covariance(self, *, n_integration_points=64):
        r"""Return the covariance matrix, with shape ``(dimension, dimension)``.

        .. math::

            \mathrm{Cov}(X)=\mathbb E[(X-\mathbb E[X])(X-\mathbb E[X])^\top].

        Reference-cell covariances are exact. Mapped covariances use fixed
        integration points; ``n_integration_points`` sets their number per cell.
        """
        if self._forward is None:
            mean = self.mean()
            second = np.empty((self.dimension, self.dimension))
            for row in range(self.dimension):
                for column in range(row, self.dimension):
                    order = [0] * self.dimension
                    order[row] += 1
                    order[column] += 1
                    value = self._reference_moment(tuple(order))
                    second[row, column] = second[column, row] = value
            return second - np.outer(mean, mean)
        points, weights = self._integration_points(n_integration_points)
        mean = np.average(points, axis=0, weights=weights)
        centered = points - mean
        return (centered * weights[:, None]).T @ centered

    def entropy(self, *, n_integration_points=64):
        r"""Return differential entropy ``-E[log(pdf(X))]``, in nats.

        .. math::

            h(X)=-\int p(x)\log p(x)\,dx.

        Reference-cell mixtures use an exact formula, including the cell weights.
        Mapped laws require ``backward`` and integrate ``-logpdf`` using
        ``n_integration_points`` fixed points per cell.
        """
        if self._forward is not None:
            if self._backward is None:
                raise TypeError("entropy requires an inverse map and its log-Jacobian")
            return self.expect(
                lambda points: -self.logpdf(points),
                n_integration_points=n_integration_points,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            mixing = -np.where(
                self.weights > 0.0,
                self.weights * np.log(self.weights),
                0.0,
            ).sum()
        return self.weights @ self.reference._cell_entropies[self.cells] + mixing

    def _reference_moment(self, order):
        values = self.reference._cell_moments(order)[self.cells]
        return np.average(values, weights=self.weights)

    def _map(self, forward, backward=None):
        """Apply ``forward`` after the existing map; combine their inverses for density."""

        def mapped(points, labels):
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
