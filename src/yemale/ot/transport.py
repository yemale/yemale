"""Exact candidate-augmented transport."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from yemale._array import readonly, restore, rows, scalars

from .reference import Reference
from .reference import reference as make_reference


@dataclass(frozen=True, eq=False, repr=False)
class Transport:
    """Exact candidate-augmented assignment and its source cells."""

    _target: np.ndarray
    _target_center: np.ndarray
    _centered_target: np.ndarray
    reference: Reference | None
    _source_center: np.ndarray
    _source_scale: float
    _affine_offsets: np.ndarray
    _assignment_tree: tuple[np.ndarray, np.ndarray, int]
    _one_dimensional_lookup: tuple[np.ndarray, np.ndarray] | None
    _normalized_sources: np.ndarray

    def __repr__(self):
        n, dimension = self._normalized_sources.shape
        return (
            f"Transport(n={n}, dimension={dimension}, "
            f"reference={self.reference is not None})"
        )

    @cached_property
    def _ranks(self):
        if self.reference is not None:
            return self.reference.ranks
        return readonly(np.linalg.norm(self._target, axis=1))

    @cached_property
    def _signs(self):
        if self.reference is not None:
            return self.reference.signs
        return readonly(
            np.divide(
                self._target,
                self._ranks[:, None],
                out=np.zeros_like(self._target),
                where=self._ranks[:, None] > 0,
            )
        )

    @cached_property
    def _source_potential(self):
        # Used only by the smooth inverse warm start.
        source_cells = np.argsort(self._assignment_tree[0])[:-1]
        value = (self._normalized_sources * self._centered_target[source_cells]).sum(
            axis=1
        )
        return readonly(value - self._affine_offsets[source_cells])

    @cached_property
    def _inverse_domain(self):
        # Shared by smooth inverse and pullback views at every temperature.
        centered = self._centered_target
        scale = np.abs(centered).max()
        dimension = self._target.shape[1]
        if scale == 0 or np.linalg.matrix_rank(centered / scale) < dimension:
            raise ValueError(
                "smooth inverse requires targets that affinely span the space"
            )
        matrix = np.ones((dimension + 1, len(self._target) + 1))
        matrix[:-1, :-1] = (centered / scale).T
        matrix[:-1, -1] = 0
        return readonly(matrix), self._target_center, scale

    def label(self, point):
        """Return the assigned cell label for each candidate point."""
        labels, shape = self._labels(point)
        return restore(labels, shape)

    def __call__(self, point) -> np.ndarray:
        """Return assigned target centres for a point ``(d,)`` or batch ``(..., d)``."""
        labels, shape = self._labels(point)
        return restore(self._target[labels], shape)

    def rank(self, point) -> np.ndarray:
        labels, shape = self._labels(point)
        return restore(self._ranks[labels], shape)

    def sign(self, point) -> np.ndarray:
        labels, shape = self._labels(point)
        return restore(self._signs[labels], shape)

    def evaluate(self, point):
        """Return label, target, rank, sign and potential in one dictionary."""
        points, shape = self._query(point)
        labels = self._maximize(points)
        target = self._target[labels]
        return {
            "label": restore(labels, shape),
            "target": restore(target, shape),
            "rank": restore(self._ranks[labels], shape),
            "sign": restore(self._signs[labels], shape),
            "potential": restore(self._potential_from_labels(points, labels), shape),
        }

    def potential(self, point):
        """Evaluate the convex potential in original source coordinates."""
        normalized, shape = self._query(point)
        labels = self._maximize(normalized)
        return restore(self._potential_from_labels(normalized, labels), shape)

    def _potential_from_labels(self, points, labels):
        value = (points * self._centered_target[labels]).sum(axis=1)
        value -= self._affine_offsets[labels]
        return self._source_scale * (value + points @ self._target_center)

    def halfspaces(self, label):
        """Return A, b for the closed source cell {z: A @ z <= b}."""
        if not isinstance(label, (int, np.integer)) or not 0 <= label < len(
            self._target
        ):
            raise ValueError(
                f"label must be an integer in [0, {len(self._target) - 1}]; got {label!r}"
            )
        keep = np.arange(len(self._target)) != label
        matrix = self._target[keep] - self._target[label]
        offset = self._source_scale * (
            self._affine_offsets[keep] - self._affine_offsets[label]
        )
        return matrix, matrix @ self._source_center + offset

    def assignment(self, point) -> np.ndarray:
        """Return independent n + 1 assignments; one candidate returns a vector."""
        labels, shape = self._labels(point)
        from ._core import lap

        if len(labels) != 1:
            return restore(lap.assignments(*self._assignment_tree, labels), shape)
        inverse = lap.assign(*self._assignment_tree, int(labels[0]))
        assignment = np.empty_like(inverse)
        assignment[inverse] = np.arange(len(inverse))
        return assignment

    def law(self, point):
        """Return the conditional reference law of one candidate."""
        from .law import Law

        reference = self._require_reference()
        labels, _ = self._labels(point)
        if len(labels) != 1:
            raise ValueError(
                f"law expects one point; got {len(labels)}. Call law once per point."
            )
        return Law(reference, labels)

    def extend(self, filling, *, inverse=None, inverse_logabsdet=None):
        """Fill assigned source cells through Q_j, with equal cell masses.

        ``filling(labels, points)`` maps ``(q,)``, ``(q, d)`` to ``(q, d)``,
        preserving ``self.label(filling(labels, points)) == labels`` almost surely.
        Density needs a one-to-one ``inverse(labels, points)`` of shape ``(q, d)``
        and ``inverse_logabsdet(labels, points)``: a scalar, ``(q,)``, or ``(q, 1)``.
        The filling is an explicit modeling choice.
        """
        from .law import Law

        reference = self._require_reference()
        if (inverse is None) != (inverse_logabsdet is None):
            raise ValueError("inverse and inverse_logabsdet must be supplied together")
        backward = None
        if inverse is not None:

            def backward(value):
                labels, _ = self._labels(value)
                target, _ = rows(inverse(labels, value), reference.dimension, "inverse")
                if len(target) != len(value):
                    raise ValueError(
                        f"inverse must return {len(value)} points; got {len(target)}"
                    )
                jacobian = scalars(
                    inverse_logabsdet(labels, value), len(value), "inverse_logabsdet"
                ).copy()
                # The inverse must return to this branch, not merely any reference cell.
                jacobian[reference.locate(target) != labels] = -np.inf
                return target, jacobian

        return Law(reference, forward=filling, backward=backward)

    def smooth(self, temperature=None):
        """Return a smooth view; None uses temperature 0.05 times the source scale."""
        from .smoothing import SmoothMap

        tau = 0.05 if temperature is None else float(temperature) / self._source_scale
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("temperature must be positive and finite")
        return SmoothMap(self, tau)

    def _require_reference(self):
        if self.reference is None:
            raise ValueError(
                "this operation requires reference cells; pass a Reference as target"
            )
        return self.reference

    def _query(self, point):
        points, shape = rows(point, self._target.shape[1])
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = (points - self._source_center) / self._source_scale
        if not np.isfinite(normalized).all():
            raise ValueError("point must be finite in normalized source coordinates")
        return normalized, shape

    def _labels(self, point) -> tuple[np.ndarray, tuple[int, ...]]:
        normalized, shape = self._query(point)
        return self._maximize(normalized), shape

    def _maximize(self, normalized_points: np.ndarray) -> np.ndarray:
        if self._one_dimensional_lookup is None:
            from ._core import hard

            return hard.max_affine(
                normalized_points, self._centered_target, self._affine_offsets
            )
        sorted_source, target_order = self._one_dimensional_lookup
        position = np.searchsorted(sorted_source, normalized_points[:, 0], side="left")
        labels = target_order[position]
        candidates = np.flatnonzero(position < len(sorted_source))
        tied = candidates[
            normalized_points[candidates, 0] == sorted_source[position[candidates]]
        ]
        for row in tied:
            stop = np.searchsorted(
                sorted_source, normalized_points[row, 0], side="right"
            )
            labels[row] = target_order[position[row] : stop + 1].min()
        return labels


def fit(source, *, target=None) -> Transport:
    """Fit source points to a target or the canonical reference.

    A supplied target has shape ``(n + 1, d)`` and is expressed in the
    centred, globally scaled coordinates of the source points.
    """
    source = np.ascontiguousarray(source, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(
            "source must be 2-D of shape (n, d); "
            "for 1-D data pass np.asarray(source)[:, None]"
        )
    n, dimension = source.shape
    if n == 0:
        raise ValueError("source must contain at least one point")
    if dimension == 0:
        raise ValueError("source must have at least one dimension")
    if not np.isfinite(source).all():
        raise ValueError("source must be finite")
    reference = make_reference(n, dimension) if target is None else target
    if isinstance(reference, Reference):
        if reference.n != n or reference.dimension != dimension:
            raise ValueError(
                f"reference must match source (n={n}, dimension={dimension}); "
                f"got (n={reference.n}, dimension={reference.dimension}). "
                "Omit target to use the matching n + 1 reference cells."
            )
        target = reference.centers
    else:
        reference = None
        target = readonly(target)
        expected_target_shape = (n + 1, dimension)
        if target.shape != expected_target_shape:
            raise ValueError(
                f"target must have shape {expected_target_shape}, got {target.shape}"
            )
        if not np.isfinite(target).all():
            raise ValueError("target must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        source_center = source.mean(axis=0)
        normalized_source = source - source_center
        magnitude = np.abs(normalized_source).max()
        source_scale = (
            float(
                magnitude * (np.linalg.norm(normalized_source / magnitude) / np.sqrt(n))
            )
            if magnitude
            else 1.0
        )
    if not np.isfinite(source_scale) or source_scale <= 0:
        raise ValueError("source cannot be centered and scaled to finite coordinates")
    normalized_source /= source_scale

    # Row/column constants in squared distances cannot affect a full assignment.
    # Use only -<x, m> to avoid cancellation, and center targets for stable queries.
    with np.errstate(over="ignore", invalid="ignore"):
        target_center = target.mean(axis=0)
        assignment_target = target - target_center
        finite_norms = np.isfinite(np.sum(target * target, axis=1)).all()
    if not finite_norms or not np.isfinite(assignment_target).all():
        raise ValueError("target squared norms must be finite")

    from ._core import lap

    if dimension == 1:
        leave_one_costs, base_assignment, predecessor, free_target, lookup = (
            lap.solve_1d(normalized_source[:, 0], assignment_target[:, 0])
        )
        if np.any(np.diff(target[lookup[1], 0]) == 0):
            lookup = None
    else:
        cost = np.empty((n + 1, n + 1))
        np.matmul(normalized_source, assignment_target.T, out=cost[:n])
        cost[:n] *= -1.0
        row_bias = cost[:n].min(axis=1)
        cost[:n] -= row_bias[:, None]
        cost[n] = 0.0
        # Undo row reduction and add scaled source norms only in JV's initial scan.
        row_bias += 0.5 * np.abs(assignment_target).max() * np.sum(
            normalized_source * normalized_source, axis=1
        )
        leave_one_costs, base_assignment, predecessor, free_target = lap.solve(
            cost, row_bias
        )
        lookup = None

    with np.errstate(over="ignore", invalid="ignore"):
        affine_offsets = leave_one_costs - leave_one_costs.mean()
    if not np.isfinite(affine_offsets).all():
        raise ValueError(
            "assignment costs must remain finite; reduce target magnitudes"
        )

    return Transport(
        _target=target,
        _target_center=readonly(target_center),
        _centered_target=readonly(assignment_target),
        reference=reference,
        _source_center=readonly(source_center),
        _source_scale=source_scale,
        _affine_offsets=readonly(affine_offsets),
        _assignment_tree=(base_assignment, predecessor, free_target),
        _one_dimensional_lookup=lookup,
        _normalized_sources=readonly(normalized_source),
    )
