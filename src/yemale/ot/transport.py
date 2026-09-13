"""Exact candidate-augmented transport."""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from yemale._array import readonly, restore, rows, scalars

from .reference import Reference
from .reference import reference as make_reference


@dataclass(frozen=True, eq=False, repr=False)
class Transport:
    """Assign each candidate to a target after appending it to the fitted source.

    Build with ``fit``. Each query is a separate ``n + 1`` assignment.
    Queries use original source coordinates: ``(d,)`` or ``(..., d)``.
    Map and sign outputs keep the coordinate axis; label, rank and potential
    outputs do not. In one dimension, scalar and flat-vector queries also work.
    """

    _target: np.ndarray
    _target_center: np.ndarray
    _centered_target: np.ndarray
    reference: Reference | None
    _source_center: np.ndarray
    _source_scale: float
    _affine_offsets: np.ndarray
    _query_limit: float
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
    def phi(self):
        """Branch offsets in ``max_j <z, m_j> - phi[j]``, in source coordinates."""
        with np.errstate(over="ignore", invalid="ignore"):
            value = (
                self._target @ self._source_center
                + self._source_scale * self._affine_offsets
            )
        if not np.isfinite(value).all():
            raise ValueError(
                "phi is not representable; reduce source or target magnitudes"
            )
        return readonly(value)

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
        # Used to choose starting points for the smooth inverse.
        source_cells = np.argsort(self._assignment_tree[0])[:-1]
        value = (self._normalized_sources * self._centered_target[source_cells]).sum(
            axis=1
        )
        return readonly(value - self._affine_offsets[source_cells])

    @cached_property
    def _inverse_domain(self):
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
        """Return the assigned target index, from 0 to n, for each candidate."""
        labels, shape = self._labels(point)
        return restore(labels, shape)

    def __call__(self, point) -> np.ndarray:
        """Return assigned target centres for a point ``(d,)`` or batch ``(..., d)``."""
        labels, shape = self._labels(point)
        return restore(self._target[labels], shape)

    def rank(self, point) -> np.ndarray:
        """Return the assigned target's radius (distance from the origin)."""
        labels, shape = self._labels(point)
        return restore(self._ranks[labels], shape)

    def sign(self, point) -> np.ndarray:
        """Return the assigned target's unit direction; a zero target gives zero."""
        labels, shape = self._labels(point)
        return restore(self._signs[labels], shape)

    def evaluate(self, point):
        """Return label, target, rank, sign and potential in one dictionary.

        Use this when several outputs are needed; their target lookup is shared.
        Values match the corresponding methods, with ``target`` from ``self(point)``.
        """
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
        """Return the convex potential whose gradient is this map away from ties.

        Values use original source coordinates, with one scalar per candidate.
        """
        normalized, shape = self._query(point)
        labels = self._maximize(normalized)
        return restore(self._potential_from_labels(normalized, labels), shape)

    def _potential_from_labels(self, points, labels):
        value = (points * self._centered_target[labels]).sum(axis=1)
        value -= self._affine_offsets[labels]
        return self._scale_potential(points, value)

    def _scale_potential(self, points, value):
        with np.errstate(over="ignore", invalid="ignore"):
            value = self._source_scale * (value + points @ self._target_center)
        if not np.isfinite(value).all():
            raise ValueError(
                "potential is not representable; reduce source or target magnitudes"
            )
        return value

    def halfspaces(self, label):
        """Return A, b describing the source cell, including its boundary: A @ z <= b."""
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

    def quantile_region(self, coverage):
        """Return the smallest quantile region reaching the requested ``coverage``.

        Its achieved coverage can be larger because cells at the same radius are
        included together. Requires the default reference-cell construction.
        """
        reference = self._require_reference()
        try:
            coverage = float(coverage)
        except (TypeError, ValueError):
            raise ValueError(
                f"coverage must be between 0 and 1; got {coverage!r}"
            ) from None
        if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError(f"coverage must be between 0 and 1; got {coverage!r}")
        if coverage == 0.0:
            radius = 0.0
        else:
            position = math.ceil(coverage * (reference.n + 1)) - 1
            radius = float(np.partition(reference.ranks, position)[position])
        tolerance = 16 * np.finfo(float).eps * max(1.0, abs(radius))
        selected = np.flatnonzero(reference.ranks <= radius + tolerance)
        if len(selected):
            radius = float(reference.ranks[selected].max())
        labels = readonly(selected, dtype=np.int64)
        return QuantileRegion(self, radius, labels)

    def assignment(self, point) -> np.ndarray:
        """Return target indices for all source points followed by the candidate.

        A point ``(d,)`` returns ``(n + 1,)``. Batches ``(..., d)`` return
        ``(..., n + 1)`` independent assignments, keeping singleton batch axes.
        The output uses ``8 * q * (n + 1)`` bytes for ``q`` candidates;
        use ``label`` if only each candidate's target index is needed.
        """
        labels, shape = self._labels(point)
        from ._core import lap

        if len(labels) != 1:
            return restore(lap.assignments(*self._assignment_tree, labels), shape)
        inverse = lap.assign(*self._assignment_tree, int(labels[0]))
        assignment = np.empty_like(inverse)
        assignment[inverse] = np.arange(len(inverse))
        return restore(assignment[None], shape)

    def reference_distribution(self, point):
        """Return the reference distribution within one candidate's assigned cell.

        Samples are reference points, not predictions in source coordinates.
        Requires a Reference target.
        """
        from .law import Law

        reference = self._require_reference()
        labels, _ = self._labels(point)
        if len(labels) != 1:
            raise ValueError(
                f"reference_distribution expects one point; got {len(labels)}. "
                "Call reference_distribution once per point."
            )
        return Law(reference, labels)

    def predictive_distribution(
        self, map_from_reference, *, inverse=None, inverse_logabsdet=None
    ):
        """Create a predictive distribution.

        The transport divides source space into one cell per target. This method
        gives every cell equal probability. ``map_from_reference`` chooses how
        probability is distributed within each cell.

        Args:
            map_from_reference: Called with reference points and their cell labels.
                It returns one predictive value in the matching assigned cell for
                each input row. It must be one-to-one for density evaluation.
            inverse: Optional inverse called as
                ``inverse(source_points, cell_labels)``. Supply with
                ``inverse_logabsdet`` for density and entropy.
            inverse_logabsdet: Optional callable with the same inputs as ``inverse``.
                It returns the inverse Jacobian's log absolute determinant.

        Notes:
            ``map_from_reference`` must map reference cell j into source cell j.
            Callbacks must treat input points as read-only.
        """
        from .law import Law

        reference = self._require_reference()
        if (inverse is None) != (inverse_logabsdet is None):
            raise ValueError("inverse and inverse_logabsdet must be supplied together")
        backward = None
        if inverse is not None:

            def backward(value):
                labels, _ = self._labels(value)
                target, _ = rows(inverse(value, labels), reference.dimension, "inverse")
                if len(target) != len(value):
                    raise ValueError(
                        f"inverse must return {len(value)} points; got {len(target)}"
                    )
                jacobian = scalars(
                    inverse_logabsdet(value, labels), len(value), "inverse_logabsdet"
                ).copy()
                # A source cell has density only through its matching reference cell.
                jacobian[reference.locate(target) != labels] = -np.inf
                return target, jacobian

        def forward(labels, points):
            return map_from_reference(points, labels)

        return Law(reference, forward=forward, backward=backward)

    def smooth(self, temperature=None):
        """Return a SmoothMap that blends target centres instead of choosing one.

        ``temperature`` must be positive, in the units of ``potential``.
        Larger values give a smoother map. ``None`` uses 0.05 times the source
        scale described in ``fit``.
        """
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
        magnitude = np.abs(normalized).max(initial=0.0)
        if not np.isfinite(magnitude):
            raise ValueError(
                "point must be finite and representable relative to the fitted source scale"
            )
        if magnitude > self._query_limit:
            raise ValueError(
                "point is too large for finite target comparisons; reduce its magnitude"
            )
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


@dataclass(frozen=True, eq=False, repr=False)
class QuantileRegion:
    """A finite-sample union of source cells selected by reference radius."""

    _transport: Transport
    radius: float
    labels: np.ndarray

    def __repr__(self):
        return (
            f"QuantileRegion(coverage={self.coverage:.6g}, radius={self.radius:.6g}, "
            f"cell_count={len(self.labels)})"
        )

    @property
    def coverage(self):
        """Finite-sample coverage of this region."""
        return len(self.labels) / len(self._transport._target)

    def contains(self, point):
        """Return whether each point belongs to the region."""
        return self._transport.rank(point) <= self.radius

    def halfspaces(self):
        """Return ``(A, b)`` for every closed source cell in the region."""
        return tuple(self._transport.halfspaces(int(label)) for label in self.labels)


def fit(source, *, target=None) -> Transport:
    """Fit a reusable Transport from n observations to n + 1 targets.

    Each later query supplies the extra candidate in original source coordinates.

    Args:
        source: Finite observations of shape ``(n, d)``. Use ``(n, 1)`` for scalar data.
        target: Matching Reference or finite target array of shape ``(n + 1, d)``.
            Defaults to ``reference(n, d)``. Target coordinates are used as supplied;
            they need not match the source's centre or scale. Array targets support
            assignment and smoothing, but do not provide reference-cell laws.

    Notes:
        For ``d > 1``, fitting allocates a temporary cost matrix of
        ``8 * (n + 1)**2`` bytes. The one-dimensional solver avoids this matrix.
        Sources are centred and divided by their root-mean-square distance from
        the mean, using one scale for all coordinates (1 for identical sources).
        Assignment labels are invariant to a common target translation or positive
        scalar rescaling, up to floating-point precision. Returned targets retain
        the supplied values.
    """
    source = np.ascontiguousarray(source, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(
            f"source must be 2-D of shape (n, d); got {source.shape}. "
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
                f"target must have shape {expected_target_shape}, got {target.shape}; "
                "use n + 1 targets for n source points and one candidate"
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

    # Center targets to keep their common offset out of the dot products.
    with np.errstate(over="ignore", invalid="ignore"):
        target_center = target.mean(axis=0)
        assignment_target = target - target_center
        finite_norms = np.isfinite(np.sum(target * target, axis=1)).all()
    if not finite_norms or not np.isfinite(assignment_target).all():
        raise ValueError(
            "target centered coordinates and squared norms must be finite; "
            "reduce target magnitudes"
        )

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
        # Undo row reduction and seed nearest-source matches to targets normalized
        # by their largest absolute coordinate. This scale choice preserves the seed
        # under target rescaling; 0.5 comes from expanding squared distances.
        # Only initialization uses this bias, not the assignment objective.
        row_bias += (
            0.5
            * np.abs(assignment_target).max()
            * np.sum(normalized_source * normalized_source, axis=1)
        )
        leave_one_costs, base_assignment, predecessor, free_target = lap.solve(
            cost, row_bias
        )
        lookup = None

    # Cross-term costs cancel the target-norm term in the potential. Row reduction
    # shifts all leave-one costs equally; centering fixes the additive constant.
    with np.errstate(over="ignore", invalid="ignore"):
        affine_offsets = leave_one_costs - leave_one_costs.mean()
    if not np.isfinite(affine_offsets).all():
        raise ValueError(
            "assignment costs must remain finite; reduce target magnitudes"
        )

    # Bound each affine sum within floating-point range, allowing for rounding.
    target_bound = np.abs(assignment_target).sum(axis=1).max()
    with np.errstate(over="ignore", divide="ignore"):
        query_limit = (
            (np.finfo(float).max - np.abs(affine_offsets).max()) / 2 / target_bound
        )

    return Transport(
        _target=target,
        _target_center=readonly(target_center),
        _centered_target=readonly(assignment_target),
        reference=reference,
        _source_center=readonly(source_center),
        _source_scale=source_scale,
        _affine_offsets=readonly(affine_offsets),
        _query_limit=float(query_limit),
        _assignment_tree=(base_assignment, predecessor, free_target),
        _one_dimensional_lookup=lookup,
        _normalized_sources=readonly(normalized_source),
    )
