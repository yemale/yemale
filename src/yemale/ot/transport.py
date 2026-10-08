"""Exact candidate-augmented transport."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from yemale._array import batch, readonly, restore, rows, scalars

from .reference import Reference
from .reference import reference as make_reference


@dataclass(frozen=True, eq=False, repr=False)
class Transport:
    """Assign each candidate to a target after appending it to the fitted source.

    Create with ``ot.fit(source)``. Each query is a separate ``n + 1`` assignment.
    Queries use original source coordinates: ``(d,)`` or ``(..., d)``.
    Map and sign outputs keep the coordinate axis; label, rank and potential
    outputs do not. In one dimension, scalar and flat-vector queries also work.

    Attributes:
        reference: Reference used to construct the targets, or None for array targets.
    """

    # Internal queries use x = (z - source_center) / source_scale.
    # Internal branches are <x, centered_target[j]> - affine_offsets[j].
    _target: np.ndarray
    _target_center: np.ndarray
    _centered_target: np.ndarray
    reference: Reference | None
    _source_center: np.ndarray
    _source_scale: float
    _affine_offsets: np.ndarray
    _query_limit: float
    # (target-to-row assignment, vacancy predecessors, auxiliary-row target)
    _assignment_tree: tuple[np.ndarray, np.ndarray, int]
    # (sorted normalized sources, target labels in sorted-target order)
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
        """Branch offsets in ``potential``, with shape ``(n + 1,)``.

        Use original source coordinates. A common additive constant is fixed
        by the fit; it does not affect assignments or derivatives.
        """
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
        """Hard potential values at the fitted sources in internal coordinates."""
        target_to_row, _, _ = self._assignment_tree
        source_cells = np.argsort(target_to_row)[:-1]  # Omit the auxiliary row.
        value = (self._normalized_sources * self._centered_target[source_cells]).sum(
            axis=1
        )
        return readonly(value - self._affine_offsets[source_cells])

    def label(self, point):
        """Return the assigned target index, from 0 to n, for each candidate."""
        labels, shape = self._labels(point)
        return restore(labels, shape)

    def __call__(self, point) -> np.ndarray:
        r"""Map source points to their assigned centres in target coordinates.

        .. math::

            T(z)=m_{k(z)},\qquad k(z)=\sigma_z(n+1).

        Accept a point ``(d,)`` or batch ``(..., d)`` and keep its shape.
        Here sigma_z is the augmented assignment from ``fit``, and m_j are
        the target centres. With the default reference, outputs lie in the unit ball.
        """
        labels, shape = self._labels(point)
        return restore(self._target[labels], shape)

    def rank(self, point) -> np.ndarray:
        r"""Return the center-outward rank: the radius of the assigned target.

        .. math::

            \mathrm{Rank}(z)=\|T(z)\|.

        Smaller radii indicate more central reference cells. Default-reference
        ranks take discrete values in ``[0, 1)``; the minimum need not be zero.
        Custom targets retain their own radii. Return one value per candidate.
        """
        labels, shape = self._labels(point)
        return restore(self._ranks[labels], shape)

    def sign(self, point) -> np.ndarray:
        r"""Return the assigned target's unit direction; a zero target gives zero.

        .. math::

            \mathrm{Sign}(z)=T(z)/\|T(z)\|\qquad\text{when }T(z)\ne0.

        This direction is in target space, not from the source mean to the point.
        Return one vector per candidate, keeping the coordinate axis.
        """
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
        r"""Return the convex potential whose gradient is this map away from ties.

        .. math::

            \Phi(z)=\max_{1\leq j\leq n+1}\{\langle z,m_j\rangle-\phi_j\}.

        Here m_j are the targets and phi_j are the offsets in ``phi``.
        Accept ``(d,)`` or ``(..., d)`` in original source coordinates;
        return one scalar per candidate.
        """
        normalized, shape = self._query(point)
        labels = self._maximize(normalized)
        return restore(self._potential_from_labels(normalized, labels), shape)

    def _potential_from_labels(self, points, labels):
        value = self._internal_potential(points, labels)
        return self._scale_potential(points, value)

    def _internal_potential(self, points, labels):
        """Evaluate selected affine pieces in normalized, centered coordinates."""
        value = (points * self._centered_target[labels]).sum(axis=1)
        value -= self._affine_offsets[labels]
        return value

    def _scale_potential(self, points, value):
        """Convert the centered-target potential at normalized points to source units."""
        # Phi(z) = source_scale * (internal_phi(x) + <x, target_center>).
        with np.errstate(over="ignore", invalid="ignore"):
            value = self._source_scale * (value + points @ self._target_center)
        if not np.isfinite(value).all():
            raise ValueError(
                "potential is not representable; reduce source or target magnitudes"
            )
        return value

    def halfspaces(self, label):
        r"""Return A, b describing the closed source cell: ``A @ z <= b``.

        .. math::

            V_j=\bigcap_{k\ne j}\{z:\langle z,m_k-m_j\rangle
            \leq\phi_k-\phi_j\}.

        For a label in ``0, ..., n``, return shapes ``(n, d)`` and ``(n,)``.
        Closed cells share boundaries; ``label(point)`` assigns boundary points.
        """
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

    def region(self, reference_set):
        r"""Return the source region mapped into a fixed target set B.

        .. math:: \mathcal Q_T(B)=\{z:T(z)\in B\}.

        ``reference_set`` takes target points ``(q, d)`` and returns one Boolean
        per point. For the default 1-D reference, ``u = 2*p - 1`` converts
        probability levels to reference coordinates: [0.05, 0.95] becomes
        [-0.9, 0.9]. Use ``quantile_region(0.9)`` for central 90% coverage.
        For a data-independent B, marginal coverage is the fraction of target
        centres in B, not generally its continuous reference probability.
        """
        if not callable(reference_set):
            raise TypeError(
                "reference_set must return one Boolean per target point; "
                "for a coverage level, use quantile_region(0.9)"
            )
        accepted = np.asarray(reference_set(self._target))
        if accepted.shape != (len(self._target),) or accepted.dtype != bool:
            raise ValueError(
                f"reference_set must return {len(self._target)} Booleans; "
                f"got shape {accepted.shape}, dtype {accepted.dtype}. "
                "For one coordinate use points[:, 0]."
            )
        return Region(self, readonly(accepted, dtype=bool))

    def quantile_region(self, coverage, *, randomized=True, rng=None):
        r"""Return a quantile region with the requested marginal coverage.

        .. math::

            W_j\sim\nu(\,\cdot\mid L_j),\qquad
            \Omega=\bigcup_{j:\|W_j\|\leq 1-\alpha}V_j^\dagger.

        Args:
            coverage: Probability in [0, 1], equal to 1 - alpha.
            randomized: Draw once per reference cell, then keep membership fixed.
                Defaults to True. False selects whole shells of target centres.
            rng: Seed or NumPy Generator for reproducible randomization.

        Under exchangeability and almost-sure uniqueness of the augmented
        assignment, coverage averages over observations and randomization, not
        one fitted sample. The selected-cell fraction can differ from the request.
        With ``randomized=False``, return the smallest region {z: ||T(z)|| <= r}
        reaching coverage; equal-radius cells enter together and can overshoot.
        Requires reference cells. Return a QuantileRegion.
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
        if not isinstance(randomized, (bool, np.bool_)):
            raise TypeError("randomized must be True or False")
        if randomized:
            radius = float(coverage)
            lower, upper, _ = reference._partition
            # Membership uses only radius, so no angular draw is needed.
            # The 1-D partition uses signed intervals; abs gives the radius.
            radii = np.abs(np.random.default_rng(rng).uniform(lower, upper))
            accepted = radii <= radius
            if radius == 0 or radius == 1:
                accepted[:] = bool(radius)
        elif rng is not None:
            raise ValueError("rng requires randomized=True")
        elif coverage == 0.0:
            radius = 0.0
            accepted = np.zeros(reference.n + 1, dtype=bool)
        else:
            cell_count = reference.n + 1
            required = int(coverage * cell_count)
            if required / cell_count < coverage:
                required += 1
            position = required - 1
            radius = float(np.partition(reference.ranks, position)[position])
            tolerance = 16 * np.finfo(float).eps * max(1.0, abs(radius))
            accepted = reference.ranks <= radius + tolerance
            radius = float(reference.ranks[accepted].max())
        return QuantileRegion(self, readonly(accepted, dtype=bool), radius, randomized)

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
        target_to_row = lap.assign(*self._assignment_tree, int(labels[0]))
        assignment = np.empty_like(target_to_row)
        assignment[target_to_row] = np.arange(len(target_to_row))
        return restore(assignment[None], shape)

    def reference_distribution(self, point):
        r"""Return the reference distribution within one candidate's assigned cell.

        .. math::

            K(z,\cdot)=\nu(\,\cdot\mid L_{k(z)}).

        Samples and summaries use reference coordinates within that cell.
        Accept one candidate; return a Law. Requires a Reference target.
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
        r"""Create a predictive Law by choosing how to fill each source cell.

        Write Q_j for ``map_from_reference`` on reference cell L_j.
        Map each L_j into its matching source cell; each gets probability 1 / (n + 1).

        .. math::

            \Pi=\frac1{n+1}\sum_{j=1}^{n+1}
            (Q_j)_\#\nu(\,\cdot\mid L_j).

        Draw a cell uniformly, draw U within it, and return Q_j(U).
        The supplied Q_j specifies how probability fills its source cell.

        Args:
            map_from_reference: Called as ``map_from_reference(points, labels)``
                with reference points ``(q, d)`` and cell labels ``(q,)``. Returns
                points ``(q, d)`` in the matching source cells.
                For density, use a differentiable, one-to-one map with nonsingular
                Jacobian.
            inverse: Optional inverse called as
                ``inverse(source_points, cell_labels)``. Supply with
                ``inverse_logabsdet`` for density and entropy. Returns ``(q, d)``.
            inverse_logabsdet: Optional callable with the same inputs as ``inverse``.
                Returns the inverse Jacobian's log absolute determinant, as a
                scalar, ``(q,)``, or ``(q, 1)``.

        Notes:
            Callbacks must treat input points as read-only.
            Outside the mapped support, the inverse may return finite placeholders
            paired with a ``-inf`` log determinant.
        """
        from .law import Law

        reference = self._require_reference()
        if not callable(map_from_reference):
            raise TypeError("map_from_reference must be callable as (points, labels)")
        for name, callback in (
            ("inverse", inverse),
            ("inverse_logabsdet", inverse_logabsdet),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable as (points, labels)")
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

        return Law(reference, forward=map_from_reference, backward=backward)

    @cached_property
    def _default_temperature(self):
        """Assignment-margin temperature in normalized source coordinates."""
        sources = self._normalized_sources
        n, dimension = sources.shape
        radii = self._ranks
        tolerance = 16 * np.finfo(float).eps * max(1.0, radii.max())
        if np.ptp(radii) <= tolerance:
            return 0.05

        generator = np.random.default_rng(0)
        size = max(4000, n)
        probes = sources[generator.integers(0, n, size=size)]
        if n > 1:
            covariance = np.atleast_2d(np.cov(sources, rowvar=False))
            covariance += 1e-9 * np.eye(dimension)
            probes += 0.02 * generator.multivariate_normal(
                np.zeros(dimension), covariance, size
            )

        gaps = np.empty(size)
        # Bound query-target storage; only the scalar temperature is cached.
        for start in range(0, size, 256):
            logits = probes[start : start + 256] @ self._centered_target.T
            logits -= self._affine_offsets
            winners = logits.argmax(axis=1)
            best = logits[np.arange(len(logits)), winners]
            same_level = np.abs(radii[None, :] - radii[winners, None]) <= tolerance
            logits[same_level] = -np.inf
            gaps[start : start + len(logits)] = best - logits.max(axis=1)
        gaps = gaps[np.isfinite(gaps) & (gaps > 1e-10 / self._source_scale)]
        return 100 * float(np.median(gaps)) if len(gaps) else 0.05

    def smooth(self, temperature=None):
        """Return a SmoothMap that blends target centres instead of choosing one.

        ``temperature`` must be positive, in the units of ``potential``.
        Larger values give a smoother map. ``None`` uses 100 times the median
        winning margin over targets at a different radius, retaining margins
        greater than 1e-10 in potential units. Probes resample the sources with
        2% covariance-scaled Gaussian jitter (at least 4000 points, seed 0).
        The choice is cached; if no margin qualifies, use 0.05 times the source
        scale. An explicit temperature bypasses this calculation.
        """
        from .smoothing import SmoothMap

        tau = (
            self._default_temperature
            if temperature is None
            else float(temperature) / self._source_scale
        )
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
        """Return normalized rows and the original batch shape."""
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
        # In 1-D, sorted matching gives the candidate its insertion-position target.
        position = np.searchsorted(sorted_source, normalized_points[:, 0], side="left")
        labels = target_order[position]
        candidates = np.flatnonzero(position < len(sorted_source))
        tied = candidates[
            normalized_points[candidates, 0] == sorted_source[position[candidates]]
        ]
        # Equal source values allow every insertion slot across the tied run.
        for row in tied:
            stop = np.searchsorted(
                sorted_source, normalized_points[row, 0], side="right"
            )
            labels[row] = target_order[position[row] : stop + 1].min()
        return labels


@dataclass(frozen=True, eq=False, repr=False)
class Region:
    """Source points whose assigned targets belong to a chosen reference set.

    Create with ``transport.region(reference_set)``.
    """

    _transport: Transport
    _accepted: np.ndarray

    def __repr__(self):
        return f"Region(coverage={self.coverage:.6g}, cell_count={len(self.labels)})"

    @cached_property
    def labels(self):
        """Zero-based labels of the included cells."""
        return readonly(np.flatnonzero(self._accepted), dtype=np.int64)

    @property
    def coverage(self):
        """Fraction of reference cells included in the region.

        For a fixed reference set, this is marginal coverage under exchangeability
        and almost-sure uniqueness of the augmented assignment.
        """
        return np.count_nonzero(self._accepted) / len(self._accepted)

    def contains(self, point):
        """Return whether each point belongs to the region."""
        labels, shape = self._transport._labels(point)
        return restore(self._accepted[labels], shape)

    def select(self, points):
        """Return accepted observations in input order, preserving their shape.

        ``points`` has shape ``(q, d)``, or ``(q,)`` for scalar observations.
        """
        points = np.asarray(points)
        return points[self.contains(batch(points, "points"))]

    def halfspaces(self):
        """Return ``(A, b)`` for every closed source cell in the region."""
        return tuple(self._transport.halfspaces(int(label)) for label in self.labels)


@dataclass(frozen=True, eq=False, repr=False)
class QuantileRegion(Region):
    """Source cells selected by reference radius.

    Create with ``transport.quantile_region(coverage)``.
    Randomization is enabled by default.

    Attributes:
        radius: Reference-ball radius; applied to cell draws when randomized,
            and to target centres otherwise.
        randomized: Whether reference-cell draws selected the cells.
    """

    radius: float
    randomized: bool

    @property
    def coverage(self):
        """Marginal coverage, averaging over data and any region randomization.

        For a randomized region, the realized fraction ``len(labels) / (n + 1)``
        can differ from this probability.
        """
        return self.radius if self.randomized else super().coverage

    def __repr__(self):
        return (
            f"QuantileRegion(coverage={self.coverage:.6g}, radius={self.radius:.6g}, "
            f"cell_count={len(self.labels)}, randomized={self.randomized})"
        )


def fit(source, *, target=None) -> Transport:
    r"""Fit a reusable Transport from n observations to n + 1 targets.

    Each later query appends one candidate: zeta(z) = (Z_1, ..., Z_n, z).
    With target centres m_j, its assignment minimizes squared Euclidean cost:

    .. math::

        \sigma_z\in\arg\min_{\sigma\in\mathfrak S_{n+1}}
        \sum_{i=1}^{n+1}\|\zeta_i(z)-m_{\sigma(i)}\|^2.

    A query uses original source coordinates and does not solve a new assignment.

    Args:
        source: Finite observations of shape ``(n, d)`` or ``(n,)`` for scalar data.
            A dataset containing one vector has shape ``(1, d)``.
        target: Matching Reference or finite target array of shape ``(n + 1, d)``.
            Scalar targets may use ``(n + 1,)``.
            Defaults to ``reference(n, d)``. Target coordinates are used as supplied;
            they need not match the source's centre or scale. Array targets support
            assignment and smoothing, but do not provide reference-cell laws.

    Notes:
        For ``d > 1``, fitting allocates a temporary cost matrix of
        ``8 * (n + 1)**2`` bytes. The one-dimensional solver avoids this matrix.
        Sources are centred and divided by their root-mean-square distance from
        the mean, using one scale for all coordinates (1 for identical sources).
        Coordinates are not standardized separately; if needed, choose feature
        scales independently of calibration and apply them to sources and queries.
        Assignment labels are invariant to a common target translation or positive
        scalar rescaling, up to floating-point precision. Returned targets retain
        the supplied values.
    """
    source = np.ascontiguousarray(batch(source, "source"), dtype=np.float64)
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
        target = readonly(batch(target, "target"))
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
        source_scale = 1.0
        if magnitude:
            # Compute RMS distance without overflowing squared coordinates.
            scaled_rms = np.linalg.norm(normalized_source / magnitude) / np.sqrt(n)
            source_scale = float(magnitude * scaled_rms)
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
        # Repeated targets can tie across insertion slots; use affine label lookup.
        if np.any(np.diff(target[lookup[1], 0]) == 0):
            lookup = None
    else:
        # Use -<x, m>; full assignments differ from squared cost by constants.
        # Write directly into the real-source rows; the final row is auxiliary.
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
        seed_bias = row_bias  # Reuse its storage; row minima are no longer needed.
        seed_bias += (
            0.5
            * np.abs(assignment_target).max()
            * np.sum(normalized_source * normalized_source, axis=1)
        )
        leave_one_costs, base_assignment, predecessor, free_target = lap.solve(
            cost, seed_bias
        )
        lookup = None

    # For costs -<x, m>, let q_j be the leave-one cost.
    # Squared-distance leave-one costs are K - ||m_j||² + 2*q_j.
    # Thus affine offsets equal q_j up to a constant, including row reduction.
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
