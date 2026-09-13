"""Smooth transport, numerical inversion, and source distributions."""

from dataclasses import dataclass

import numpy as np

from yemale._array import restore, rows

from ._core import smooth
from .law import Law
from .transport import Transport


@dataclass(eq=False)
class SmoothMap:
    """A smooth transport that blends target centres instead of choosing one.

    Build with ``Transport.smooth``. Source queries follow Transport's shape
    conventions. ``inverse`` maps target points back to source coordinates.
    """

    _transport: Transport
    _tau: float

    def __repr__(self):
        temperature = self._tau * self._transport._source_scale
        return f"SmoothMap(transport={self._transport!r}, temperature={temperature})"

    def __call__(self, point):
        """Map each source point to a softmax average of target centres."""
        points, shape = self._transport._query(point)
        value = smooth.read(
            points,
            self._transport._centered_target,
            self._transport._affine_offsets,
            self._tau,
        )
        return restore(value + self._transport._target_center, shape)

    def potential(self, point):
        """Return the smooth potential in source coordinates; its gradient is this map."""
        points, shape = self._transport._query(point)
        value = smooth.potential(
            points,
            self._transport._centered_target,
            self._transport._affine_offsets,
            self._tau,
        )
        return restore(self._transport._scale_potential(points, value), shape)

    def _statistics(self, point):
        points, shape = self._transport._query(point)
        mapped, jacobian = smooth.map_jacobian(
            points,
            self._transport._centered_target,
            self._transport._affine_offsets,
            self._tau,
        )
        mapped += self._transport._target_center
        return mapped, jacobian / self._transport._source_scale, shape

    def jacobian(self, point):
        """Differentiate the map with respect to original source coordinates.

        Return ``(d, d)`` for one point or ``(..., d, d)`` for a batch.
        """
        _, jacobian, shape = self._statistics(point)
        return restore(jacobian, shape)

    def map_jacobian(self, point):
        """Return ``(self(point), self.jacobian(point))`` with shared computation."""
        mapped, jacobian, shape = self._statistics(point)
        return restore(mapped, shape), restore(jacobian, shape)

    def reference_distribution(self, point):
        """Return the reference-cell mixture weighted by the smooth map at one point.

        Requires a Reference target; samples stay in reference coordinates.
        """
        reference = self._transport._require_reference()
        points, _ = self._transport._query(point)
        if len(points) != 1:
            raise ValueError(
                f"reference_distribution expects one point; got {len(points)}. "
                "Call reference_distribution once per point."
            )
        weight = smooth.weights(
            points,
            self._transport._centered_target,
            self._transport._affine_offsets,
            self._tau,
        )[0]
        return Law(reference, weights=weight)

    def _check_targets(self, targets):
        if not np.isfinite(targets).all():
            raise ValueError("target must be finite")
        sites = self._transport._target
        matrix, center, scale = self._transport._inverse_domain
        message = "target must lie strictly inside the convex hull of target centers"
        if sites.shape[1] == 1:
            if np.any(targets <= sites.min()) or np.any(targets >= sites.max()):
                raise ValueError(message)
            return
        from scipy.optimize import linprog

        # Express the target as a weighted average of the centres and their mean.
        # Positive weight on the mean puts it strictly inside the convex hull.
        objective = np.zeros(len(sites) + 1)
        objective[-1] = -1.0
        for target in targets:
            with np.errstate(over="ignore", invalid="ignore"):
                rhs = (target - center) / scale
            if not np.isfinite(rhs).all():
                raise ValueError(message)
            result = linprog(
                objective,
                A_eq=matrix,
                b_eq=np.r_[rhs, 1.0],
                bounds=(0.0, None),
                method="highs",
                options={
                    "primal_feasibility_tolerance": 1e-9,
                    "dual_feasibility_tolerance": 1e-9,
                },
            )
            if not result.success or result.x[-1] <= 1e-8:
                raise ValueError(message)

    def inverse(self, target):
        """Map target points back to source coordinates.

        Targets must lie strictly inside the convex hull of the target centres.
        """
        sites = self._transport._centered_target
        targets, shape = rows(target, sites.shape[1], "target")
        self._check_targets(targets)
        # Keep the stopping tolerance independent of target units.
        scale = self._transport._inverse_domain[2]
        sites = sites / scale
        targets = (targets - self._transport._target_center) / scale
        offsets = self._transport._affine_offsets / scale
        temperature = self._tau / scale
        source_potential = self._transport._source_potential / scale
        starts = smooth.read(
            targets,
            self._transport._normalized_sources,
            source_potential,
            temperature,
        )
        result, residual = smooth.inverse(targets, starts, sites, offsets, temperature)
        failed = ~(residual <= 1e-9)
        if np.any(failed):
            # Solve at decreasing temperatures, using each solution to start the next.
            tau = max(temperature, float(np.ptp(offsets)), 0.05)
            trial = smooth.read(
                targets[failed],
                self._transport._normalized_sources,
                source_potential,
                tau,
            )
            for _ in range(64):
                trial, error = smooth.inverse(
                    targets[failed], trial, sites, offsets, tau
                )
                if not np.all(error <= 1e-9) or tau == temperature:
                    break
                tau = max(temperature, tau * 0.5)
            if tau != temperature or not np.all(error <= 1e-9):
                raise RuntimeError(
                    "smooth inverse did not converge at this temperature"
                )
            result[failed] = trial
        return restore(
            self._transport._source_center + self._transport._source_scale * result,
            shape,
        )

    def pullback(self, target_law):
        """Return the distribution obtained by mapping ``target_law`` through inverse.

        This is a Dempster-Hill predictive distribution only when reference cells
        have equal probability and the inverse preserves their labels.
        """
        target_law = target_law if isinstance(target_law, Law) else Law(target_law)
        if target_law.reference.dimension != self._transport._target.shape[1]:
            raise ValueError(
                f"target law dimension must be {self._transport._target.shape[1]}; "
                f"got {target_law.reference.dimension}"
            )
        _ = self._transport._inverse_domain
        if (
            self._transport.reference is not None
            and target_law._forward is None
            and len(target_law.cells) == target_law.reference.n + 1
            and np.all(target_law.weights > 0.0)
        ):
            raise ValueError(
                "cannot pull back the full reference distribution: some of its "
                "support lies outside the smooth inverse domain"
            )

        def backward(point):
            mapped, jacobian, _ = self._statistics(point)
            sign, logdet = np.linalg.slogdet(jacobian)
            return mapped, np.where(sign > 0.0, logdet, -np.inf)

        return target_law._map(self.inverse, backward)
