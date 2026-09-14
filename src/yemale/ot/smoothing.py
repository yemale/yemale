"""Smooth transport, numerical inversion, and source distributions."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from yemale._array import restore, rows

from ._core import smooth
from .density import _density_region
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
        r"""Map each source point to a softmax average of target centres.

        .. math::

            T_\tau(z)=\sum_{j=1}^{n+1}p_{\tau,j}(z)m_j,\qquad
            p_{\tau,j}(z)=
            \frac{e^{(\langle z,m_j\rangle-\phi_j)/\tau}}
            {\sum_k e^{(\langle z,m_k\rangle-\phi_k)/\tau}}.

        Here m_j are the targets, phi_j are the fitted offsets, and tau is
        ``temperature``. Accept ``(d,)`` or ``(..., d)``; keep the input shape.
        """
        points, shape = self._transport._query(point)
        value = smooth.read(
            points,
            self._transport._centered_target,
            self._transport._affine_offsets,
            self._tau,
        )
        return restore(value + self._transport._target_center, shape)

    def potential(self, point):
        r"""Return the smooth potential; its gradient is this map.

        .. math::

            \Phi_\tau(z)=\tau\log\sum_{j=1}^{n+1}
            e^{(\langle z,m_j\rangle-\phi_j)/\tau}.

        Here tau is ``temperature``, m_j are the targets, and phi_j the fitted
        offsets. Use original source coordinates; return one value per point.
        """
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
        return mapped, jacobian, shape

    def jacobian(self, point):
        r"""Differentiate the map with respect to original source coordinates.

        .. math::

            DT_\tau(z)=\frac1\tau\mathrm{Cov}_{p_\tau(z)}(m_j).

        The covariance uses the target centres and their softmax weights;
        tau is ``temperature``.
        Return ``(d, d)`` for one point or ``(..., d, d)`` for a batch.
        """
        _, jacobian, shape = self._statistics(point)
        return restore(jacobian / self._transport._source_scale, shape)

    def map_jacobian(self, point):
        """Return ``(self(point), self.jacobian(point))`` with shared computation."""
        mapped, jacobian, shape = self._statistics(point)
        return restore(mapped, shape), restore(
            jacobian / self._transport._source_scale, shape
        )

    def _density_coordinates(self, point):
        mapped, jacobian, shape = self._statistics(point)
        sign, logdet = np.linalg.slogdet(jacobian)
        logdet -= jacobian.shape[-1] * np.log(self._transport._source_scale)
        return mapped, np.where(sign > 0.0, logdet, -np.inf), shape

    def log_density(self, points):
        """Return the log of ``density`` for one point or a batch."""
        reference = self._transport._require_reference()
        _ = self._transport._inverse_domain
        mapped, logdet, shape = self._density_coordinates(points)
        value = np.asarray(reference.logpdf(mapped)).reshape(-1)
        valid = (value != -np.inf) & (logdet != -np.inf)
        value[~valid] = -np.inf
        value[valid] += logdet[valid]
        return restore(value, shape)

    def density(self, points):
        r"""Evaluate the smooth density in source coordinates.

        .. math:: p_\tau^Z(z)=p_\nu(T_\tau(z))|\det DT_\tau(z)|.

        Its integral is nu(K), not one, where K is the convex hull of the
        target centres. No normalization is applied. Requires a Reference
        target whose centres affinely span the space. Accept ``(d,)`` or
        ``(..., d)`` and return one value per point.
        """
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(self.log_density(points))

    def density_region(self, mass, *, n_integration_points=256):
        r"""Select a smooth-density superlevel set by absolute integrated mass.

        .. math:: A_t=\{z:p_\tau^Z(z)\ge t\},\qquad
            \int_{A_t}p_\tau^Z(z)\,dz\approx\mathrm{mass}.

        ``mass`` is positive and cannot exceed nu(K), the density's total
        mass. The returned ``mass`` is approximate, not a coverage guarantee.
        ``n_integration_points`` sets the reference points per cell; inverse
        evaluations are reused for subsequent mass requests at that resolution.
        A request for the full known mass returns all source space.
        """
        mass = float(mass)
        total = self._density_mass
        if total is None:
            upper = float(self._transport.reference.ranks.max())
            if mass > upper:
                raise ValueError(
                    f"requested mass {mass:g} exceeds an upper bound on "
                    f"this density's total mass, {upper:.6g}"
                )
        return _density_region(
            self,
            self.log_density,
            self._density_points,
            mass,
            n_integration_points,
            total,
        )

    @cached_property
    def _hull(self):
        from scipy.spatial import ConvexHull

        _, center, scale = self._transport._inverse_domain
        return ConvexHull((self._transport._target - center) / scale)

    @cached_property
    def _density_mass(self):
        reference = self._transport._require_reference()
        _ = self._transport._inverse_domain
        if reference.dimension == 1:
            return float(np.ptp(reference.centers[:, 0]) / 2)
        if reference.dimension == 2:
            vertices = reference.centers[self._hull.vertices]
            following = np.roll(vertices, -1, axis=0)
            tangent = following - vertices
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            normal = np.c_[tangent[:, 1], -tangent[:, 0]]
            distance = np.sum(vertices * normal, axis=1)
            # Integrate the radial extent of each polygon edge under nu.
            angles = np.arcsinh(
                np.sum(following * tangent, axis=1) / distance
            ) - np.arcsinh(np.sum(vertices * tangent, axis=1) / distance)
            return float(np.sum(distance * angles) / (2 * np.pi))
        return None

    def _density_points(self, size):
        reference = self._transport._require_reference()
        points, weights = reference._law._integration_points(size)
        if reference.dimension <= 2:
            if reference.dimension == 1:
                radius = self._density_mass
                sites = reference.centers[:, 0]
                points = points * radius + (sites.min() + sites.max()) / 2
            else:
                direction = points / np.linalg.norm(points, axis=1)[:, None]
                _, center, scale = self._transport._inverse_domain
                inverse_radius = np.zeros(len(points))
                for equation in self._hull.equations:
                    normal = equation[:-1]
                    distance = normal @ center - scale * equation[-1]
                    np.maximum(
                        inverse_radius,
                        direction @ normal / distance,
                        out=inverse_radius,
                    )
                radius = 1 / inverse_radius
                points = points * radius[:, None]
            # Rescale the uniform radius to the hull; its length multiplies the weight.
            return self._inverse(points), weights * radius
        inside = self._contains_targets(points)
        points = points[inside]
        return (self._inverse(points) if len(points) else points), weights[inside]

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

    def _contains_targets(self, targets):
        sites = self._transport._target
        matrix, center, scale = self._transport._inverse_domain
        if sites.shape[1] == 1:
            return ((targets > sites.min()) & (targets < sites.max())).all(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = (targets - center) / scale
        inside = np.isfinite(normalized).all(axis=1)
        if sites.shape[1] == 2:
            normalized[~inside] = 0.0
            equations = self._hull.equations
            block_size = max(1, (1 << 17) // len(equations))
            for start in range(0, len(targets), block_size):
                block = slice(start, start + block_size)
                inside[block] &= np.all(
                    normalized[block] @ equations[:, :-1].T + equations[:, -1] < -1e-10,
                    axis=1,
                )
            return inside
        from scipy.optimize import linprog

        # Express the target as a weighted average of the centres and their mean.
        # Positive weight on the mean puts it strictly inside the convex hull.
        objective = np.zeros(len(sites) + 1)
        objective[-1] = -1.0
        for i in np.flatnonzero(inside):
            result = linprog(
                objective,
                A_eq=matrix,
                b_eq=np.r_[normalized[i], 1.0],
                bounds=(0.0, None),
                method="highs",
                options={
                    "primal_feasibility_tolerance": 1e-9,
                    "dual_feasibility_tolerance": 1e-9,
                },
            )
            inside[i] = result.success and result.x[-1] > 1e-8
        return inside

    def inverse(self, target):
        r"""Map target points back to source coordinates using Newton iterations.

        .. math::

            Q_\tau(u)=\arg\min_z\{\Phi_\tau(z)-\langle u,z\rangle\}.

        Phi_tau is ``self.potential``.
        Targets must lie strictly inside the convex hull of the target centres.
        The centres must affinely span all d dimensions.
        Accept ``(d,)`` or ``(..., d)`` and keep the input shape.
        """
        sites = self._transport._centered_target
        targets, shape = rows(target, sites.shape[1], "target")
        if not np.isfinite(targets).all():
            raise ValueError("target must be finite")
        if not self._contains_targets(targets).all():
            raise ValueError(
                "target must lie strictly inside the convex hull of target centers"
            )
        return restore(self._inverse(targets), shape)

    def _inverse(self, targets):
        sites = self._transport._centered_target
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
        return self._transport._source_center + self._transport._source_scale * result

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
            mapped, logdet, _ = self._density_coordinates(point)
            return mapped, logdet

        return target_law._map(self.inverse, backward)
