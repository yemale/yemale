"""Smooth transport, augmented dual maps, and source distributions."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from yemale._array import restore, rows

from ._core import smooth
from .law import Law
from .transport import Transport


@dataclass(eq=False)
class SmoothMap:
    """A smooth transport that blends target centres instead of choosing one.

    Create with ``transport.smooth()``. Source queries follow Transport's shape
    conventions. ``inverse`` blends the fitted sources and a supplied candidate.
    """

    _transport: Transport
    _tau: float  # Public temperature divided by the fitted source scale.

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
        """Return target points, Jacobians in normalized source coordinates, and shape."""
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
        # D_z T = D_x T / source_scale, with x the normalized source point.
        logdet -= jacobian.shape[-1] * np.log(self._transport._source_scale)
        # The exact Jacobian is positive definite; nonpositive determinants give zero density.
        return mapped, np.where(sign > 0.0, logdet, -np.inf), shape

    @cached_property
    def _target_rank(self):
        centered = self._transport._centered_target
        scale = np.abs(centered).max()
        return np.linalg.matrix_rank(centered / scale) if scale > 0 else 0

    def log_density(self, points):
        """Return the log of ``density`` for one point or a batch."""
        reference = self._transport._require_reference()
        if self._target_rank < reference.dimension:
            raise ValueError(
                "smooth density requires targets that affinely span the space"
            )
        mapped, logdet, shape = self._density_coordinates(points)
        value = np.asarray(reference.logpdf(mapped)).reshape(-1)
        # Mask zero density first: reference logpdf can be +inf at the origin.
        valid = (value != -np.inf) & (logdet != -np.inf)
        value[~valid] = -np.inf
        value[valid] += logdet[valid]
        return restore(value, shape)

    def density(self, points):
        r"""Evaluate the smooth density in source coordinates.

        .. math:: p_\tau(z)=p_\nu(T_\tau(z))|\det DT_\tau(z)|.

        Its integral is nu(K), not one, where K is the convex hull of the
        target centres. No normalization is applied. Requires a Reference
        target whose centres affinely span the space. Accept ``(d,)`` or
        ``(..., d)`` and return one value per point.
        """
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(self.log_density(points))

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

    def inverse(self, target, *, candidate):
        r"""Map reference points to a softmax average of all augmented sources.

        .. math::

            Q_\tau(u;z)=\frac{\sum_{i=1}^{n+1}
            e^{(\langle u,\zeta_i\rangle-\Phi(\zeta_i))/\tau}\zeta_i}
            {\sum_{i=1}^{n+1}
            e^{(\langle u,\zeta_i\rangle-\Phi(\zeta_i))/\tau}}.

        Here zeta consists of the fitted observations followed by ``candidate``;
        Phi is the hard transport potential. Supply one candidate in source
        coordinates. ``target`` uses reference coordinates, with shape ``(d,)``
        or ``(..., d)``; keep the input batch shape.
        This dual-side map is defined everywhere, not a numerical inverse of
        the forward smooth map.
        """
        return self._augmented_inverse(candidate)(target)

    def _augmented_inverse(self, candidate):
        """Bind one candidate and its hard potential to the dual softmax."""
        transport = self._transport
        point, _ = transport._query(candidate)
        if len(point) != 1:
            raise ValueError(
                f"candidate must be one source point; got {len(point)} points"
            )
        potential = transport._potential_from_labels(point, transport._maximize(point))
        sites = np.concatenate((transport._normalized_sources, point))
        offsets = np.r_[
            transport._source_potential,
            potential / transport._source_scale - point @ transport._target_center,
        ]

        def inverse(target):
            targets, shape = rows(target, sites.shape[1], "target")
            if not np.isfinite(targets).all():
                raise ValueError("target must be finite")
            mapped = smooth.read(
                targets - transport._target_center, sites, offsets, self._tau
            )
            return restore(
                transport._source_center + transport._source_scale * mapped, shape
            )

        return inverse

    def pullback(self, target_law, *, candidate):
        r"""Map a reference law through the augmented dual map at one candidate.

        .. math:: \Pi_\tau(\,\cdot\,;z)=Q_\tau(\,\cdot\,;z)_\#\nu.

        ``target_law`` is a Reference or Law. The candidate stays fixed across
        samples and expectations. Return a Law providing sample, expect, moment,
        mean and cov through this map. Selecting reference cells gives the
        corresponding component laws.
        """
        target_law = target_law if isinstance(target_law, Law) else Law(target_law)
        if target_law.reference.dimension != self._transport._target.shape[1]:
            raise ValueError(
                f"target law dimension must be {self._transport._target.shape[1]}; "
                f"got {target_law.reference.dimension}"
            )
        return target_law._map(self._augmented_inverse(candidate))
