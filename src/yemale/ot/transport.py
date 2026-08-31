"""Exact candidate-augmented transport."""

from dataclasses import dataclass

import numpy as np

from ._core import hard, lap
from ._reference import _ranks_signs, _reference


@dataclass
class _Transport:
    _target: np.ndarray
    _ranks: np.ndarray
    _signs: np.ndarray
    _source_center: np.ndarray
    _source_scale: float
    _affine_offsets: np.ndarray
    _assignment_tree: tuple[np.ndarray, np.ndarray, int]
    _one_dimensional_lookup: tuple[np.ndarray, np.ndarray] | None

    def __call__(self, point) -> np.ndarray:
        labels, shape = self._labels(point)
        return _restore(self._target[labels], shape)

    def rank(self, point) -> np.ndarray:
        labels, shape = self._labels(point)
        return _restore(self._ranks[labels], shape)

    def sign(self, point) -> np.ndarray:
        labels, shape = self._labels(point)
        return _restore(self._signs[labels], shape)

    def assignment(self, point) -> np.ndarray:
        """Return the full assignment when one candidate point is appended."""
        labels, _ = self._labels(point)
        if len(labels) != 1:
            raise ValueError("assignment expects one point")
        inverse = lap.assign(*self._assignment_tree, int(labels[0]))
        assignment = np.empty_like(inverse)
        assignment[inverse] = np.arange(len(inverse))
        return assignment

    def _labels(self, point) -> tuple[np.ndarray, tuple[int, ...]]:
        value = np.ascontiguousarray(point, dtype=np.float64)
        dimension = self._target.shape[1]
        if dimension == 1 and value.ndim <= 1:
            shape = () if value.size == 1 else value.shape
        else:
            shape = value.shape[:-1]
        points = value.reshape(-1, dimension)
        normalized_points = (points - self._source_center) / self._source_scale
        return self._maximize(normalized_points), tuple(shape)

    def _maximize(self, normalized_points: np.ndarray) -> np.ndarray:
        if self._one_dimensional_lookup is None:
            return hard.max_affine(normalized_points, self._target, self._affine_offsets)
        sorted_source, target_order = self._one_dimensional_lookup
        position = np.searchsorted(sorted_source, normalized_points[:, 0], side="left")
        return target_order[position]


def _restore(value: np.ndarray, shape: tuple[int, ...]):
    result = value.reshape(shape + value.shape[1:])
    return result.item() if result.ndim == 0 else result


def fit(source, *, target=None) -> _Transport:
    """Fit source points to a target or the canonical reference.

    A supplied target has shape ``(n + 1, d)`` and is expressed in the
    centred, globally scaled coordinates of the source points.
    """
    source = np.ascontiguousarray(source, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(
            "source must be 2-D of shape (n, d); for 1-D data pass source[:, None]"
        )
    n, dimension = source.shape
    if n == 0:
        raise ValueError("source must contain at least one point")
    if dimension == 0:
        raise ValueError("source must have at least one dimension")
    if not np.isfinite(source).all():
        raise ValueError("source must be finite")
    if target is None:
        target, ranks, signs = _reference(n, dimension)
    else:
        target = np.ascontiguousarray(target, dtype=np.float64)
        expected_target_shape = (n + 1, dimension)
        if target.shape != expected_target_shape:
            raise ValueError(
                f"target must have shape {expected_target_shape}, got {target.shape}"
            )
        if not np.isfinite(target).all():
            raise ValueError("target must be finite")
        ranks, signs = _ranks_signs(target)

    source_center = source.mean(axis=0)
    normalized_source = source - source_center
    source_scale = float(np.linalg.norm(normalized_source) / np.sqrt(n))
    if source_scale == 0.0:
        source_scale = 1.0
    normalized_source /= source_scale

    source_squared = np.sum(normalized_source * normalized_source, axis=1)
    target_squared = np.sum(target * target, axis=1)

    if dimension == 1:
        leave_one_costs, base_assignment, predecessor, free_target, lookup = lap.solve_1d(
            normalized_source[:, 0], target[:, 0]
        )
    else:
        cost = np.empty((n + 1, n + 1))
        np.matmul(normalized_source, target.T, out=cost[:n])
        cost[:n] *= -2.0
        cost[:n] += source_squared[:, None]
        cost[:n] += target_squared
        cost[n] = 0.0
        leave_one_costs, base_assignment, predecessor, free_target = lap.solve(cost)
        lookup = None

    affine_offsets = 0.5 * (leave_one_costs + target_squared)
    affine_offsets -= affine_offsets.mean()

    return _Transport(
        _target=target,
        _ranks=ranks,
        _signs=signs,
        _source_center=source_center,
        _source_scale=source_scale,
        _affine_offsets=affine_offsets,
        _assignment_tree=(base_assignment, predecessor, free_target),
        _one_dimensional_lookup=lookup,
    )
