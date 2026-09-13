"""Softmax averages, derivatives, and inverse maps computed with Newton's method."""

import numpy as np
from numba import get_num_threads, njit, prange


@njit(cache=True, boundscheck=False)
def _row_weights(point, sites, offsets, tau, weights):
    for j in range(len(sites)):
        weights[j] = -offsets[j]
        for k in range(sites.shape[1]):
            weights[j] += point[k] * sites[j, k]
    top = weights.max()
    total = 0.0
    for j in range(len(sites)):
        weights[j] = np.exp((weights[j] - top) / tau)
        total += weights[j]
    weights /= total
    return top + tau * np.log(total)


@njit(cache=True, boundscheck=False)
def weights(points, sites, offsets, tau):
    result = np.empty((len(points), len(sites)))
    for i in range(len(points)):
        _row_weights(points[i], sites, offsets, tau, result[i])
    return result


@njit(cache=True, boundscheck=False)
def _read_one(point, sites, offsets, tau, output):
    weight = np.empty(len(sites))
    _row_weights(point, sites, offsets, tau, weight)
    for k in range(sites.shape[1]):
        output[k] = 0.0
        for j in range(len(sites)):
            output[k] += weight[j] * sites[j, k]


@njit(cache=True, boundscheck=False)
def _read(points, sites, offsets, tau, output):
    for i in range(len(points)):
        _read_one(points[i], sites, offsets, tau, output[i])


@njit(cache=True, boundscheck=False, parallel=True)
def _read_parallel(points, sites, offsets, tau, output):
    for i in prange(len(points)):
        _read_one(points[i], sites, offsets, tau, output[i])


def read(points, sites, offsets, tau):
    result = np.empty((len(points), sites.shape[1]))
    kernel = _read_parallel if len(points) >= 512 else _read
    kernel(points, sites, offsets, tau, result)
    return result


@njit(cache=True, boundscheck=False)
def _potential(points, sites, offsets, tau):
    result = np.empty(len(points))
    weight = np.empty(len(sites))
    for i in range(len(points)):
        result[i] = _row_weights(points[i], sites, offsets, tau, weight)
    return result


@njit(cache=True, boundscheck=False, parallel=True)
def _potential_parallel(points, sites, offsets, tau):
    result = np.empty(len(points))
    for i in prange(len(points)):
        weight = np.empty(len(sites))
        result[i] = _row_weights(points[i], sites, offsets, tau, weight)
    return result


def potential(points, sites, offsets, tau):
    work = len(points) * sites.size
    kernel = _potential_parallel if len(points) >= 512 and work >= 65536 else _potential
    return kernel(points, sites, offsets, tau)


@njit(cache=True, boundscheck=False)
def _row_statistics(point, sites, offsets, tau, weight, mapped, jacobian):
    value = _row_weights(point, sites, offsets, tau, weight)
    dimension = sites.shape[1]
    for r in range(dimension):
        mapped[r] = 0.0
        for j in range(len(sites)):
            mapped[r] += weight[j] * sites[j, r]
    # Center sites around the mapped point to avoid cancellation in small derivatives.
    for r in range(dimension):
        for c in range(r + 1):
            total = 0.0
            for j in range(len(sites)):
                total += (
                    weight[j] * (sites[j, r] - mapped[r]) * (sites[j, c] - mapped[c])
                )
            jacobian[r, c] = total / tau
            jacobian[c, r] = jacobian[r, c]
    return value


@njit(cache=True, boundscheck=False)
def _map_jacobian(points, sites, offsets, tau):
    dimension = sites.shape[1]
    mapped = np.empty((len(points), dimension))
    jacobian = np.empty((len(points), dimension, dimension))
    weight = np.empty(len(sites))
    for i in range(len(points)):
        _row_statistics(points[i], sites, offsets, tau, weight, mapped[i], jacobian[i])
    return mapped, jacobian


@njit(cache=True, boundscheck=False, parallel=True)
def _map_jacobian_parallel(points, sites, offsets, tau):
    dimension = sites.shape[1]
    mapped = np.empty((len(points), dimension))
    jacobian = np.empty((len(points), dimension, dimension))
    for i in prange(len(points)):
        weight = np.empty(len(sites))
        _row_statistics(points[i], sites, offsets, tau, weight, mapped[i], jacobian[i])
    return mapped, jacobian


def map_jacobian(points, sites, offsets, tau):
    work = len(points) * sites.size * sites.shape[1]
    parallel = len(points) >= 512 and work >= 65536
    kernel = _map_jacobian_parallel if parallel else _map_jacobian
    return kernel(points, sites, offsets, tau)


@njit(cache=True, boundscheck=False)
def _inverse_one(target, start, sites, offsets, tau, tolerance, max_iterations, output):
    dimension = len(target)
    point = start.copy()
    mapped = np.empty(dimension)
    jacobian = np.empty((dimension, dimension))
    weight = np.empty(len(sites))
    for _ in range(max_iterations):
        value = _row_statistics(point, sites, offsets, tau, weight, mapped, jacobian)
        gradient = mapped - target
        residual = np.linalg.norm(gradient)
        if residual <= tolerance or not np.isfinite(residual):
            break
        if not np.isfinite(jacobian).all():
            output[:] = point
            return np.inf
        # Add to the diagonal to keep the Newton step solvable near zero derivatives.
        ridge = 1e-12 * max(jacobian.max(), 1.0)
        for k in range(dimension):
            jacobian[k, k] += ridge
        direction = np.linalg.solve(jacobian, gradient)
        decrement = np.dot(gradient, direction)
        objective = value - np.dot(target, point)
        step, accepted = 1.0, False
        for _ in range(60):
            trial = point - step * direction
            trial_value = _row_weights(trial, sites, offsets, tau, weight)
            if trial_value - np.dot(target, trial) <= (
                objective - 1e-4 * step * decrement + 1e-14 * (1.0 + abs(objective))
            ):
                point = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    _row_statistics(point, sites, offsets, tau, weight, mapped, jacobian)
    output[:] = point
    return np.linalg.norm(mapped - target)


@njit(cache=True, boundscheck=False)
def _inverse(targets, starts, sites, offsets, tau, tolerance, max_iterations):
    result = np.empty_like(targets)
    residual = np.empty(len(targets))
    for i in range(len(targets)):
        residual[i] = _inverse_one(
            targets[i],
            starts[i],
            sites,
            offsets,
            tau,
            tolerance,
            max_iterations,
            result[i],
        )
    return result, residual


@njit(cache=True, boundscheck=False, parallel=True)
def _inverse_parallel(targets, starts, sites, offsets, tau, tolerance, max_iterations):
    result = np.empty_like(targets)
    residual = np.empty(len(targets))
    for i in prange(len(targets)):
        residual[i] = _inverse_one(
            targets[i],
            starts[i],
            sites,
            offsets,
            tau,
            tolerance,
            max_iterations,
            result[i],
        )
    return result, residual


def inverse(targets, starts, sites, offsets, tau, tolerance=1e-9, max_iterations=100):
    work = len(targets) * sites.shape[0] * sites.shape[1]
    parallel = get_num_threads() > 1 and len(targets) > 1 and work >= 4096
    kernel = _inverse_parallel if parallel else _inverse
    return kernel(targets, starts, sites, offsets, tau, tolerance, max_iterations)
