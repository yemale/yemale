"""Softmax averages, potentials, and derivatives."""

import numpy as np
from numba import njit, prange


@njit(cache=True, boundscheck=False)
def _row_weights(point, sites, offsets, tau, weights):
    """Fill softmax weights and return the log-sum-exp potential."""
    for j in range(len(sites)):
        weights[j] = -offsets[j]
        for k in range(sites.shape[1]):
            weights[j] += point[k] * sites[j, k]
    # Subtract the maximum before scaling so exponential arguments stay nonpositive.
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
    # Dispatch thresholds here and below keep small batches out of parallel kernels.
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
    """Fill the potential's gradient and Hessian; return its value."""
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
