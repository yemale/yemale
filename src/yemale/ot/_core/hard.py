"""Streaming reads of max-affine labels."""

import numpy as np
from numba import njit, prange


_PARALLEL_MIN = 512


@njit(cache=True, boundscheck=False)
def _row_max(point, sites, offsets):
    label = 0
    best = -offsets[0]
    for k in range(sites.shape[1]):
        best += point[k] * sites[0, k]
    for j in range(1, sites.shape[0]):
        value = -offsets[j]
        for k in range(sites.shape[1]):
            value += point[k] * sites[j, k]
        if value > best:
            label, best = j, value
    return label


@njit(cache=True, boundscheck=False)
def _max_affine(points, sites, offsets, labels):
    for row in range(len(points)):
        labels[row] = _row_max(points[row], sites, offsets)


@njit(cache=True, boundscheck=False, parallel=True)
def _max_affine_parallel(points, sites, offsets, labels):
    for row in prange(len(points)):
        labels[row] = _row_max(points[row], sites, offsets)


def max_affine(points, sites, offsets):
    labels = np.empty(len(points), dtype=np.int64)
    # Thread setup costs more than it saves for short query batches.
    kernel = _max_affine if len(points) < _PARALLEL_MIN else _max_affine_parallel
    kernel(points, sites, offsets, labels)
    return labels
