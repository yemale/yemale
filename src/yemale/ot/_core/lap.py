"""Find assignment costs for each candidate target using Jonker-Volgenant."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, boundscheck=False)
def _lapjv(cost, row_bias=None):
    n = cost.shape[0] - 1
    free_row_idx = n
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.empty(n + 1, dtype=np.float64)
    r2c = np.full(n + 1, -1, dtype=np.int64)
    c2r = np.full(n + 1, -1, dtype=np.int64)

    for j in range(n + 1):
        best_val = np.inf
        best_i = -1
        for i in range(n + 1):
            if i == free_row_idx:
                continue
            value = cost[i, j]
            if row_bias is not None:
                value += row_bias[i]
            if value < best_val:
                best_val = value
                best_i = i
        v[j] = best_val
        if r2c[best_i] == -1:
            r2c[best_i] = j
            c2r[j] = best_i

    for i in range(n + 1):
        if r2c[i] == -1:
            continue
        j1 = r2c[i]
        best = np.inf
        for j in range(n + 1):
            if j != j1:
                rc = cost[i, j] - v[j]
                if rc < best:
                    best = rc
        u[i] = best
        v[j1] = cost[i, j1] - best

    dist = np.empty(n + 1, dtype=np.float64)
    pred = np.empty(n + 1, dtype=np.int64)
    scanned = np.empty(n + 1, dtype=np.bool_)

    for free_i in range(n + 1):
        if r2c[free_i] != -1:
            continue
        u_fi = u[free_i]
        frow = cost[free_i]
        for j in range(n + 1):
            dist[j] = frow[j] - u_fi - v[j]
            pred[j] = free_i
            scanned[j] = False

        sink = -1
        h = 0.0
        while True:
            min_d = np.inf
            j_star = -1
            for j in range(n + 1):
                if not scanned[j] and dist[j] < min_d:
                    min_d = dist[j]
                    j_star = j
            scanned[j_star] = True
            if c2r[j_star] == -1:
                sink = j_star
                h = min_d
                break
            i_m = c2r[j_star]
            u_im = u[i_m]
            crow = cost[i_m]
            for j in range(n + 1):
                if not scanned[j]:
                    nd = min_d + crow[j] - u_im - v[j]
                    if nd < dist[j]:
                        dist[j] = nd
                        pred[j] = i_m

        u[free_i] += h
        for j in range(n + 1):
            if scanned[j] and j != sink:
                delta = dist[j] - h
                v[j] += delta
                i_row = c2r[j]
                if i_row >= 0:
                    u[i_row] -= delta

        j = sink
        while True:
            i = pred[j]
            c2r[j] = i
            old_j = r2c[i]
            r2c[i] = j
            j = old_j
            if i == free_i:
                break

    total = 0.0
    for i in range(n + 1):
        total += cost[i, r2c[i]]
    return total, r2c[free_row_idx], c2r, u, v


@njit(cache=True, boundscheck=False)
def _loo_costs(cost, c2r, u, v, free_target, base):
    n = cost.shape[0] - 1
    dist = np.full(n + 1, np.inf, dtype=np.float64)
    done = np.zeros(n + 1, dtype=np.bool_)
    pred = np.full(n + 1, -1, dtype=np.int64)
    dist[free_target] = 0.0

    for _ in range(n + 1):
        best = np.inf
        current = -1
        for j in range(n + 1):
            if not done[j] and dist[j] < best:
                best = dist[j]
                current = j
        done[current] = True
        for target in range(n + 1):
            if done[target]:
                continue
            source = c2r[target]
            reduced = cost[source, current] - u[source] - v[current]
            if reduced < 0.0:
                reduced = 0.0
            candidate = best + reduced
            if candidate < dist[target]:
                dist[target] = candidate
                pred[target] = current

    leave_one = np.empty(n + 1, dtype=np.float64)
    for target in range(n + 1):
        leave_one[target] = base + dist[target] + v[free_target] - v[target]
    return leave_one, pred


@njit(cache=True, boundscheck=False)
def assign(base, predecessor, free_target, reserved):
    """Return target-to-row indices, assigning the candidate row to ``reserved``."""
    result = base.copy()
    target = reserved
    while target != free_target:
        previous = predecessor[target]
        result[previous] = base[target]
        target = previous
    result[reserved] = len(base) - 1
    return result


@njit(cache=True, boundscheck=False)
def _assignments(base, predecessor, free_target, labels, result):
    for row in range(len(labels)):
        inverse = assign(base, predecessor, free_target, labels[row])
        for target in range(len(base)):
            result[row, inverse[target]] = target


@njit(cache=True, boundscheck=False, parallel=True)
def _assignments_parallel(base, predecessor, free_target, labels, result):
    for row in prange(len(labels)):
        inverse = assign(base, predecessor, free_target, labels[row])
        for target in range(len(base)):
            result[row, inverse[target]] = target


def assignments(base, predecessor, free_target, labels):
    result = np.empty((len(labels), len(base)), dtype=np.int64)
    parallel = len(labels) >= 512 and result.size >= 65536
    kernel = _assignments_parallel if parallel else _assignments
    kernel(base, predecessor, free_target, labels, result)
    return result


def solve_1d(source: np.ndarray, target: np.ndarray):
    """Sort sources and targets to compute costs with each target left out."""
    source_order = np.argsort(source, kind="stable")
    target_order = np.argsort(target, kind="stable")
    x = source[source_order]
    u = target[target_order]
    n = len(source)

    left = -x * u[:-1]
    right = -x * u[1:]
    prefix = np.concatenate(([0.0], np.cumsum(left)))
    suffix = np.concatenate((np.cumsum(right[::-1])[::-1], [0.0]))
    sorted_cost = prefix + suffix

    leave_one = np.empty(n + 1)
    leave_one[target_order] = sorted_cost
    free_position = int(np.flatnonzero(sorted_cost == sorted_cost.min())[-1])
    free_target = int(target_order[free_position])

    inverse = np.empty(n + 1, dtype=np.int64)
    inverse[target_order] = np.insert(source_order, free_position, n)

    sorted_pred = np.full(n + 1, -1, dtype=np.int64)
    sorted_pred[:free_position] = target_order[1 : free_position + 1]
    sorted_pred[free_position + 1 :] = target_order[free_position:-1]
    pred = np.empty(n + 1, dtype=np.int64)
    pred[target_order] = sorted_pred
    return leave_one, inverse, pred, free_target, (x, target_order)


def solve(
    cost: np.ndarray, row_bias: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute minimum assignment costs with each target left for the candidate.

    ``cost`` has n source rows plus a zero-cost row, and n + 1 target columns.
    Return costs, target-to-row indices, predecessors, and the free target.
    ``row_bias`` only seeds the matching; it does not change the objective.
    """
    total, free_column, column_to_row, u, v = _lapjv(cost, row_bias)
    free_column = int(free_column)
    leave_one, pred = _loo_costs(cost, column_to_row, u, v, free_column, float(total))
    return leave_one, column_to_row, pred, free_column
