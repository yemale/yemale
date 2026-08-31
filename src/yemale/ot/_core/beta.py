"""Exact directional barycentres for spherical reference cells."""

import numpy as np
from scipy.special import betainc, betaincinv, betaln


def directional_barycentres(dimension: int, cell_count: int) -> np.ndarray:
    """Return ``E[Theta | A_l]`` for the equal-mass directional cells.

    Equal-volume quantile cells ``P_l`` in ``[0, 1]^(d - 1)`` are mapped by
    the recursive spherical construction to directional cells ``A_l``.
    """
    u_bounds = _quantile_cells(dimension, cell_count)
    beta_bounds = _beta_quantile_bounds(dimension, u_bounds)
    barycentres = np.empty((cell_count, dimension))
    for coordinate in range(dimension):
        mean = _directional_mean(dimension, u_bounds, beta_bounds, coordinate)
        barycentres[:, coordinate] = mean
    return barycentres


def _quantile_cells(dimension: int, cell_count: int) -> np.ndarray:
    """Return bounds for equal-volume ``P_l`` in quantile coordinates."""
    cells = np.empty((cell_count, dimension - 1, 2))
    _fill_quantile_cells(dimension, cell_count, cells, 0, 0)
    return cells


def _beta_quantile_bounds(dimension: int, u_bounds: np.ndarray) -> np.ndarray:
    """Return rescaled-Beta bounds for the non-circular coordinate laws.

    The final two-dimensional direction is parametrised directly by its
    uniform angle quantile in ``u_bounds``.
    """
    beta_bounds = np.empty((len(u_bounds), dimension - 2, 2))
    for level in range(dimension - 2):
        sphere_dimension = dimension - level
        alpha, beta = _coordinate_beta_parameters(sphere_dimension)
        beta_bounds[:, level] = betaincinv(alpha, beta, u_bounds[:, level])
    return beta_bounds


def _directional_mean(
    dimension: int,
    u_bounds: np.ndarray,
    beta_bounds: np.ndarray,
    coordinate: int,
) -> np.ndarray:
    """Return ``E[Theta_j | A_l]`` for one coordinate in every cell."""
    mean = np.ones(len(u_bounds))
    for level in range(coordinate + 1):
        sphere_dimension = dimension - level
        if sphere_dimension == 2:
            # Theta^(2) = (cos(phi), sin(phi)), with phi = 2 pi U.
            lower, upper = 2.0 * np.pi * u_bounds[:, level].T
            if coordinate == level:
                return mean * (np.sin(upper) - np.sin(lower)) / (upper - lower)
            return mean * (np.cos(lower) - np.cos(upper)) / (upper - lower)

        alpha, beta = _coordinate_beta_parameters(sphere_dimension)
        lower, upper = beta_bounds[:, level].T
        probability = u_bounds[:, level, 1] - u_bounds[:, level, 0]
        if coordinate == level:
            moment = _beta_interval_moment(alpha, beta, 1.0, 0.0, lower, upper)
            conditional_mean = moment / probability
            return mean * (2.0 * conditional_mean - 1.0)

        moment = _beta_interval_moment(alpha, beta, 0.5, 0.5, lower, upper)
        conditional_scale = moment / probability
        mean *= 2.0 * conditional_scale
    return mean


def _fill_quantile_cells(
    sphere_dimension: int,
    cell_count: int,
    cells: np.ndarray,
    first_cell: int,
    level: int,
) -> None:
    """Recursively fill equal-volume quantile cells ``P_l``."""
    if sphere_dimension <= 2:
        u_edges = np.linspace(0.0, 1.0, cell_count + 1)
        cells[first_cell : first_cell + cell_count, level, 0] = u_edges[:-1]
        cells[first_cell : first_cell + cell_count, level, 1] = u_edges[1:]
        return

    child_counts = _child_counts(sphere_dimension, cell_count)
    u_edges = np.cumsum([0] + list(child_counts)) / cell_count
    for band, child_count in enumerate(child_counts):
        cells[first_cell : first_cell + child_count, level, 0] = u_edges[band]
        cells[first_cell : first_cell + child_count, level, 1] = u_edges[band + 1]
        _fill_quantile_cells(
            sphere_dimension - 1, child_count, cells, first_cell, level + 1
        )
        first_cell += child_count


def _child_counts(sphere_dimension: int, cell_count: int) -> tuple[int, ...]:
    """Return final-cell counts assigned to each first-coordinate band."""
    if cell_count <= 1:
        return (cell_count,)

    band_count = 2
    if cell_count > sphere_dimension:
        ideal_band_count = round(cell_count ** (1.0 / (sphere_dimension - 1)))
        maximum_band_count = cell_count // sphere_dimension
        band_count = max(2, min(ideal_band_count, maximum_band_count))
    base, extra = divmod(cell_count, band_count)
    child_counts = [base + 1] * extra + [base] * (band_count - extra)
    if cell_count > sphere_dimension and max(child_counts) < sphere_dimension:
        # When m < count < 2m, retain one full m-cell child subtree.
        child_counts = [cell_count - sphere_dimension, sphere_dimension]
    return tuple(child_counts)


def _coordinate_beta_parameters(sphere_dimension: int) -> tuple[float, float]:
    """Return the Beta parameters of ``(T_m + 1) / 2``."""
    alpha = (sphere_dimension - 1) / 2.0
    return alpha, alpha


def _beta_interval_moment(
    alpha, beta, v_power, one_minus_v_power, lower, upper
) -> np.ndarray:
    """Return the Beta interval moment over ``[lower, upper]``."""
    shifted_alpha = alpha + v_power
    shifted_beta = beta + one_minus_v_power
    # ``betainc`` is regularised; this ratio restores the Beta moment.
    log_normalizer = betaln(shifted_alpha, shifted_beta) - betaln(alpha, beta)
    normalizer = np.exp(log_normalizer)
    lower_probability = betainc(shifted_alpha, shifted_beta, lower)
    upper_probability = betainc(shifted_alpha, shifted_beta, upper)
    return normalizer * (upper_probability - lower_probability)
