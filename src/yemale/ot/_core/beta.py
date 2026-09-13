"""Construct cells on the sphere and compute their directional moments."""

import numpy as np
from scipy.special import betainc, betaincinv, betaln


def directional_barycentres(
    dimension: int, cell_count: int, *, bounds=None, beta_bounds=None
) -> np.ndarray:
    """Return the mean direction within each cell on the sphere.

    Without ``bounds``, split the unit cube into equal-volume boxes, then
    map them to cells on the sphere.
    """
    u_bounds = _quantile_cells(dimension, cell_count) if bounds is None else bounds
    if beta_bounds is None:
        beta_bounds = _beta_quantile_bounds(dimension, u_bounds)
    barycentres = np.empty((cell_count, dimension))
    for coordinate in range(dimension):
        mean = _directional_mean(dimension, u_bounds, beta_bounds, coordinate)
        barycentres[:, coordinate] = mean
    return barycentres


def angular_moments(
    dimension: int, bounds: np.ndarray, order, *, beta_bounds=None
) -> np.ndarray:
    """Return the requested directional moment within each cell."""
    value = np.ones(len(bounds))
    if beta_bounds is None:
        beta_bounds = _beta_quantile_bounds(dimension, bounds)
    for level in range(dimension - 1):
        remaining = dimension - level
        if remaining == 2:
            return value * _arc_moment(order[level], order[level + 1], bounds[:, level])
        lower, upper = (2.0 * beta_bounds[:, level] - 1.0).T
        mass = bounds[:, level, 1] - bounds[:, level, 0]
        value *= _head_moment(
            remaining,
            order[level],
            sum(order[level + 1 :]),
            lower,
            upper,
            mass,
        )
    return value


def _quantile_cells(dimension: int, cell_count: int) -> np.ndarray:
    """Return lower and upper bounds of equal-volume boxes in the unit cube."""
    cells = np.empty((cell_count, dimension - 1, 2))
    _fill_quantile_cells(dimension, cell_count, cells, 0, 0)
    return cells


def _beta_quantile_bounds(dimension: int, u_bounds: np.ndarray) -> np.ndarray:
    """Convert probability bounds to Beta-distributed coordinate bounds.

    The last two coordinates use an angle, so they need no Beta conversion.
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
    """Return one coordinate of the mean direction in every spherical cell."""
    mean = np.ones(len(u_bounds))
    for level in range(coordinate + 1):
        sphere_dimension = dimension - level
        if sphere_dimension == 2:
            # In two dimensions, average cosine and sine over the angle interval.
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
    """Split the unit cube into equal-volume boxes, one coordinate at a time."""
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
    """Return how many cells to place in each band of the first coordinate."""
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
        # Keep one child large enough to span the remaining dimensions.
        child_counts = [cell_count - sphere_dimension, sphere_dimension]
    return tuple(child_counts)


def _coordinate_beta_parameters(sphere_dimension: int) -> tuple[float, float]:
    """Return Beta parameters for a sphere coordinate shifted from [-1, 1] to [0, 1]."""
    alpha = (sphere_dimension - 1) / 2.0
    return alpha, alpha


def _beta_interval_moment(
    alpha, beta, v_power, one_minus_v_power, lower, upper
) -> np.ndarray:
    """Integrate ``v**v_power * (1-v)**one_minus_v_power`` over ``[lower, upper]``
    against the Beta density, without dividing by the interval's probability.
    """
    shifted_alpha = alpha + v_power
    shifted_beta = beta + one_minus_v_power
    # betainc uses a normalized density; adjust its scale for the changed powers.
    log_normalizer = betaln(shifted_alpha, shifted_beta) - betaln(alpha, beta)
    normalizer = np.exp(log_normalizer)
    lower_probability = betainc(shifted_alpha, shifted_beta, lower)
    upper_probability = betainc(shifted_alpha, shifted_beta, upper)
    return normalizer * (upper_probability - lower_probability)


def _head_moment(remaining, power, tail_power, lower, upper, mass):
    """Average ``t**power * (1 - t**2)**(tail_power / 2)`` over a coordinate interval."""
    value = np.zeros(len(lower))
    negative = lower < 0.0
    if np.any(negative):
        value[negative] = (-1.0) ** power * _positive_head_moment(
            remaining,
            power,
            tail_power,
            -np.minimum(upper[negative], 0.0),
            -lower[negative],
            mass[negative],
        )
    positive = upper > 0.0
    if np.any(positive):
        value[positive] += _positive_head_moment(
            remaining,
            power,
            tail_power,
            np.maximum(lower[positive], 0.0),
            upper[positive],
            mass[positive],
        )
    return value


def _positive_head_moment(remaining, power, tail_power, lower, upper, mass):
    # Integrate x = t**2 directly; expanding a shifted power loses precision.
    alpha = (power + 1.0) / 2.0
    beta = (tail_power + remaining - 1.0) / 2.0
    # Compute 1 - t**2 accurately near both zero and one.
    probability = _beta_interval_probability(
        alpha,
        beta,
        lower * lower,
        upper * upper,
        np.where(lower < 0.5, 1.0 - lower * lower, (1.0 - lower) * (1.0 + lower)),
        np.where(upper < 0.5, 1.0 - upper * upper, (1.0 - upper) * (1.0 + upper)),
    )
    with np.errstate(divide="ignore"):
        log_value = (
            -np.log(2.0)
            + betaln(alpha, beta)
            - betaln(0.5, (remaining - 1.0) / 2.0)
            + np.log(probability)
            - np.log(mass)
        )
    return np.exp(log_value)


def _arc_moment(cosine_power, sine_power, bounds):
    lower, upper = bounds.T
    width = 2.0 * np.pi * (upper - lower)
    value = np.zeros(len(bounds))
    signs = (
        1.0,
        (-1.0) ** cosine_power,
        (-1.0) ** (cosine_power + sine_power),
        (-1.0) ** sine_power,
    )
    for quadrant in range(4):
        start, stop = quadrant / 4.0, (quadrant + 1.0) / 4.0
        left, right = np.maximum(lower, start), np.minimum(upper, stop)
        active = right > left
        if not np.any(active):
            continue
        left_angle = 2.0 * np.pi * (left[active] - start)
        right_angle = 2.0 * np.pi * (right[active] - start)
        x_lower = np.sin(left_angle) ** 2
        x_upper = np.sin(right_angle) ** 2
        complement_lower = np.cos(left_angle) ** 2
        complement_upper = np.cos(right_angle) ** 2
        x_lower[left[active] == start] = 0.0
        complement_lower[left[active] == start] = 1.0
        x_upper[right[active] == stop] = 1.0
        complement_upper[right[active] == stop] = 0.0
        cosine, sine = (
            (cosine_power, sine_power)
            if quadrant % 2 == 0
            else (sine_power, cosine_power)
        )
        alpha, beta = (sine + 1.0) / 2.0, (cosine + 1.0) / 2.0
        probability = _beta_interval_probability(
            alpha,
            beta,
            x_lower,
            x_upper,
            complement_lower,
            complement_upper,
        )
        with np.errstate(divide="ignore"):
            contribution = np.exp(
                -np.log(2.0)
                + betaln(alpha, beta)
                + np.log(probability)
                - np.log(width[active])
            )
        value[active] += signs[quadrant] * contribution
    return value


def _beta_interval_probability(
    alpha, beta, lower, upper, complement_lower, complement_upper
):
    """Compute a Beta interval probability without subtracting values near one."""
    lower_cdf = betainc(alpha, beta, lower)
    upper_cdf = betainc(alpha, beta, upper)
    from_lower = upper_cdf - lower_cdf
    from_upper = betainc(beta, alpha, complement_lower) - betainc(
        beta, alpha, complement_upper
    )
    value = np.where(lower_cdf + upper_cdf <= 1.0, from_lower, from_upper)
    return np.maximum(value, 0.0)


def directions_from_coords(dimension, bounds, coordinates):
    """Turn independent uniform values in [0, 1] into directions within the given cells."""

    def rec(remaining, level):
        lower, upper = bounds[:, level].T
        coordinate = lower + coordinates[:, level] * (upper - lower)
        if remaining == 2:
            angle = 2.0 * np.pi * coordinate
            return np.column_stack([np.cos(angle), np.sin(angle)])
        alpha, beta = _coordinate_beta_parameters(remaining)
        head = 2.0 * betaincinv(alpha, beta, coordinate) - 1.0
        scale = np.sqrt(np.maximum(1.0 - head * head, 0.0))
        return np.column_stack([head, scale[:, None] * rec(remaining - 1, level + 1)])

    return rec(dimension, 0)


def direction_cdf(unit):
    """Convert directions back to the [0, 1] coordinates used to construct the cells."""
    dimension = unit.shape[1]
    coordinates = np.empty((len(unit), dimension - 1))
    tail = unit
    for level in range(dimension - 1):
        remaining = dimension - level
        if remaining == 2:
            angle = np.arctan2(tail[:, 1], tail[:, 0]) % (2.0 * np.pi)
            coordinates[:, level] = angle / (2.0 * np.pi)
            break
        head = np.clip(tail[:, 0], -1.0, 1.0)
        alpha, beta = _coordinate_beta_parameters(remaining)
        coordinates[:, level] = betainc(alpha, beta, (head + 1.0) / 2.0)
        scale = np.sqrt(np.maximum(1.0 - head * head, 0.0))
        next_tail = np.zeros((len(unit), remaining - 1))
        moved = scale > 0.0
        next_tail[moved] = tail[moved, 1:] / scale[moved, None]
        tail = next_tail
    return coordinates
