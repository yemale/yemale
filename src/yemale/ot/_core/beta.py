"""Construct spherical cells through independent uniform probability coordinates.
For ambient m > 2, Theta_m = (T, sqrt(1 - T**2) * Theta_(m-1)), with
(T + 1)/2 ~ Beta(a, a), a = (m - 1)/2, independent of uniform Theta_(m-1).
For m = 2, use (cos(phi), sin(phi)), with phi uniform on [0, 2*pi).
bounds/u_bounds bound d - 1 CDF coordinates, the last being phi/(2*pi);
beta_bounds bound the corresponding (T + 1)/2 values at the first d - 2 levels.
Box volumes are spherical probabilities; conditioning preserves independence.
sphere_dimension and remaining mean ambient coordinate counts.
Reference handles d = 1 separately.
"""

import numpy as np
from scipy.special import betainc, betaincinv, betaln


def directional_barycentres(
    dimension: int, cell_count: int, *, bounds=None, beta_bounds=None
) -> np.ndarray:
    """Return the mean direction within each cell on the sphere.

    Without ``bounds``, split the unit cube into equal-volume boxes, then
    map them to cells on the sphere.
    """
    u_bounds = bounds
    if u_bounds is None:
        u_bounds = _quantile_cells(dimension, cell_count)
    if beta_bounds is None:
        beta_bounds = _beta_quantile_bounds(dimension, u_bounds)
    barycentres = np.empty((cell_count, dimension))

    # Product of the preceding E[sqrt(1 - T**2) | band] factors.
    prefix_scale = np.ones(len(u_bounds))
    for level in range(dimension - 2):
        alpha, beta = _coordinate_beta_parameters(dimension - level)
        lower, upper = beta_bounds[:, level].T
        probability = u_bounds[:, level, 1] - u_bounds[:, level, 0]

        moment = _beta_interval_moment(alpha, beta, 1.0, 0.0, lower, upper)
        conditional_mean = moment / probability
        barycentres[:, level] = prefix_scale * (2.0 * conditional_mean - 1.0)

        moment = _beta_interval_moment(alpha, beta, 0.5, 0.5, lower, upper)
        conditional_scale = moment / probability
        prefix_scale *= 2.0 * conditional_scale

    # The final two coordinates are cosine and sine of a uniform angle.
    lower, upper = 2.0 * np.pi * u_bounds[:, dimension - 2].T
    arc_width = upper - lower
    cosine_integral = np.sin(upper) - np.sin(lower)
    sine_integral = np.cos(lower) - np.cos(upper)

    barycentres[:, -2] = prefix_scale * cosine_integral / arc_width
    barycentres[:, -1] = prefix_scale * sine_integral / arc_width
    return barycentres


def angular_moments(
    dimension: int, bounds: np.ndarray, order, *, beta_bounds=None
) -> np.ndarray:
    """Return E[product(Theta_r**order[r]) | cell] for each probability box.

    At each level, every later coordinate contains the factor sqrt(1 - T**2).
    Their combined exponent is sum(order[level + 1:]); independence in
    probability coordinates lets these level contributions multiply.
    """
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
    """Allocate bands with an exact-arithmetic affine-span invariant.
    For m = sphere_dimension and cell_count > m, keep at least two nonempty
    bands and a child with at least m cells. Its means affinely span the
    remaining m - 1 coordinates; another band supplies the final dimension.
    The induction starts with at least three noncollinear circular-arc means.
    The root count sets resolution; no numerical conditioning bound is claimed.
    """
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
    """Average T**power * (1 - T**2)**(tail_power / 2) on [lower, upper].
    T is a uniform-sphere coordinate in ambient dimension remaining.
    mass is the whole interval's probability under that marginal;
    both signed partial integrals use this same denominator.
    """
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
    """Average cos(theta)**cosine_power * sin(theta)**sine_power on each arc.
    bounds are turn fractions; divide the integral by 2*pi*(upper - lower).
    Translate each quadrant to phi in [0, pi/2], restore its signs, and swap
    powers in odd quadrants. For local sine/cosine powers s,c, x = sin(phi)**2
    gives half an incomplete Beta integral with parameters (s + 1)/2, (c + 1)/2.
    Separate complements and exact quadrant endpoints preserve tail accuracy.
    """
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
