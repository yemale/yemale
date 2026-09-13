"""Shared array boundaries."""

from operator import index

import numpy as np


def rows(value, dimension, name="point"):
    array = np.ascontiguousarray(value, dtype=np.float64)
    if dimension == 1 and array.ndim <= 1:
        shape = () if array.size == 1 else array.shape
    elif array.ndim and array.shape[-1] == dimension:
        shape = array.shape[:-1]
    else:
        raise ValueError(
            f"{name} must have shape ({dimension},) or (..., {dimension}); got {array.shape}"
        )
    return array.reshape(-1, dimension), tuple(shape)


def scalars(value, size, name):
    array, _ = rows(value, 1, name)
    if len(array) not in (1, size):
        raise ValueError(
            f"{name} returned {len(array)} values for {size} points; "
            "return a scalar or one value per point"
        )
    return np.broadcast_to(array[:, 0], (size,))


def restore(value, shape):
    result = value.reshape(shape + value.shape[1:])
    return result.item() if result.ndim == 0 else result


def readonly(value, *, dtype=np.float64):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def count(value, name="size", *, minimum=1):
    try:
        result = index(value)
    except TypeError:
        raise ValueError(
            f"{name} must be an integer >= {minimum}; got {value!r}"
        ) from None
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return result
