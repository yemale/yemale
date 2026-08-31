"""Spherical reference-cell barycentres."""

from functools import lru_cache

import numpy as np

from ._core import beta


@lru_cache(maxsize=32)
def _reference(n: int, dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical reference-cell barycentres ``m_j``, ranks, and signs."""
    reference_count = n + 1
    if dimension == 1:
        edge = np.linspace(-1.0, 1.0, reference_count + 1)
        barycentres = ((edge[1:] + edge[:-1]) / 2.0)[:, None]
    else:
        shell_count = 1
        if reference_count > dimension:
            ideal_shell_count = int(np.ceil(reference_count ** (1.0 / dimension)))
            # Keep at least d + 1 directional cells in every radial shell.
            maximum_shell_count = reference_count // (dimension + 1)
            shell_count = min(ideal_shell_count, maximum_shell_count)

        base, extra = divmod(reference_count, shell_count)
        cell_counts = np.full(shell_count, base, dtype=np.int64)
        cell_counts[:extra] += 1
        radial_edges = np.r_[0.0, np.cumsum(cell_counts) / reference_count]
        radial_lower = np.repeat(radial_edges[:-1], cell_counts)
        radial_upper = np.repeat(radial_edges[1:], cell_counts)
        radial_means = (radial_lower + radial_upper) / 2.0

        directional_means = np.empty((reference_count, dimension))
        start = 0
        for cell_count in cell_counts:
            stop = start + cell_count
            direction = beta.directional_barycentres(dimension, cell_count)
            directional_means[start:stop] = direction
            start = stop

        # m_j = E[R | radial cell] E[Theta | directional cell].
        barycentres = radial_means[:, None] * directional_means

    ranks, signs = _ranks_signs(barycentres)
    return barycentres, ranks, signs


def _ranks_signs(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.linalg.norm(target, axis=1)
    signs = np.divide(
        target,
        ranks[:, None],
        out=np.zeros_like(target),
        where=ranks[:, None] > 0.0,
    )
    return ranks, signs
