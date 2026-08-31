"""Independent checks for hard candidate-augmented transport."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import yemale.ot as ot
from yemale.ot._core import beta, lap
from yemale.ot._reference import _reference


def _augmented_inverse(source, target, point):
    """Return the target-to-source inverse assignment for one candidate point."""
    augmented_source = np.vstack([source, np.asarray(point)[None]])
    squared_cost = ((augmented_source[:, None] - target[None]) ** 2).sum(axis=2)
    row, column = linear_sum_assignment(squared_cost)
    inverse = np.empty(len(target), dtype=np.int64)
    inverse[column] = row
    return inverse


def test_leave_one_solver_matches_independent_hungarian_oracle():
    rng = np.random.default_rng(0)
    for n in range(2, 12):
        cost = np.vstack((rng.uniform(size=(n, n + 1)), np.zeros(n + 1)))
        leave_one, base, predecessor, free_target = lap.solve(cost)
        expected = []
        for reserved in range(n + 1):
            reduced = np.delete(cost[:-1], reserved, axis=1)
            _, column = linear_sum_assignment(reduced)
            expected.append(reduced[np.arange(n), column].sum())
        assert np.allclose(leave_one, expected, atol=1e-11)
        for reserved in range(n + 1):
            inverse = lap.assign(base, predecessor, free_target, reserved)
            keep = np.arange(n + 1) != reserved
            assert inverse[reserved] == n
            assert sorted(inverse[keep]) == list(range(n))
            assert np.allclose(cost[inverse[keep], keep].sum(), expected[reserved])

        source = rng.normal(size=n)
        target = rng.normal(size=n + 1)
        source_target_cost = (source[:, None] - target) ** 2
        candidate_cost = np.zeros(n + 1)
        augmented_cost = np.vstack((source_target_cost, candidate_cost))
        one_dimensional_costs = lap.solve_1d(source, target)[0]
        general_costs = lap.solve(augmented_cost)[0]
        assert np.allclose(one_dimensional_costs, general_costs)


def test_hard_transport_and_assignment_match_augmented_hungarian_oracle():
    rng = np.random.default_rng(1)
    for dimension in (1, 3):
        for _ in range(12):
            source = rng.normal(size=(rng.integers(5, 25), dimension))
            target = rng.normal(size=(len(source) + 1, dimension))
            transport = ot.fit(source, target=target)
            point = rng.normal(size=dimension)
            source_center = source.mean(axis=0)
            normalized_source = source - source_center
            source_scale = np.linalg.norm(normalized_source) / np.sqrt(len(source))
            if source_scale == 0.0:
                source_scale = 1.0
            normalized_source /= source_scale

            normalized_point = (point - source_center) / source_scale
            inverse = _augmented_inverse(
                normalized_source,
                target,
                normalized_point,
            )
            label = int(np.flatnonzero(inverse == len(source))[0])
            assignment = transport.assignment(point)
            assert assignment[-1] == label
            assert np.array_equal(assignment[inverse], np.arange(len(inverse)))
            assert np.allclose(transport(point), target[label])
            assert np.allclose(transport.rank(point), np.linalg.norm(target[label]))
            norm = np.linalg.norm(target[label])
            expected_sign = target[label] / norm if norm else np.zeros(dimension)
            assert np.allclose(transport.sign(point), expected_sign)


def test_fit_is_translation_scale_invariant():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(40, 3))
    point = rng.normal(size=(100, 3))
    expected = ot.fit(source)(point)
    assert np.allclose(ot.fit(source + 1e12)(point + 1e12), expected)
    assert np.allclose(ot.fit(source * 1e-9)(point * 1e-9), expected)


def test_1d_flat_batches_match_column_batches():
    transport = ot.fit(np.arange(5.0)[:, None])
    points = np.linspace(-2.0, 2.0, 7)
    column = points[:, None]
    assert np.allclose(transport(points), transport(column))
    assert np.allclose(transport.rank(points), transport.rank(column))
    assert np.allclose(transport.sign(points), transport.sign(column))


def test_parallel_batch_matches_individual_queries():
    rng = np.random.default_rng(5)
    transport = ot.fit(rng.normal(size=(20, 3)))
    points = rng.normal(size=(512, 3))
    assert np.allclose(transport(points), np.array([transport(point) for point in points]))
    assert np.allclose(transport.rank(points), [transport.rank(point) for point in points])


def test_fit_rejects_invalid_source_or_target():
    with pytest.raises(ValueError, match="source must be 2-D"):
        ot.fit(np.arange(3.0))
    with pytest.raises(ValueError, match="source must contain at least one point"):
        ot.fit(np.empty((0, 2)))
    with pytest.raises(ValueError, match="source must have at least one dimension"):
        ot.fit(np.empty((2, 0)))

    for value in (np.nan, np.inf):
        source = np.array([[1.0, 2.0], [value, 3.0], [4.0, 5.0]])
        with pytest.raises(ValueError, match="source must be finite"):
            ot.fit(source)
    source = np.zeros((3, 2))
    with pytest.raises(ValueError, match="target must have shape"):
        ot.fit(source, target=np.zeros((3, 2)))
    target = np.zeros((4, 2))
    target[0, 0] = np.nan
    with pytest.raises(ValueError, match="target must be finite"):
        ot.fit(source, target=target)

    transport = ot.fit(source)
    with pytest.raises(ValueError, match="assignment expects one point"):
        transport.assignment(np.zeros((2, 2)))


def test_reference_barycentres_are_first_cell_moments():
    assert np.allclose(_reference(3, 1)[0][:, 0], [-0.75, -0.25, 0.25, 0.75])
    assert np.allclose(
        _reference(3, 2)[0],
        np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) / np.pi,
    )
    assert np.allclose(
        beta.directional_barycentres(3, 4)[0],
        [-0.75, 0.0, 0.0],
    )


def test_leave_one_row_symmetry_gives_a_permutation_of_labels():
    rng = np.random.default_rng(9)
    source = rng.normal(size=(9, 2))
    labels = []
    for row in range(len(source)):
        calibration = np.delete(source, row, axis=0)
        labels.append(int(ot.fit(calibration).assignment(source[row])[-1]))
    assert sorted(labels) == list(range(len(source)))
