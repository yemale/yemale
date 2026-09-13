"""Independent checks for hard candidate-augmented transport."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from yemale import ot
from yemale.ot._core import beta, lap


def _augmented_inverse(source, target, point):
    """Return the target-to-source inverse assignment for one candidate point."""
    augmented_source = np.vstack([source, np.asarray(point)[None]])
    squared_cost = ((augmented_source[:, None] - target[None]) ** 2).sum(axis=2)
    row, column = linear_sum_assignment(squared_cost)
    inverse = np.empty(len(target), dtype=np.int64)
    inverse[column] = row
    return inverse


@pytest.mark.parametrize("biased", [False, True])
def test_leave_one_solver_matches_independent_hungarian_oracle(biased):
    rng = np.random.default_rng(0)
    for n in range(2, 12):
        cost = np.vstack((rng.uniform(size=(n, n + 1)), np.zeros(n + 1)))
        row_bias = rng.normal(size=n) if biased else None
        leave_one, base, predecessor, free_target = lap.solve(cost, row_bias)
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
        source_target_cost = -source[:, None] * target
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
    assert np.allclose(ot.fit(source * 1e160)(point * 1e160), expected)


def test_1d_flat_batches_match_column_batches():
    transport = ot.fit(np.arange(5.0)[:, None])
    points = np.linspace(-2.0, 2.0, 7)
    column = points[:, None]
    assert np.allclose(transport(points), transport(column))
    assert np.allclose(transport.rank(points), transport.rank(column))
    assert np.allclose(transport.sign(points), transport.sign(column))
    assert np.array_equal(transport.assignment(points), transport.assignment(column))


def test_parallel_batch_matches_individual_queries():
    rng = np.random.default_rng(5)
    transport = ot.fit(rng.normal(size=(20, 3)))
    points = rng.normal(size=(4096, 3))
    assert np.allclose(
        transport(points), np.array([transport(point) for point in points])
    )
    assert np.allclose(
        transport.rank(points), [transport.rank(point) for point in points]
    )
    expected = np.array([transport.assignment(point) for point in points])
    for batch in (points[:2], points.reshape(2, 2048, 3)):
        assignments = transport.assignment(batch)
        assert assignments.shape == batch.shape[:-1] + (21,)
        assert np.array_equal(assignments.reshape(-1, 21), expected[: batch.size // 3])
    assert np.array_equal(transport.assignment(points[:1]), expected[0])
    assert transport.assignment(points[:0]).shape == (0, 21)
    for batch in (points[0], points.reshape(2, 2048, 3), points[:0]):
        result = transport.evaluate(batch)
        for name, query in (
            ("label", transport.label),
            ("target", transport),
            ("rank", transport.rank),
            ("sign", transport.sign),
            ("potential", transport.potential),
        ):
            np.testing.assert_array_equal(result[name], query(batch))


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
        with pytest.raises(ValueError, match="point must be finite"):
            ot.fit([[0.0, 0.0]]).label([value, 0.0])
    source = np.zeros((3, 2))
    with pytest.raises(ValueError, match="target must have shape"):
        ot.fit(source, target=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="n=3.*n=2"):
        ot.fit(source, target=ot.Reference(n=2, dimension=2))
    target = np.zeros((4, 2))
    target[0, 0] = np.nan
    with pytest.raises(ValueError, match="target must be finite"):
        ot.fit(source, target=target)


def test_reference_barycentres_are_first_cell_moments():
    for n in (0, 3):
        reference = ot.Reference(n=n, dimension=2)
        assert reference.n == n and reference.centers.shape == (n + 1, 2)
        assert np.array_equal(reference.centers, ot.reference(n, 2).centers)
    assert np.allclose(ot.reference(3, 1).centers[:, 0], [-0.75, -0.25, 0.25, 0.75])
    assert np.allclose(
        ot.reference(3, 2).centers,
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


def test_cell_geometry_in_original_coordinates():
    transport = ot.fit([[1.0], [5.0]], target=[[-2.0], [0.0], [2.0]])
    points = np.array([0.0, 1.0, 3.0, 5.0, 6.0])
    assert np.array_equal(transport.label(points), [0, 0, 1, 1, 2])
    np.testing.assert_allclose(
        transport.potential(points) - transport.potential(3),
        2 * np.maximum(np.abs(points - 3) - 2, 0),
    )
    matrix, offset = transport.halfspaces(1)
    np.testing.assert_allclose(matrix, [[-2], [2]])
    np.testing.assert_allclose(offset, [-2, 10])


@pytest.mark.parametrize("dimension", [1, 2])
def test_target_translation_preserves_assignments(dimension):
    rng = np.random.default_rng(43)
    source = rng.normal(size=(5, dimension))
    target = rng.normal(size=(6, dimension))
    points = rng.normal(size=(20, dimension))
    transport = ot.fit(source, target=target)
    shifted = ot.fit(source, target=target + 1e8)
    assert np.array_equal(shifted.label(points), transport.label(points))
    assert np.array_equal(
        shifted.assignment(points[0]), transport.assignment(points[0])
    )
    np.testing.assert_allclose(shifted(points) - 1e8, transport(points), atol=2e-8)


def test_far_targets_preserve_small_assignment_margins():
    for shift, margin in ((1e8, 1e-10), (1e15, 0.01)):
        target = np.array([[shift, shift], [shift + 1, shift]])
        transport = ot.fit([[0.0, 0.0]], target=target)
        points = [[margin, 1.0], [-margin, 1.0]]
        np.testing.assert_array_equal(transport.assignment(points), [[0, 1], [1, 0]])
        np.testing.assert_array_equal(transport.evaluate(points)["label"], [1, 0])


@pytest.mark.parametrize("dimension", [1, 2])
def test_target_scale_preserves_assignments_and_potential(dimension):
    rng = np.random.default_rng(4)
    source = rng.normal(size=(7, dimension))
    target = rng.normal(size=(8, dimension))
    points = rng.normal(size=(100, dimension))
    transport = ot.fit(source, target=target)
    for scale in (1e-150, 1e15, 1e150):
        scaled = ot.fit(source, target=scale * target)
        np.testing.assert_array_equal(
            scaled.assignment(points), transport.assignment(points)
        )
        np.testing.assert_allclose(
            scaled.potential(points) / scale, transport.potential(points), atol=1e-12
        )


@pytest.mark.parametrize(
    "source,target,point",
    [
        ([[0.0]], [[1.0], [-1.0]], 0.0),
        ([[-1.0], [1.0]], [[1.0], [-1.0], [1.0]], 2.0),
        ([[0.0], [0.0]], [[1.0], [-1.0], [0.0]], 0.0),
        ([[0, 0], [0, 0], [1, 1]], [[-1, -1], [1, -1], [-1, 1], [1, 1]], [0, 0]),
        (np.arange(18).reshape(9, 2), np.full((10, 2), 1e150), [0, 0]),
    ],
)
def test_ties_choose_the_smallest_optimal_label(source, target, point):
    transport = ot.fit(source, target=target)
    assert transport.label(point) == 0
    assert transport.assignment(point)[-1] == 0
    batch = np.broadcast_to(point, (512, np.shape(source)[1]))
    expected = np.broadcast_to(transport.assignment(point), (512, len(target)))
    np.testing.assert_array_equal(transport.assignment(batch), expected)
    np.testing.assert_array_equal(transport.evaluate(batch)["label"], np.zeros(512))
