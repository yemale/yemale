"""Conformal operations compose the existing DH regions and laws."""

import numpy as np
import pytest

import yemale
from yemale import ot
from yemale.conformal import ScoreMap


@pytest.mark.parametrize("dimension", [1, 2])
def test_residual_regions_are_dh_regions(dimension):
    rng = np.random.default_rng(2)
    predictions = rng.normal(size=(39, dimension))
    residuals = rng.normal(size=predictions.shape)
    outcomes = predictions + residuals
    cp = yemale.conformalize(
        predictions, outcomes[:, 0] if dimension == 1 else outcomes
    )
    direct = ot.fit(residuals).quantile_region(0.8, rng=3)
    prediction = np.zeros(dimension)
    candidates = rng.normal(size=(20, dimension))
    cpd = cp.predict(prediction)
    region = cpd.region(0.8, rng=3)
    np.testing.assert_allclose(
        cpd.density(candidates), cp.transport.smooth().density(candidates)
    )
    np.testing.assert_array_equal(
        cpd.potential_gradient(candidates), cpd.transform(candidates)
    )
    np.testing.assert_array_equal(
        region.contains(candidates), direct.contains(candidates)
    )
    np.testing.assert_array_equal(
        region.select(candidates), candidates[direct.contains(candidates)]
    )
    assert region.coverage == direct.coverage
    if dimension == 1:
        cp = yemale.conformalize(predictions[:, 0], outcomes[:, 0])
        assert cp.predict(0).region(0.8, rng=3).contains(candidates[:, 0]).shape == (
            20,
        )


@pytest.mark.parametrize(
    "dtype,predictions,outcomes",
    [
        (np.uint8, [10, 10, 10], [0, 20, 30]),
        (np.int8, [120, 120, 120], [-120, 0, 120]),
    ],
)
def test_integer_residuals_do_not_overflow(dtype, predictions, outcomes):
    predictions = np.array(predictions, dtype=dtype)
    outcomes = np.array(outcomes, dtype=dtype)
    float_predictions = predictions.astype(np.float64)
    float_outcomes = outcomes.astype(np.float64)
    actual = yemale.conformalize(predictions, outcomes).predict(predictions[0])
    expected = yemale.conformalize(float_predictions, float_outcomes).predict(
        float_predictions[0]
    )
    np.testing.assert_array_equal(actual.rank(outcomes), expected.rank(float_outcomes))


def test_scores_receive_whole_arrays():
    calls = []

    def score(predictions, outcomes):
        calls.append((predictions.shape, outcomes.shape))
        return outcomes - predictions - 10

    cp = yemale.conformalize(
        np.zeros((19, 2)), np.arange(38).reshape(19, 2), score=score
    )
    assert calls == [((19, 2), (19, 2))]
    cpd = cp.predict([3, 4])
    region = cpd.region(reference_set=lambda u: u[:, 0] > 0)
    candidates = np.arange(80).reshape(40, 2)
    selected = region.select(candidates)
    assert calls == [((19, 2), (19, 2)), ((1, 2), (40, 2))]
    expected = cp.transport.region(lambda u: u[:, 0] > 0).contains(
        candidates - [3, 4] - 10
    )
    np.testing.assert_array_equal(selected, candidates[expected])
    for name in ("transform", "rank", "sign", "potential"):
        direct = cp.transport if name == "transform" else getattr(cp.transport, name)
        np.testing.assert_array_equal(
            getattr(cpd, name)(candidates),
            direct(candidates - [3, 4] - 10),
        )
    np.testing.assert_allclose(
        cpd.transform(candidates), cpd.rank(candidates)[:, None] * cpd.sign(candidates)
    )
    with pytest.raises(TypeError):
        cpd.potential_gradient(candidates)


def test_score_jacobian_chain_rule():
    # One prediction coordinate, two outcome coordinates, one score coordinate:
    # S(y) = y_1**2 + 2*y_2 - prediction, with D_y S = [2*y_1, 2].
    score = ScoreMap(
        forward=lambda prediction, y: y[:, :1] ** 2 + 2 * y[:, 1:] - prediction,
        jacobian=lambda prediction, y: np.column_stack(
            [2 * y[:, 0], np.full(len(y), 2)]
        )[:, None, :],
    )
    cp = yemale.conformalize(
        [0, 0], [[1, 3], [3, 3]], score=score, target=[[-3], [2], [5]]
    )
    cpd = cp.predict([0, 0])  # Two scalar predictions, paired with the two outcomes.
    candidates = [[2, 4], [3, 4]]
    np.testing.assert_array_equal(cpd.transform(candidates), [[2], [5]])
    np.testing.assert_array_equal(
        cpd.potential_gradient(candidates), [[8, 4], [30, 10]]
    )


def test_region_indexing_and_paired_membership():
    cp = yemale.conformalize(np.zeros((19, 2)), np.arange(38).reshape(19, 2))
    predictions = np.array([[0, 1], [4, 5], [8, 9]])
    outcomes = predictions + [[0, 0], [10, 11], [100, 100]]
    cpd = cp.predict(predictions)
    region = cpd.region(0.9, rng=5)
    paired = region.contains(outcomes)
    for i in range(3):
        np.testing.assert_array_equal(
            region[i].contains(outcomes[i]), paired[i : i + 1]
        )
        np.testing.assert_array_equal(
            region[i].select(outcomes), outcomes[region[i].contains(outcomes)]
        )
    with pytest.raises(ValueError, match=r"regions\[i\]\.select"):
        region.select(outcomes)
    np.testing.assert_array_equal(region[:2].contains(outcomes[:2]), paired[:2])


def test_regions_select_original_labels():
    def score(predictions, outcomes):
        return predictions[:, :1] + (outcomes == "yes")

    predictions = np.linspace(-1, 1, 19)[:, None]
    outcomes = np.where(np.arange(19) % 2, "yes", "no")
    cp = yemale.conformalize(predictions, outcomes, score=score)

    def reference_set(u):
        return u[:, 0] >= 0

    region = cp.predict(0).region(reference_set=reference_set)
    candidates = np.array(["no", "yes", "yes", "no"])
    expected = cp.transport.region(reference_set).contains(
        (candidates == "yes")[:, None]
    )
    np.testing.assert_array_equal(region.contains(candidates), expected)
    np.testing.assert_array_equal(region.select(candidates), candidates[expected])
    with pytest.raises(ValueError):
        cp.predict(0).region(0.9, reference_set=reference_set)


def test_predictive_law_applies_inverse_score():
    score = ScoreMap(
        forward=lambda prediction, y: (y - prediction) / 2,
        inverse=lambda prediction, z: prediction + 2 * z,
        jacobian=lambda prediction, y: np.eye(1) / 2,
    )
    cp = yemale.conformalize([0.0], [0.0], score=score)
    # Uniform scores on [-1, 1] give Y = 3 + 2*Z, uniform on [1, 5].
    law = cp.transport.predictive_distribution(
        lambda points, labels: points,
        inverse=lambda points, labels: points,
        inverse_logabsdet=lambda points, labels: 0,
    )
    cpd = cp.predict(3.0, law=law)
    np.testing.assert_allclose(cpd.mean(), [3])
    np.testing.assert_allclose(cpd.cov(n_integration_points=512), [[4 / 3]], atol=2e-6)
    np.testing.assert_allclose(
        cpd.moment(2, n_integration_points=512), 31 / 3, atol=2e-6
    )
    np.testing.assert_allclose(cpd.expect(lambda y: y[:, 0]), 3)
    np.testing.assert_allclose(cpd.pdf([0, 2, 4, 6]), [0, 0.25, 0.25, 0])
    np.testing.assert_allclose(cpd.logpdf([2, 4]), -np.log(4))
    np.testing.assert_allclose(cpd.entropy(), np.log(4))
    assert cpd.density_region(0.8).contains([2, 3, 4]).all()
    batch = cp.predict([3.0, 5.0], law=law)
    np.testing.assert_allclose(batch.mean(), [[3], [5]])
    sample = batch.sample(8, rng=7)
    assert sample.shape == (2, 8, 1)
    assert np.all(np.abs(sample - np.array([3, 5])[:, None, None]) <= 2)
    assert not np.array_equal(sample[0] - 3, sample[1] - 5)
    generator = np.random.default_rng(7)
    np.testing.assert_array_equal(
        sample,
        np.stack([batch[i].sample(8, rng=generator) for i in range(2)]),
    )
    np.testing.assert_allclose(batch.cov(), cpd.cov()[None].repeat(2, axis=0))
    with pytest.raises(TypeError):
        cp.predict(0).mean()
    forward_only = yemale.conformalize([0.0], [0.0], score=score.forward)
    with pytest.raises(TypeError):
        forward_only.predict(0, law=law).mean()


def test_candidate_law_composes_nonlinear_score():
    score = ScoreMap(
        forward=lambda prediction, y: np.arcsinh(y - prediction),
        inverse=lambda prediction, z: prediction + np.sinh(z),
        jacobian=lambda prediction, y: (
            np.eye(2)[None] / np.sqrt(1 + (y - prediction) ** 2)[..., None]
        ),
    )
    rng = np.random.default_rng(8)
    predictions = rng.normal(size=(19, 2))
    outcomes = predictions + np.sinh(rng.normal(scale=0.4, size=(19, 2)))
    cp = yemale.conformalize(predictions, outcomes, score=score)
    prediction, candidate = np.array([2.0, -1.0]), np.array([2.4, -0.7])
    score_law = cp.transport.smooth().pullback(
        cp.transport.reference,
        candidate=score.forward(prediction[None], candidate[None]),
    )
    cpd = cp.predict(prediction, candidate=candidate)
    np.testing.assert_allclose(
        cpd.sample(8, rng=7), prediction + np.sinh(score_law.sample(8, rng=7))
    )
    np.testing.assert_allclose(cpd.mean(), prediction + score_law.expect(np.sinh))
    values = outcomes[:8]
    scores = score.forward(prediction[None], values)
    expected = cp.transport.smooth(0.4).log_density(scores)
    expected -= 0.5 * np.log1p((values - prediction) ** 2).sum(axis=1)
    np.testing.assert_allclose(cpd.log_density(values, temperature=0.4), expected)
    np.testing.assert_allclose(
        cp.predict(prediction).log_density(values, temperature=0.4), expected
    )
    with pytest.raises(ValueError):
        cp.predict(prediction, candidate=candidate, law=score_law)
    with pytest.raises(ValueError):
        cp.predict(predictions, candidate=outcomes)


def test_extend_delegates_to_a_plain_predictor():
    class Predictor:
        def predict(self, inputs, *, offset=0):
            return np.asarray(inputs)[:, 0] + offset

        def fit(self, inputs, outcomes):
            return self

    predictor = Predictor()
    model = yemale.extend(predictor)
    assert model.fit.__self__ is predictor
    with pytest.raises(RuntimeError):
        model.predict_region([[0]])
    inputs = np.arange(19)[:, None]
    outcomes = inputs[:, 0] + np.linspace(-1, 1, 19)
    assert model.conformalize(inputs, outcomes) is model
    region = model.predict_region([[1], [2]], 0.8, rng=5, offset=1)
    expected = model.conformal_.predict([2, 3]).region(0.8, rng=5)
    np.testing.assert_array_equal(region.contains([2, 3]), expected.contains([2, 3]))
    actual = model.predict_distribution([[1]], candidate=2.5, offset=1)
    expected = model.conformal_.predict(2, candidate=2.5)
    np.testing.assert_array_equal(actual.sample(8, rng=7), expected.sample(8, rng=7))
