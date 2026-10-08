# Conformal prediction

Build prediction regions with scalar or vector-valued scores, and choose a
predictive law for sampling and summaries. Start with a fitted model or with
prediction arrays and their observed outcomes.

## Use your model

Given a fitted model and calibration data not used to train it:

```python
from yemale import extend

model = extend(model)
model.conformalize(X_cal, y_cal)
cpd = model.predict_distribution(X_new)

region = cpd.region(0.9)
region.contains(y_new)  # Test outcomes, paired with the new inputs.
region.coverage        # 0.9
```

`model.predict` keeps returning point predictions. After changing or retraining
the model, calibrate again. Regions are randomized by default; pass `rng=0`
to `region` for reproducibility. Membership stays fixed once the region is built.

## Use prediction arrays

The model wrapper calls the same array-based API. For example, with scalar
predictions and observed outcomes:

```python
import numpy as np
from yemale import conformalize

rng = np.random.default_rng(0)
predictions = rng.normal(size=99)
outcomes = predictions + rng.normal(scale=0.5, size=99)

cp = conformalize(predictions, outcomes)
cpd = cp.predict([0.2, 1.0])
region = cpd.region(0.9, rng=0)
region.contains([0.3, 1.2])  # One outcome for each prediction.

candidates = np.linspace(-2, 2, 100)
region[0].select(candidates)  # Keep candidates for the first prediction.
```

Calibration accepts `(n,)` scalar arrays or `(n, d)` vector arrays. A prediction
batch has one region per row; indexing selects a region, and `select` returns
the original candidate outcomes.

## Scores

The default score is `outcomes - predictions`. Supply `score=` to
`conformalize` or `extend` for another batch function: it receives predictions
and outcomes and returns one numeric scalar or vector score per pair. The
transport is fitted once to those scores and reused for every prediction.

## Sampling and summaries

A score law describes the distribution of scores; the inverse score converts
its samples into outcomes. For residuals, this simply adds the prediction.
For one prediction, supply `candidate=` to build a smooth law from the
calibration scores and that candidate's score:

```python
cpd = cp.predict(0.2, candidate=0.3)
cpd.sample(1000, rng=0)
cpd.mean()
cpd.cov()
```

The candidate stays fixed while samples vary. This path uses the default OT
temperature and provides sampling, expectations and moments. For a custom
score, supply its forward and inverse functions in a `ScoreMap`.

To choose the score distribution yourself, pass `law=` through
`cp.predict(prediction, law=score_law)` or
`model.predict_distribution(inputs, law=score_law)`. Density and entropy use
this path with the [density callbacks](../../../docs/conformal.md#density-and-entropy).

## Further reading

- [API and coverage assumptions](../../../docs/conformal.md#coverage)
- [Choosing a predictive law](../../../docs/usage.md#predictive-distributions)
- [The independent OT engine](../ot/README.md)
