# yemale

**Build prediction regions. Generate possible outcomes. Evaluate their consequences.**

yemale builds multivariate conformal predictive distributions from predictions
and observed outcomes. Use them to construct prediction regions, draw samples,
assign probabilities to events, and compute expectations—for example, the
expected cost of a decision.

The conformal construction provides finite-sample calibration without assuming
a particular data distribution.

It uses optimal transport to handle scalar and vector-valued scores, without
first reducing vectors to a scalar summary. The transport engine also works
independently.

## Install

```bash
pip install yemale
```

Alpha release.

## With your model

Start with a fitted model and held-out data `X_cal, y_cal` that were not used
to train it. These data are used to calibrate the prediction regions.

```python
from yemale import extend

model = extend(model)
model.conformalize(X_cal, y_cal)

cpd = model.predict_distribution(X_new)
region = cpd.region(0.9, rng=0)
region.contains(y_new)  # One Boolean per new outcome.
```

`model.predict(X_new)` still returns the original point predictions.

Already have predictions? Use `yemale.conformalize(predictions, outcomes)`.

## Sampling and summaries

Use the model's prediction as the fixed candidate for sampling:

```python
candidate = model.predict(X_new[:1])[0]
cpd = model.predict_distribution(X_new[:1], candidate=candidate)
cpd.sample(1000, rng=0)
cpd.mean()
cpd.cov()
```

## Explore

- [Conformal prediction](https://yemale.github.io/conformal.html): use a model or prediction
  arrays, test outcomes, and select candidates.
- [Optimal transport](https://yemale.github.io/ot.html): fit point clouds, construct regions,
  and work with reference and predictive distributions.
- [Mathematical guide](https://yemale.github.io/usage.html): the construction and its connection to the API.
- [Changelog](https://github.com/yemale/yemale/blob/main/CHANGELOG.md).

## License

Apache-2.0.
