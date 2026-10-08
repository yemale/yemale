# yemale

Prediction regions and predictive distributions for your model.

yemale adds uncertainty estimates to a fitted model without changing its
predictions. It handles one output or several outputs together, such as a
location's two coordinates. Choose a distribution to generate possible outcomes
and compute means and covariances.

Under the hood, it uses multivariate conformal prediction and optimal transport,
with scalar or vector-valued scores. The transport engine also works on its own.

## Install

Requires Python 3.10–3.14. The first PyPI alpha, `0.1.0a1`, is being prepared.
Until it is published, install from GitHub:

```bash
python -m pip install "git+https://github.com/yemale/yemale.git"
```

The API may change before 1.0; see the
[changelog](https://github.com/yemale/yemale/blob/main/CHANGELOG.md).

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
By default, yemale uses the prediction errors, `outcomes - predictions`.
Pass arrays of shape `(n,)` for one output or `(n, d)` for several outputs.
Regions use randomization; `rng=0` makes the example reproducible. The requested
coverage holds on average under the
[documented assumptions](https://yemale.github.io/conformal.html#coverage).

Already have predictions? Use `yemale.conformalize(predictions, outcomes)`
directly; `extend` only connects this operation to your model.

## Sampling and summaries

To generate possible outcomes, first choose a distribution. This example uses
`candidate=` to construct one for a single new input. The candidate is the
model's prediction and stays fixed while we draw samples:

```python
candidate = model.predict(X_new[:1])[0]
cpd = model.predict_distribution(X_new[:1], candidate=candidate)
cpd.sample(1000, rng=0)
cpd.mean()
cpd.cov()
```

You can also [choose your own distribution](https://yemale.github.io/conformal.html#predictive-laws)
with `law=`. The coverage guarantee for regions does not automatically apply
to the distribution used for sampling.

## Explore

- [Conformal prediction](https://yemale.github.io/conformal.html): use a model or prediction
  arrays, test outcomes, and select candidates.
- [Optimal transport](https://yemale.github.io/ot.html): fit point clouds, construct regions,
  and work with reference and predictive distributions.
- [Mathematical guide](https://yemale.github.io/usage.html): the construction and its connection to the API.

## License

Apache-2.0.
