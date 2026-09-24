# yemale

Add predictive regions and distribution summaries to your model, keeping its
point predictions and ordinary methods.

yemale uses center-outward ranks from optimal transport for multivariate
conformal prediction. Work with predictions and observed outcomes, or connect
your fitted model with `extend`. The independent optimal-transport engine also
works directly on point clouds.

## Install

Requires Python 3.10–3.14.

```bash
python -m pip install "git+https://github.com/yemale/yemale.git@main"
```

## With your model

Start with a fitted model and calibration data `X_cal, y_cal` that were not
used to train it.

```python
from yemale import extend

model = extend(model)
model.conformalize(X_cal, y_cal)

cpd = model.predict_distribution(X_new)
region = cpd.region(0.9)
region.contains(y_new)  # One Boolean per new outcome.
```

`model.predict(X_new)` still returns the original point predictions.
The default score is the residual, `outcomes - predictions`. Regions have
marginal coverage under the [score assumptions](docs/conformal.md#coverage).
For sampling and summaries,
[choose a predictive law](docs/conformal.md#predictive-laws).

Already have predictions? Use `yemale.conformalize(predictions, outcomes)`
directly; `extend` only connects this operation to your model.

## Explore

- [Conformal prediction](src/yemale/conformal/README.md): use a model or prediction
  arrays, test outcomes, and select candidates.
- [Optimal transport](src/yemale/ot/README.md): fit point clouds, construct regions,
  and work with reference and predictive distributions.
- [Mathematical guide](docs/usage.md): the construction and its connection to the API.

## License

Apache-2.0.
