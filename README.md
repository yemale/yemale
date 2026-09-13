# yemale

Multivariate ranks and quantile regions with candidate-augmented optimal transport.

Fit `n` observations once. Each query adds its candidate to an optimal assignment
of `n + 1` points, without solving a new assignment problem.

## Install

Requires Python 3.10–3.14.

```bash
python -m pip install "git+https://github.com/yemale/yemale.git@main"
```

## Quantile regions

```python
import numpy as np
import yemale.ot as ot

observations = np.random.default_rng(0).normal(size=(99, 2))
T = ot.fit(observations)
region = T.quantile_region(coverage=0.9)

region.contains([[0.2, 0.4], [5.0, 5.0]])  # array([True, False])
region.coverage                            # 0.9
```

Queries accept one point or a batch. `region.coverage` gives the achieved level,
which can exceed the request when whole cells are included together.
See the [guide](docs/usage.md#quantile-regions) for coverage assumptions.

## Distributions

The reference distribution has uniform radius and direction in the unit ball.

```python
nu = T.reference
nu.sample(size=1000, rng=0)
nu.moment(powers=(2, 2))  # exact E[U_1^2 U_2^2]
nu.covariance()
```

To sample in the observed space, choose how probability fills each assigned cell.
The [predictive distribution example](docs/usage.md#predictive-distributions)
shows that choice, sampling, and expectations in one runnable workflow.

The [guide](docs/usage.md) also covers ranks, assignments, potentials, and smoothing.

## License

Apache-2.0.
