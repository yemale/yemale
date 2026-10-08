# Conformal prediction

```{include} ../src/yemale/conformal/README.md
:start-after: "# Conformal prediction"
:end-before: "## Further reading"
:relative-docs: ../../../docs
```

## Coverage

Keep the predictor and score fixed independently of calibration. Under the
exchangeability and augmented-assignment uniqueness assumptions in the
[guide](usage.md#quantile-regions), coverage averages over calibration samples,
future observations and the region's randomization. This is marginal coverage.

## Shapes

At prediction time, a vector `(width,)` binds one prediction; a matrix
`(batch, width)` keeps a prediction-batch axis. Scalars accept `3` or `[3]`
for one prediction and `[3, 5]` for two. `[[3]]` keeps a one-row batch axis.

Membership pairs prediction and outcome rows. Use `cpd[i]` to select one
prediction and test a collection of candidates. Geometric readouts keep an
outcome row axis: for q outcomes and s score coordinates, `transform` and
`sign` return `(q, s)`; `rank`, `potential` and `region.contains` return `(q,)`.

For a custom shape, `cpd.region(reference_set=B)` uses the same reference-space
predicate as `cp.transport.region(B)`. See [OT regions](regions.rst).

## Predictive laws

In the OT layer, the fitted source points are the calibration scores.
A score law $\Pi$ becomes an outcome law through the inverse score:

```math
\Pi_x=(S_x^{-1})_\#\Pi,
\qquad \mathbb E_{\Pi_x}[f(Y)]
=\mathbb E_\Pi[f(S_x^{-1}(Z))].
```

The `candidate=y` shortcut uses the OT smooth pullback at $S_x(y)$:

```math
\Pi_{x,\tau}(\,\cdot\,;y)
=\left[S_x^{-1}\circ Q_\tau(\,\cdot\,;S_x(y))\right]_\#\nu.
```

Here $y$ is fixed across draws. To specify a law within each hard score cell,
use `cp.transport.predictive_distribution` and pass the result as `law=`.
The [uniform-gap example](usage.md#example-uniform-gaps-in-one-dimension)
shows a complete construction. Equal-cell calibration requires mass
$1/(n+1)$ in every fitted score cell; the smooth construction need not preserve
these masses. Regions are determined directly by the fitted transport,
independently of the sampling law.

Supply `ScoreMap.inverse` on the chosen law's support, preserving coordinate
dimension. Batched summaries keep a leading prediction axis; `cpd.sample(size)`
returns `(batch, size, d)` for outcome width d.

## Density and entropy

The forward density uses the smooth transport and the score Jacobian:

```math
p_{\tau,x}(y)=p_\nu(T_\tau(S_x(y)))
\left|\det DT_\tau(S_x(y))\right|
\left|\det D_yS_x(y)\right|.
```

```python
density = cpd.density(outcomes)
```

This needs no candidate or inverse score. The residual score already provides
the Jacobian; custom scores provide `ScoreMap.jacobian` and must be one-to-one
on the evaluation domain. `log_density` returns the same readout in log scale.
Sampling and moments still use $Q_\tau$ through `Law`. The forward density is
not generally normalized or exactly the PDF of those samples.

For `pdf`, `logpdf` and `entropy`, supply a score law with density callbacks
through `law=`, and the score Jacobian through `ScoreMap.jacobian`.
The residual score already provides this Jacobian.
For mutually inverse differentiable score maps with nonsingular outcome Jacobian,

```math
p_{\Pi_x}(y)=p_\Pi(S_x(y))\left|\det D_y S_x(y)\right|.
```

Evaluate densities on a domain where these inverse conditions hold. For a
mapped score law, its density also requires the inverse of its reference-to-score
map and that inverse's log-Jacobian, as in the
[guide](usage.md#predictive-density). Entropy has the same requirements.

Use one prediction, or `cpd[i]`, for `pdf`, `logpdf` and `density_region(mass=0.9)`.
The latter selects probability mass under the supplied law;
`cpd.region(0.9)` targets marginal coverage of future outcomes.
The forward readouts `density` and `log_density` also accept paired batches.

## Potential and gradient

`cpd.potential(outcomes)` evaluates the fitted transport potential through the
score. `cpd.potential_gradient(outcomes)` uses `ScoreMap.jacobian` to express
its gradient in outcome coordinates. Both operate directly on the transport.

## API

```{eval-rst}
.. autofunction:: yemale.conformalize

.. autofunction:: yemale.extend

.. autoclass:: yemale.conformal.ScoreMap

.. autoclass:: yemale.conformal.Conformalizer
   :members: predict

.. autoclass:: yemale.conformal.CPD
   :members: transform, rank, sign, potential, potential_gradient, region, reference_distribution, density, log_density, sample, expect, mean, cov, moment, entropy, pdf, logpdf, density_region, transport, __getitem__

.. autoclass:: yemale.conformal.Region
   :members: contains, select, coverage, __getitem__

.. autoclass:: yemale.conformal.Extended
   :members: conformalize, predict_distribution, predict_region
```
