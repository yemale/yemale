# Optimal transport

Construct multivariate regions and ranks directly from a point cloud. This is
the Dempster–Hill (DH) layer, independent of models and conformal scores.

Fit `n` points once. Each query adds its candidate to an optimal assignment of
`n + 1` points to fixed targets under squared Euclidean cost, without solving
a new assignment problem.

## Regions

```python
import numpy as np
import yemale.ot as ot

points = np.random.default_rng(0).normal(size=(99, 2))
transport = ot.fit(points)
region = transport.quantile_region(0.9, rng=0)

candidates = np.array([[0.2, 0.4], [5.0, 5.0]])
region.contains(candidates)  # array([True, False])
region.select(candidates)    # array([[0.2, 0.4]])
region.coverage              # 0.9
```

Input arrays have shape `(n, d)`; scalar observations also accept `(n,)`.
One vector observation has shape `(1, d)`.
Regions are randomized by default; pass an integer seed such as `rng=0`
for reproducible selection. Coverage is marginal over observations and
randomization, under exchangeability and the assignment assumptions in the guide.

To choose a different region shape in reference space, supply a function
returning one Boolean per reference point. Test membership in the original
point coordinates:

```python
region = transport.region(lambda u: u[:, 0] >= 0)
region.contains(candidates)
```

## Transport, ranks and signs

```python
transport(candidates)  # Assigned reference centers.
transport.rank(candidates)  # Their radii: center-outward ranks.
transport.sign(candidates)  # Their directions.
```

## Reference distribution

The default reference distribution has uniform radius and independent uniform
direction in the unit ball. These operations return reference-space quantities:

```python
reference = transport.reference
reference.sample(size=1000, rng=0)  # Shape (1000, 2).
reference.mean()                    # array([0., 0.])
reference.cov()                     # Identity matrix divided by 6.
reference.moment((2, 0))            # E[U_1**2].

def squared_distance(points):
    return np.sum(points**2, axis=-1)

reference.expect(squared_distance)
```

## Sampling in source coordinates

Fix one candidate to construct a smooth law in the original point coordinates:

```python
law = transport.smooth().pullback(reference, candidate=[0.2, 0.4])
law.sample(1000, rng=0)
law.mean()
law.cov()
```

The law blends the observations and this fixed candidate. It provides sampling,
expectations and moments; the [smoothing page](../../../docs/smoothing.rst)
explains its relation to the transport.

To specify how probability fills each hard source cell, use
`transport.predictive_distribution(map_from_reference=...)`. This also returns
a `Law`, with the same summary methods. The guide's
[uniform-gap example](../../../docs/usage.md#example-uniform-gaps-in-one-dimension)
constructs such a law and supplies its density callbacks.

## Further reading

- [Transport API](../../../docs/transport.rst): ranks, signs, potentials and assignments.
- [Regions](../../../docs/regions.rst): coverage, randomization and reference sets.
- [Distributions](../../../docs/distributions.rst): densities, moments and expectations.
- [Predictive-law example](../../../docs/usage.md#example-uniform-gaps-in-one-dimension)
