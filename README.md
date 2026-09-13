# yemale

Exact candidate-augmented transport for vector point clouds.

Fit `n` source points once. Each query adds one candidate to an optimal
assignment of `n + 1` points to fixed target barycentres under squared Euclidean
cost, without solving a new assignment problem. The default targets are
barycentres of equal-mass Dempster–Hill reference cells.

## Install

Requires Python 3.10–3.14.

```bash
python -m pip install "git+https://github.com/yemale/yemale.git@main"
```

## Use

```python
import yemale.ot as ot

source = [[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5]]  # shape (n, d)
z = [0.2, 0.4]

T = ot.fit(source)
T(z)             # assigned target barycentre
T.rank(z)        # its radius
T.sign(z)        # its direction
T.assignment(z)  # target indices for the source points, followed by z
```

The queries above accept batches. Each candidate gets its own `n + 1` assignment.

`T.law(z)` gives the probability law inside the assigned reference cell.
`T.smooth()` gives a smooth transport map.

To supply target barycentres instead of the default reference construction:

```python
T = ot.fit(source, target=target)  # target: (n + 1, d)
```

`target` is expressed in the source-centred, globally scaled coordinates used
internally for the assignment. Returned target points retain the values you
provide.

## License

Apache-2.0.
