# yemale

Exact candidate-augmented transport for vector point clouds.

Given source points $Z_1, \ldots, Z_n$ and a query point $z$, yemale
adds $z$ to the source cloud and solves the optimal assignment of all
$n + 1$ points to fixed target barycentres $m_1, \ldots, m_{n+1}$, under
squared Euclidean cost. With the default target, these are barycentres of
equal-mass Dempster-Hill reference cells.

At fitting time, yemale computes the leave-one costs $\ell_j$: the optimal
cost of matching the $n$ source points while omitting target $m_j$. A query
then receives the label

`k(z) = argmin_j (||z - m_j||² + ℓ_j)`

and the deterministic transport is $T(z) = m_{k(z)}$. Thus each query is
evaluated as the added point in its own augmented assignment, without solving
a new assignment problem.

## Install

Until the first PyPI release, install directly from the main branch:

```bash
python -m pip install "git+https://github.com/yemale/yemale.git@main"
```

## Use

```python
import yemale.ot as ot

T = ot.fit(source)  # source: (n, d)
T(z)                # target barycentre m_{k(z)}
T.rank(z)           # its radius
T.sign(z)           # its direction
```

To supply target barycentres instead of the default reference construction:

```python
T = ot.fit(source, target=target)  # target: (n + 1, d)
```

`target` is expressed in the source-centred, globally scaled coordinates used
internally for the assignment. Returned target points retain the values you
provide.
