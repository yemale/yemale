# Mathematical guide

The transport assigns candidates to reference cells. A predictive distribution
also specifies how probability fills the corresponding cells in source space.
This guide connects the construction and formulas to the API.

For the candidate-augmented transport and conformal construction, see
[Beyond Uncertainty Sets](https://arxiv.org/abs/2511.15146).

- [Reference distribution](#reference-distribution)
- [Candidate-augmented transport](#candidate-augmented-transport)
- [Quantile regions, depth, and median](#quantile-regions)
- [Predictive distributions](#predictive-distributions)
- [Sampling and expectations](#sampling-and-expectations)
- [Moments, mean, and covariance](#moments-mean-and-covariance)
- [Density and entropy](#density-and-entropy)
- [Weighted distributions](#weighted-distributions)
- [Smoothing and custom targets](#smoothing-and-custom-targets)

Examples run in order. Mathematical cell indices run from $1$ to $n+1$;
Python labels run from `0` to `n`.

`ot.fit(source)` uses the default reference target, including its reference
law and quantile regions. `ot.fit(source, target=array)` retains the supplied
coordinates for transport, assignments, and smoothing. Rank and sign still
give each assigned target's radius and direction, but an array supplies no
reference-cell distribution.

| Symbol | Meaning | API |
| --- | --- | --- |
| $\nu$ | Default reference distribution | {py:class}`yemale.ot.Reference` |
| $L_j$ | Equal-probability reference cell | {py:meth}`yemale.ot.Reference.locate` |
| $m_j$ | Reference-cell center | {py:attr}`yemale.ot.Reference.centers` |
| $k(z)$ | Label assigned to candidate $z$ | {py:meth}`yemale.ot.Transport.label` |
| $T(z)$ | Center assigned to candidate $z$ | {py:class}`yemale.ot.Transport` |

## Reference distribution

Let $Z_1,\ldots,Z_n\in\mathbb R^d$ be the source points. The default target law
$\nu$ on the unit ball $\mathcal U$ has radius uniform on $[0,1]$ and an
independent uniform direction. Independently of the source points, partition
the ball into cells:

```math
\mathcal U=\bigsqcup_{j=1}^{n+1}L_j,\qquad
\nu(L_j)=\frac1{n+1}.
```

Write $U\sim\nu$ and $\nu_j=\nu(\,\cdot\mid L_j)$. The full law and cell centers are

```math
\nu=\frac1{n+1}\sum_{j=1}^{n+1}\nu_j,
\qquad m_j=\mathbb E_{\nu_j}[U].
```

```python
import numpy as np
import yemale.ot as ot

source = np.random.default_rng(0).normal(size=(99, 2))
T = ot.fit(source)
nu = T.reference
z = np.array([0.2, 0.4])

nu.centers     # m_j, one row per reference cell
nu.dimension   # number of coordinates per point
```

`nu` is continuous. Its `n + 1` cell centers are the fixed assignment targets.

## Candidate-augmented transport

Each candidate supplies the extra point in $\zeta(z)=(Z_1,\ldots,Z_n,z)$.
The augmented assignment $\sigma_z$ minimizes squared Euclidean cost;
see `help(ot.fit)` for its definition.

Its label, target, and assigned source cell are

```math
k(z)=\sigma_z(n+1),\qquad T(z)=m_{k(z)},\qquad
V_j^\dagger=\{z:k(z)=j\}.
```

On shared boundaries, `T.label` assigns each point to one cell, choosing the
smallest label for exactly equal computed scores. The assigned cells therefore
partition source space. Each query has its own augmented assignment and reuses
the fitted transport.

### One fit for every candidate

Let $\ell_j$ be the minimum cost of assigning the source points to every target
except $j$. Then

```math
k(z)\in\arg\min_j\{\|z-m_j\|^2+\ell_j\},\qquad
\phi_j=\frac{\|m_j\|^2+\ell_j}{2}.
```

Equivalently,

```math
\Phi(z)=\max_j\{\langle z,m_j\rangle-\phi_j\},\qquad
T(z)\in\partial\Phi(z).
```

One fit computes all leave-one costs; later candidates reuse them.
`T.phi` fixes a common additive constant; this does not affect labels or derivatives.

```python
T(z)
T.label(z)       # k(z), using zero-based indices
T.assignment(z)  # sigma_z: source points first, candidate last
T.potential(z)   # Phi(z)
T.phi            # phi_j in original source coordinates
T.evaluate(z)    # label, target, rank, sign, and potential in one dictionary
```

### Cell geometry

```python
A, b = T.halfspaces(T.label(z))  # closed cell: A @ z <= b
```

Closed cells can share boundaries. `T.label` determines which assigned cell
owns a boundary point.

## Quantile regions

If the source points and next candidate are exchangeable and the augmented
optimal assignment is almost surely unique,

```math
k(Z_{n+1})\sim\mathrm{Uniform}\{1,\ldots,n+1\}.
```

For a deterministic center-radius region with radius $r$, define

```math
J_r=\{j:\|m_j\|\leq r\},\qquad
\Omega_r=\{z:\|T(z)\|\leq r\}
=\bigcup_{j\in J_r}V_j^\dagger.
```

Its marginal coverage is $|J_r|/(n+1)$. This averages over the source points and
the next candidate; it is not a conditional guarantee for every fitted dataset.
For requested coverage $1-\alpha$, the deterministic rule chooses

```math
r_{\alpha,n+1}
=\inf\left\{r\in[0,1]:\frac{|J_r|}{n+1}\geq1-\alpha\right\}.
```

```python
deterministic = T.quantile_region(coverage=0.9, randomized=False)
```

Cells whose centers have the same radius enter together, so deterministic
coverage can exceed the request. `deterministic.radius` is the center-radius
cutoff and `deterministic.coverage` is the achieved level.

By default, `quantile_region` instead draws one independent point from each
reference cell, independently of the data. For requested coverage $c$, it returns

```math
W_j\sim\nu(\,\cdot\mid L_j),\qquad
\bigcup_{j:\|W_j\|\leq c}V_j^\dagger.
```

The reference radius is uniform on $[0,1]$, so this region has marginal coverage
$c$, averaging over the source points, the next candidate, and these draws.
It is not a conditional guarantee for a fitted dataset or a realized set of
draws. The selected cells are frozen when the region is constructed: repeated
membership queries, `labels`, and `halfspaces()` use the same selection.

```python
region = T.quantile_region(coverage=0.9, rng=0)
region.contains([[0.2, 0.4], [5.0, 5.0]])  # array([True, False])
region.select([[0.2, 0.4], [5.0, 5.0]])    # array([[0.2, 0.4]])
region.coverage                            # 0.9
region.radius                              # 0.9
region.labels
region.halfspaces()  # one (A, b) pair per closed cell
```

Pass an integer seed or a NumPy `Generator` as `rng` to control the draws.
Reusing the same integer seed gives nested regions as coverage increases;
separate unseeded constructions need not be nested.
For randomized regions, `region.radius` is the reference-ball radius $c$ and
`region.coverage` is the requested marginal level; the realized fraction
`len(region.labels) / (nu.n + 1)` can differ. This selection uses the sampled
points $W_j$, so it need not equal the deterministic region $\Omega_r$.

### Rank and depth

For the default unit-ball reference, depth is one minus the rank.

```python
T.rank(z)
T.sign(z)
depth = 1 - T.rank(z)
```

### Median set

As an advanced geometric summary, the convex potential defines the median set

```math
\mathcal M=\arg\min_{z}\Phi(z)=\partial\Phi^*(0),
\qquad
\Phi^*(u)=\sup_z\{\langle u,z\rangle-\Phi(z)\}.
```

This definition uses the whole subgradient. In particular, $\mathcal M$ can
remain nonempty when `T.quantile_region(0)` is empty.

With the default reference, the following linear program computes one median
point and the inequalities describing the whole set:

```math
\begin{aligned}
t_* &= \min_{z,t}\{t:\langle z,m_j\rangle-t\leq\phi_j\ \text{for every }j\},\\
\mathcal M&=\{z:\langle z,m_j\rangle\leq\phi_j+t_*\ \text{for every }j\}.
\end{aligned}
```

```python
from scipy.optimize import linprog

result = linprog(
    np.r_[np.zeros(nu.dimension), 1.0],
    A_ub=np.c_[nu.centers, -np.ones(nu.n + 1)],
    b_ub=T.phi,
    bounds=[(None, None)] * (nu.dimension + 1),
)
if not result.success:
    raise RuntimeError(result.message)
median_point = result.x[:-1]
median_A, median_b = nu.centers, T.phi + result.fun
```

## Predictive distributions

To obtain a distribution in source coordinates, specify how probability fills
the assigned source cells. When all assigned cells are nonempty, choose a law
within each cell:

```math
\lambda_j(V_j^\dagger)=1,\qquad
\Pi=\frac1{n+1}\sum_{j=1}^{n+1}\lambda_j.
```

The package specifies these laws through measurable maps

```math
Q_j:L_j\longrightarrow V_j^\dagger,\qquad
k(Q_j(u))=j\quad\text{for }\nu_j\text{-almost every }u.
```

Writing $Q(u)=Q_j(u)$ on $L_j$ gives

```math
\lambda_j=(Q_j)_\#\nu_j,\qquad \Pi=Q_\#\nu.
```

Here $Q_\#\nu$ means the distribution of $Q(U)$ when $U\sim\nu$.
The hard map $T$ sends each whole source cell to one center. Choose $Q$ to
specify a distribution within that cell. All maps satisfying the cell condition
above preserve

```math
\Pi\left(\bigcup_{j\in J}V_j^\dagger\right)=\frac{|J|}{n+1}.
```

### Example: uniform gaps in one dimension

Suppose the source points lie between 0 and 1, and choose [0, 1] as the support
for future source-space values. Choose a uniform law on each gap between
sorted source points and those bounds. If
$L_j=[a_j,b_j]$ and the corresponding gap has endpoints $c_j,d_j$, use

```math
Q_j(u)=c_j+\frac{d_j-c_j}{b_j-a_j}(u-a_j).
```

```python
source_1d = np.array([0.2, 0.5, 0.8])
T1 = ot.fit(source_1d[:, None])
nu1 = T1.reference
edges = np.r_[0.0, np.sort(source_1d), 1.0]
widths = np.diff(edges)

def uniform_gaps(points, labels):
    fraction = (points[:, 0] + 1) * len(widths) / 2 - labels
    return (edges[labels] + fraction * widths[labels])[:, None]

predictive = T1.predictive_distribution(map_from_reference=uniform_gaps)
```

In one dimension the reference is uniform on [-1, 1], split into equal intervals.
For zero-based label j and N = `len(widths)`, the cell endpoints are
-1 + 2j/N and -1 + 2(j + 1)/N. Thus `fraction` rescales that interval to [0, 1].

Supply bounds and within-gap laws appropriate to your problem. For unbounded
exterior gaps, choose proper tail distributions with total probability one.

### Reference-cell distributions

For quantities in reference coordinates, `T.reference_distribution(z)` returns
the reference law within the candidate's assigned cell:

```python
cell = T.reference_distribution(z)
cell.sample(1000, rng=0)  # reference points in L_{k(z)}
```

Under the calibration assumptions above, $W_z\sim\nu_{k(z)}$ gives
$W_{Z_{n+1}}\sim\nu$ marginally. These draws lie in the reference ball;
the map $Q$ defines the separate distribution in source coordinates.

## Sampling and expectations

Sampling from $\Pi$ has three steps:

```math
J\sim\mathrm{Uniform}\{1,\ldots,n+1\},\qquad
U\mid J=j\sim\nu_j,\qquad X=Q_J(U).
```

Omitting the final map samples from $\nu$.

```python
nu.sample(size=1000, rng=0)          # reference points, shape (1000, 2)
predictive.sample(size=1000, rng=0)  # predictive points, shape (1000, 1)
```

For integrable $f$, predictive expectations reduce to integration on the reference:

```math
\mathbb E_{\Pi}[f(X)]
=\frac1{n+1}\sum_{j=1}^{n+1}\mathbb E_{\nu_j}[f(Q_j(U))]
=\mathbb E_\nu[f\circ Q].
```

```python
def squared_norm(points):
    return np.sum(points**2, axis=-1)

nu.expect(squared_norm)
cell.expect(squared_norm)
predictive.expect(squared_norm)
```

### Numerical integration

For arbitrary functions, `expect` averages values at points within each cell.
The points are fixed by default; passing `rng` draws them randomly.
Each cell gets the same count, weighted by its probability when averaging.

```python
predictive.expect(squared_norm)         # fixed points
predictive.expect(squared_norm, rng=0)  # stratified random points
predictive.expect(squared_norm, n_integration_points=256)
```

For ordinary Monte Carlo, sample from the whole mixture and average:

```python
np.mean(squared_norm(predictive.sample(1000, rng=0)))
```

Both random estimators target the same expectation. Unlike stratified integration,
ordinary sampling gives random counts in each cell.

## Moments, mean, and covariance

```python
nu.mean()                  # zero vector, shape (2,)
nu.cov()                   # variances and covariances, shape (2, 2)
cell.mean()
cell.cov()
predictive.mean()          # array([0.5])
predictive.cov()           # variance, shape (1, 1)
```

For individual moments, give one nonnegative power per coordinate:

```python
nu.moment(powers=(2, 0))        # E[U_1^2]
cell.moment(powers=(1, 1))      # E[U_1 U_2] within one reference cell
predictive.moment(powers=(2,))  # E[X^2] in one dimension
```

`moment` returns raw moments; `cov()` centers coordinates by their means.
See `help(nu.moment)` for the definition and closed-form reference formula.

Reference and reference-cell statistics use analytic formulas. Mapped
statistics use fixed integration points, with `n_integration_points` setting
their number per cell. To estimate them randomly, use `expect` with `rng` and
the corresponding function.

## Density and entropy

### Reference density

`pdf` is density per unit volume; `logpdf` is its natural logarithm.

```python
u = cell.sample(1, rng=0)[0]
nu.pdf(u)
nu.logpdf(u)
cell.pdf(u)
cell.logpdf(u)
```

### Predictive density

For density, assume each $Q_j$ is continuously differentiable and one-to-one
on the cell interior, with a nonsingular Jacobian. Write $q_j=Q_j^{-1}$ on its image.
The conditional density is

```math
p_{\nu_j}(u)=(n+1)p_\nu(u)\,\mathbf 1_{L_j}(u).
```

The images lie in disjoint assigned source cells. The uniform mixture weight
cancels the conditional-density factor, giving almost everywhere:

```math
p_{\Pi}(x)=p_\nu(q_j(x))|\det Dq_j(x)|,
\qquad x\in Q_j(L_j).
```

Each map can cover a subset of its assigned source cell. Density is zero
outside the union of the images.

For the uniform-gap example, supply the inverse and its log absolute Jacobian
determinant:

```python
def inverse_gaps(points, labels):
    fraction = (points[:, 0] - edges[labels]) / widths[labels]
    return (2 * (labels + fraction) / len(widths) - 1)[:, None]

def inverse_logabsdet(points, labels):
    return np.log(2 / (len(widths) * widths[labels]))

predictive = T1.predictive_distribution(
    uniform_gaps, inverse=inverse_gaps, inverse_logabsdet=inverse_logabsdet
)
predictive.pdf([0.1, 0.4, 0.9])  # array([1.25, 0.83333333, 1.25])
predictive.logpdf([0.1, 0.4, 0.9])
```

Sampling and expectations need only the forward map.

### Entropy

Write $h$ for differential entropy, measured in nats.
For disjoint component distributions with densities,

```math
h(\Pi)=\log(n+1)+\frac1{n+1}\sum_{j=1}^{n+1}h(\lambda_j).
```

Under the change-of-variables assumptions above, when these terms are finite,

```math
h(\Pi)=h(\nu)+\frac1{n+1}\sum_{j=1}^{n+1}
\mathbb E_{\nu_j}[\log|\det DQ_j(U)|].
```

```python
nu.entropy()
cell.entropy()
predictive.entropy()
```

To evaluate a mapped law's entropy, supply the same inverse callbacks as for
its density.

## Weighted distributions

`T.predictive_distribution` gives all cells equal probability. `Law` also supports
other weights $w_j\geq0$ with $\sum_jw_j=1$:

```math
\nu^w=\sum_jw_j\nu_j,\qquad
\Pi^w=\sum_jw_j(Q_j)_\#\nu_j.
```

The superscript $w$ specifies the weights. With $w_j=1/(n+1)$, these laws
reduce to $\nu$ and $\Pi$.
Sampling chooses a cell using these weights before drawing within it.

```python
weights = [0.1, 0.2, 0.3, 0.4]
weighted_reference = ot.Law(nu1, weights=weights)
weighted_predictive = ot.Law(
    nu1,
    weights=weights,
    forward=uniform_gaps,
)
weighted_reference.sample(1000, rng=0)
weighted_predictive.expect(squared_norm)
```

The `Law` callback also takes `(points, labels)`. To select only some
cells, pass `cells=[...]` and one weight per selected cell.

For a cell-preserving map, the density and entropy become

```math
p_{\Pi^w}(x)=(n+1)w_jp_\nu(q_j(x))|\det Dq_j(x)|,
\qquad x\in Q_j(L_j),
```

```math
h(\Pi^w)=\sum_{j:w_j>0}w_j\left[
h(\nu_j)+\mathbb E_{\nu_j}[\log|\det DQ_j(U)|]-\log w_j
\right].
```

These formulas use the same regularity assumptions as above. For cell-preserving
maps, source-cell probabilities equal $w_j$; the equal-cell calibration of $\Pi$
corresponds to $w_j=1/(n+1)$.

## Smoothing and custom targets

Smoothing forms weighted averages of target centers. The smooth potential,
map, Jacobian, and inverse have their definitions in the corresponding methods.
`temperature` is $\tau>0$ in the units of the returned potential.

With `T.smooth()`, the default adapts to the fitted assignment margins. For
$a_j(v)=\langle v,m_j\rangle-\phi_j$ and its winning index $j_*(v)$, let

```math
\Delta(v)=a_{j_*(v)}(v)-\max_{j:\,\|m_j\|\ne\|m_{j_*(v)}\|}a_j(v),
\qquad
\tau=100\,\operatorname{median}\{\Delta(v):\Delta(v)>10^{-10}\}.
```

Probes resample the observations with 2% covariance-scaled Gaussian jitter,
using $\max(4000,n)$ points and seed 0. Jitter is computed in normalized source
coordinates, with a $10^{-9}I$ covariance ridge; the gap cutoff uses original
potential units.
The temperature is computed once per fit; if no gap qualifies, it falls back
to 0.05 times the source RMS scale. This is a smoothing heuristic, not a
coverage calibration. Pass `temperature=` to choose it directly:

```python
smooth = T.smooth(temperature=0.4)
mapped = smooth(z)
smooth.potential(z)
smooth.jacobian(z)
smooth.map_jacobian(z)  # map and Jacobian together
smooth.reference_distribution(z)  # Law with softmax weights
```

The reverse map blends all $n+1$ augmented sources $(Z_1,\ldots,Z_n,z)$,
using their hard potential values $\Phi(Z_i)$ and $\Phi(z)$. The supplied
candidate stays fixed while reference points vary. This dual-side map is
defined on the full reference space; it is not the numerical inverse of
the forward smooth map and need not preserve hard source-cell labels.

```python
smooth.inverse(nu.centers, candidate=z)
law = smooth.pullback(nu, candidate=z)
law.sample(1000, rng=0)
law.mean()
law.cov()
law.expect(squared_norm)

cell_law = smooth.pullback(ot.Law(nu, cells=[0]), candidate=z)
cell_law.mean()
```

These use $\Pi_\tau(\,\cdot\,;z)=Q_\tau(\,\cdot\,;z)_\#\nu$ and
$\lambda_j(\,\cdot\,;z)=Q_\tau(\,\cdot\,;z)_\#\nu_j$.
This candidate-based law provides sampling, expectations, and moments. For
`pdf` and `entropy`, construct a law with the density callbacks described above.

The separate forward-density readout belongs to the forward smooth map. It is
unnormalized, with integral $\nu(K)$, where $K$ is the target centers' convex hull.

```python
smooth.density(z)
smooth.log_density(z)
```

`Law` and `Reference` provide `density_region`. All points tied at the
cutoff are included.

Custom target arrays have shape `(n + 1, d)`:

```python
target = [[-0.75], [-0.25], [0.25], [0.75]]
custom = ot.fit(source_1d[:, None], target=target)
```

Their coordinates are retained as supplied. Arrays support assignment,
smoothing, and the assigned target's radius and direction through `rank` and
`sign`. A `Reference` target also supplies reference-cell distributions and
quantile regions.
