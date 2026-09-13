# Guide

The transport assigns candidates to reference cells. A predictive distribution
also specifies how probability fills the corresponding cells in observation space.

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

## Reference distribution

Let $Z_1,\ldots,Z_n\in\mathbb R^d$ be the observations. The default target law
$\nu$ is spherical-uniform on the unit ball $\mathcal U$:

```math
U=R\Theta,\qquad
R\sim\mathrm{Uniform}[0,1],\qquad
\Theta\sim\mathrm{Uniform}(\mathbb S^{d-1}),\qquad R\perp\Theta.
```

The radius is uniform, not the volume. In one dimension, $\nu$ is uniform on
$[-1,1]$. Independently of the observations, partition the reference into cells:

```math
\mathcal U=\bigsqcup_{j=1}^{n+1}L_j,\qquad
\nu(L_j)=\frac1{n+1}.
```

Write $\nu_j=\nu(\,\cdot\mid L_j)$. The full law and cell centres are

```math
\nu=\frac1{n+1}\sum_{j=1}^{n+1}\nu_j,
\qquad m_j=\mathbb E_{\nu_j}[U].
```

```python
import numpy as np
import yemale.ot as ot

observations = np.random.default_rng(0).normal(size=(99, 2))
T = ot.fit(observations)
nu = T.reference
z = np.array([0.2, 0.4])

nu.centers     # m_j, one row per reference cell
nu.dimension   # number of coordinates per point
```

`nu` is continuous. The assignment uses its `n + 1` cell centres, not a random
sample from it.

## Candidate-augmented transport

Each candidate supplies the extra point:

```math
\zeta(z)=(Z_1,\ldots,Z_n,z),\qquad
\sigma_z\in\arg\min_{\sigma\in\mathfrak S_{n+1}}
\sum_{i=1}^{n+1}\|\zeta_i(z)-m_{\sigma(i)}\|^2.
```

Its label, target, and assigned source cell are

```math
k(z)=\sigma_z(n+1),\qquad T(z)=m_{k(z)},\qquad
V_j^\dagger=\{z:k(z)=j\}.
```

Ties choose the smallest optimal label. Thus the assigned cells partition source
space. Each query has its own augmented assignment; it does not modify the fit.

### One fit for every candidate

Let $\ell_j$ be the minimum cost of assigning the observations to every target
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

In multiple dimensions, `fit` obtains all leave-one costs from one assignment
with a zero-cost auxiliary row and a shortest-path pass. In one dimension it
uses sorting. The auxiliary row is a computational device, not the candidate.
`T.phi` fixes a common additive constant; this does not affect labels or derivatives.

```python
T(z)
T.label(z)       # k(z), using zero-based indices
T.assignment(z)  # sigma_z: observations first, candidate last
T.potential(z)   # Phi(z)
T.phi            # phi_j in original source coordinates
T.evaluate(z)    # label, target, rank, sign, and potential in one dictionary
```

Queries use the original coordinates and accept `(d,)` or `(..., d)`. Scalar
outputs keep the batch axes; point outputs also keep the coordinate axis.
For scalar observations, fit an array `(n, 1)`.

Full assignments need `8 * q * (n + 1)` bytes for `q` candidates. Use `label`
when only the candidate's label is needed.

### Cell geometry

The closed cell associated with label $j$ is

```math
V_j=\bigcap_{k\ne j}
\{z:\langle z,m_k-m_j\rangle\leq\phi_k-\phi_j\}.
```

```python
A, b = T.halfspaces(T.label(z))  # closed cell: A @ z <= b
```

Closed cells can share boundaries. `T.label` determines which assigned cell
owns a boundary point.

## Quantile regions

Under exchangeability, no ties, and almost-sure uniqueness of the augmented
optimal assignment,

```math
k(Z_{n+1})\sim\mathrm{Uniform}\{1,\ldots,n+1\}.
```

For a radius $r$, define

```math
J_r=\{j:\|m_j\|\leq r\},\qquad
\Omega_r=\{z:\|T(z)\|\leq r\}
=\bigcup_{j\in J_r}V_j^\dagger.
```

Its marginal coverage is $|J_r|/(n+1)$. This averages over the observations and
the next candidate; it is not a conditional guarantee for every fitted dataset.
For requested coverage $1-\alpha$, choose

```math
r_{\alpha,n+1}
=\inf\left\{r\in[0,1]:\frac{|J_r|}{n+1}\geq1-\alpha\right\}.
```

```python
region = T.quantile_region(coverage=0.9)
region.contains([[0.2, 0.4], [5.0, 5.0]])  # array([True, False])
region.coverage                            # 0.9
region.radius
region.labels
region.halfspaces()  # one (A, b) pair per closed cell
```

The argument is coverage, not radius. Cells at the same radius enter together,
so achieved coverage can exceed the request.

### Rank and depth

For the default unit-ball reference,

```math
\mathrm{Rank}(z)=\|T(z)\|,\qquad
\mathrm{Sign}(z)=\frac{T(z)}{\|T(z)\|},\qquad
D(z)=1-\mathrm{Rank}(z).
```

The sign is zero when the target is zero. The finite rank takes discrete target
radii; it is not itself a uniform continuous random variable.

```python
T.rank(z)
T.sign(z)
depth = 1 - T.rank(z)
```

### Median set

The convex potential defines the set

```math
\mathcal M=\arg\min_{z}\Phi(z)=\partial\Phi^*(0),
\qquad
\Phi^*(u)=\sup_z\{\langle u,z\rangle-\Phi(z)\}.
```

This uses the whole subgradient, not just the branch selected by `T`. In
particular, `T.quantile_region(0)` can be empty even when $\mathcal M$ is not.

There is no median method. With the default reference, a linear program gives
one median point and the inequalities describing the whole set:

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

The reference law for a candidate is

```math
K(z,\cdot)=\nu_{k(z)}.
```

```python
cell = T.reference_distribution(z)
cell.sample(1000, rng=0)  # reference points in L_{k(z)}
```

Under the calibration assumptions above, $W_z\sim K(z,\cdot)$ gives
$W_{Z_{n+1}}\sim\nu$ marginally. These samples are reference coordinates,
not future observations.

When all assigned cells are nonempty, choose a law within each cell:

```math
\lambda_j(V_j^\dagger)=1,\qquad
\Pi^Z=\frac1{n+1}\sum_{j=1}^{n+1}\lambda_j.
```

The package specifies these laws through measurable maps

```math
Q_j:L_j\longrightarrow V_j^\dagger,\qquad
k(Q_j(u))=j\quad\text{for }\nu_j\text{-almost every }u.
```

Writing $Q(u)=Q_j(u)$ on $L_j$ gives

```math
\lambda_j=(Q_j)_\#\nu_j,\qquad \Pi^Z=Q_\#\nu.
```

Here $Q_\#\nu$ means the distribution of $Q(U)$ when $U\sim\nu$.
$Q$ is a modelling choice, not an inverse of the hard map $T$, which sends a
whole source cell to one centre. All valid choices preserve

```math
\Pi^Z\left(\bigcup_{j\in J}V_j^\dagger\right)=\frac{|J|}{n+1}.
```

### Example: uniform gaps in one dimension

Suppose observations and predictions lie between 0 and 1. Choose a uniform law
on each gap between sorted observations and those bounds. If
$L_j=[a_j,b_j]$ and the corresponding gap has endpoints $c_j,d_j$, use

```math
Q_j(u)=c_j+\frac{d_j-c_j}{b_j-a_j}(u-a_j).
```

```python
observations_1d = np.array([0.2, 0.5, 0.8])
T1 = ot.fit(observations_1d[:, None])
nu1 = T1.reference
edges = np.r_[0.0, np.sort(observations_1d), 1.0]
widths = np.diff(edges)

def uniform_gaps(points, labels):
    fraction = (points[:, 0] + 1) * len(widths) / 2 - labels
    return (edges[labels] + fraction * widths[labels])[:, None]

predictive = T1.predictive_distribution(map_from_reference=uniform_gaps)
```

The callback receives reference points `(q, d)` and cell labels `(q,)`, and
returns points `(q, d)` in the matching source cells. Inputs are read-only.
Here `fraction` runs from 0 to 1 within a reference cell.

The bounds and uniform gap laws are choices, not defaults. Unbounded exterior
gaps need a proper tail law; there is no uniform probability law on a half-line.

## Sampling and expectations

Sampling from $\Pi^Z$ has three steps:

```math
J\sim\mathrm{Uniform}\{1,\ldots,n+1\},\qquad
U\mid J=j\sim\nu_j,\qquad X=Q_J(U).
```

Omitting the final map samples from $\nu$.

```python
nu.sample(size=1000, rng=0)          # reference points, shape (1000, 2)
predictive.sample(size=1000, rng=0)  # predictive points, shape (1000, 1)
```

Within a reference cell,

```math
\mathbb E_{\nu_j}[f(U)]=(n+1)\int_{L_j}f(u)\,\nu(du).
```

For integrable $f$, the full reference and predictive expectations are

```math
\mathbb E_\nu[f(U)]
=\frac1{n+1}\sum_{j=1}^{n+1}\mathbb E_{\nu_j}[f(U)],
```

```math
\mathbb E_{\Pi^Z}[f(X)]
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

`function` receives `(q, d)` and returns `(q, ...)`. Averaging removes the
first axis and keeps any vector or matrix output axes.

### Numerical integration

For arbitrary functions, `expect` uses $M$ integration points per cell:

```math
\widehat{\mathbb E}_{\Pi^Z}[f(X)]
=\frac1{n+1}\sum_{j=1}^{n+1}\frac1M
\sum_{r=1}^{M}f(Q_j(u_{jr})).
```

By default, the points are fixed: interval midpoints in one dimension and
a Halton sequence mapped into each cell in higher dimensions. Passing `rng`
draws points independently inside every cell instead. This is stratified Monte
Carlo: each cell gets the same point count, with its probability applied when
averaging.

```python
predictive.expect(squared_norm)         # fixed points
predictive.expect(squared_norm, rng=0)  # stratified random points
predictive.expect(squared_norm, n_integration_points=256)
```

The default is 64 points per cell. A seed repeats the random draw; a NumPy
generator advances its state. For ordinary Monte Carlo, sample from the whole
mixture and average:

```python
np.mean(squared_norm(predictive.sample(1000, rng=0)))
```

Both random estimators target the same expectation. Unlike stratified integration,
ordinary sampling gives random counts in each cell.

## Moments, mean, and covariance

For a multi-index $\alpha=(\alpha_1,\ldots,\alpha_d)$ of nonnegative integers,

```math
U^\alpha=\prod_{r=1}^d U_r^{\alpha_r},\qquad
|\alpha|=\sum_{r=1}^d\alpha_r,\qquad
\mathrm{moment}(\alpha)=\mathbb E[U^\alpha].
```

For the full spherical-uniform reference, odd coordinate powers give zero.
Otherwise,

```math
\mathbb E_\nu[U^\alpha]
=\frac1{|\alpha|+1}
\frac{\Gamma(d/2)}{\Gamma((d+|\alpha|)/2)}
\prod_{r=1}^d\frac{\Gamma((\alpha_r+1)/2)}{\Gamma(1/2)}.
```

The reference cells separate radius and direction, so their conditional moments
also have closed forms. After a general map $Q_j$, use

```math
\mathbb E_{\Pi^Z}[X^\alpha]
=\frac1{n+1}\sum_{j=1}^{n+1}\mathbb E_{\nu_j}[Q_j(U)^\alpha].
```

```python
nu.moment(powers=(2, 2))       # E[U_1^2 U_2^2]
cell.moment(powers=(2, 2))     # the same powers within one reference cell
nu1.moment(powers=(34,))      # E[U^34] in one dimension
predictive.moment(powers=(34,))
```

The mean and covariance are

```math
\mathbb E_\nu[U]=0,\qquad \mathrm{Cov}_\nu(U)=\frac{I_d}{3d},
```

```math
\begin{aligned}
\mathbb E_{\Pi^Z}[X]
&=\frac1{n+1}\sum_{j=1}^{n+1}\mathbb E_{\nu_j}[Q_j(U)],\\
\mathrm{Cov}_{\Pi^Z}(X)
&=\mathbb E_{\Pi^Z}[XX^\top]
-\mathbb E_{\Pi^Z}[X]\mathbb E_{\Pi^Z}[X]^\top.
\end{aligned}
```

```python
nu.mean()                  # shape (2,)
nu.covariance()            # shape (2, 2)
cell.mean()
cell.covariance()
predictive.mean()          # array([0.5])
predictive.covariance()    # shape (1, 1)
```

Reference and reference-cell statistics use analytic formulas evaluated in
floating point. Mapped statistics use deterministic integration; increase
`n_integration_points` when needed. To estimate them randomly, use `expect`
with `rng` and the corresponding function.

## Density and entropy

### Reference density

With respect to volume, the full reference density is

```math
p_\nu(u)=\frac{\Gamma(d/2)}{2\pi^{d/2}}\|u\|^{1-d},
\qquad 0<\|u\|\leq1,
```

and it is zero outside the ball. At the origin it is infinite for $d>1$;
in one dimension it is $1/2$ throughout $[-1,1]$. Within a cell,

```math
p_{\nu_j}(u)=(n+1)p_\nu(u)\,\mathbf 1_{L_j}(u).
```

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
The change-of-variables formula gives, almost everywhere,

```math
p_{\lambda_j}(x)=p_{\nu_j}(q_j(x))|\det Dq_j(x)|,
\qquad x\in Q_j(L_j).
```

The images lie in disjoint assigned source cells. The uniform mixture weight
cancels the conditional-density factor:

```math
p_{\Pi^Z}(x)=p_\nu(q_j(x))|\det Dq_j(x)|,
\qquad x\in Q_j(L_j).
```

Density is zero outside the union of the images. A map need not fill the whole
source cell.

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

Both callbacks receive `(points, labels)`. The inverse returns `(q, d)` points;
the log determinant may be a scalar, `(q,)`, or `(q, 1)`. Sampling and expectations
need only the forward map.

Outside the mapped support, return a finite placeholder from `inverse` and
`-np.inf` from `inverse_logabsdet`.

### Entropy

Differential entropy is $h=-\mathbb E[\log p]$, in nats. For the reference,

```math
h(\nu)=\log\left(\frac{2\pi^{d/2}}{\Gamma(d/2)}\right)-(d-1).
```

For disjoint component distributions with densities,

```math
h(\Pi^Z)=\log(n+1)+\frac1{n+1}\sum_{j=1}^{n+1}h(\lambda_j).
```

Under the change-of-variables assumptions above, when these terms are finite,

```math
h(\Pi^Z)=h(\nu)+\frac1{n+1}\sum_{j=1}^{n+1}
\mathbb E_{\nu_j}[\log|\det DQ_j(U)|].
```

```python
nu.entropy()
cell.entropy()
predictive.entropy()
```

The implementation evaluates mapped entropy as `expect(-logpdf)`. It therefore
needs the inverse callbacks, even though the forward-Jacobian identity is another
way to compute the same quantity.

## Weighted distributions

`T.predictive_distribution` gives all cells equal probability. `Law` also supports
other weights $w_j\geq0$ with $\sum_jw_j=1$:

```math
\nu_w=\sum_jw_j\nu_j,\qquad
\Pi_w^Z=\sum_jw_j(Q_j)_\#\nu_j.
```

Sampling chooses $J\sim\mathrm{Categorical}(w_1,\ldots,w_{n+1})$ before
drawing within the selected cell. Expectations use

```math
\mathbb E_{\Pi_w^Z}[f(X)]=\sum_jw_j\mathbb E_{\nu_j}[f(Q_j(U))].
```

```python
weights = [0.1, 0.2, 0.3, 0.4]
weighted_reference = ot.Law(nu1, weights=weights)
weighted_predictive = ot.Law(
    nu1,
    weights=weights,
    forward=lambda labels, points: uniform_gaps(points, labels),
)
weighted_reference.sample(1000, rng=0)
weighted_predictive.expect(squared_norm)
```

The lower-level `Law` callback is `forward(labels, points)`. To select only some
cells, pass `cells=[...]` and one weight per selected cell.

For a cell-preserving map, the density and entropy become

```math
p_{\Pi_w^Z}(x)=(n+1)w_jp_\nu(q_j(x))|\det Dq_j(x)|,
\qquad x\in Q_j(L_j),
```

```math
h(\Pi_w^Z)=\sum_{j:w_j>0}w_j\left[
h(\nu_j)+\mathbb E_{\nu_j}[\log|\det DQ_j(U)|]-\log w_j
\right].
```

These formulas use the same regularity assumptions as above. Nonuniform weights
change source-cell probabilities; they do not retain the equal-cell calibration
of $\Pi^Z$.

## Smoothing and custom targets

For $\tau>0$, replace the maximum in $\Phi$ by

```math
\Phi_\tau(z)=\tau\log\sum_{j=1}^{n+1}
\exp\left(\frac{\langle z,m_j\rangle-\phi_j}{\tau}\right).
```

Its gradient blends the targets:

```math
p_{\tau,j}(z)=
\frac{\exp((\langle z,m_j\rangle-\phi_j)/\tau)}
{\sum_k\exp((\langle z,m_k\rangle-\phi_k)/\tau)},
\qquad
T_\tau(z)=\sum_jp_{\tau,j}(z)m_j.
```

The Jacobian and inverse are

```math
DT_\tau(z)=\frac1\tau\mathrm{Cov}_{p_\tau(z)}(m_j),
\qquad
Q_\tau(u)=\arg\min_z
\{\Phi_\tau(z)-\langle u,z\rangle\}.
```

When the targets affinely span $\mathbb R^d$, $Q_\tau=\nabla\Phi_\tau^*$ is the
inverse on the interior of their convex hull. The package computes it with Newton
iterations. `temperature` is $\tau$ in the units of the returned potential.

```python
smooth = T.smooth(temperature=0.4)
mapped = smooth(z)
smooth.inverse(mapped)
smooth.potential(z)
smooth.jacobian(z)
smooth.map_jacobian(z)  # map and Jacobian together
smooth.reference_distribution(z)  # Law with weights p_{tau,j}(z)
```

For a distribution $\eta$ supported inside this hull,
`smooth.pullback(eta)` gives $(Q_\tau)_\#\eta$. The full default reference extends
outside the hull, so it cannot be used directly. A smooth inverse is a valid
equal-cell predictive completion only if the target law gives each cell equal
mass and the inverse preserves its label.

Custom target arrays have shape `(n + 1, d)`:

```python
target = [[-0.75], [-0.25], [0.25], [0.75]]
custom = ot.fit(observations_1d[:, None], target=target)
```

Their coordinates are retained as supplied. Arrays support assignment and
smoothing; a `Reference` target also supplies reference-cell distributions and
quantile regions.
