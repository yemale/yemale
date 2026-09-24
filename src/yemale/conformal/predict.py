"""Compose Dempster–Hill (DH) transport readouts with a score map."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from yemale import ot
from yemale._array import batch, readonly, rows


@dataclass(frozen=True)
class ScoreMap:
    r"""A score S_x and its optional inverse and outcome derivative.

    Callbacks receive the fixed model prediction at x. Write p, d, and s for
    the prediction, outcome, and score widths, and q for the number of outcomes.

    Args:
        forward: ``forward(predictions, outcomes)`` returns scores ``(q, s)``.
            Outcomes have shape ``(q, d)``; predictions have shape ``(q, p)``
            or ``(1, p)`` when one prediction accompanies many outcomes.
        inverse: Optional ``inverse(predictions, scores)`` returning outcomes
            ``(q, d)``. Used to convert score samples into outcome samples.
            It must invert the score on the chosen law's support.
        jacobian: Optional ``jacobian(predictions, outcomes)`` returning D_y S_x
            with shape ``(s, d)`` or ``(q, s, d)``. Required for
            ``potential_gradient`` and density. Density requires s = d and
            mutually inverse, differentiable score maps with nonsingular
            Jacobian at every queried outcome. For a restricted inverse branch,
            evaluate density on that branch's domain.
    """

    forward: Callable
    inverse: Callable | None = None
    jacobian: Callable | None = None

    def __repr__(self):
        if self is residual:
            return "residual"
        callbacks = (
            ("forward", self.forward),
            ("inverse", self.inverse),
            ("jacobian", self.jacobian),
        )
        arguments = ", ".join(
            f"{name}={getattr(callback, '__name__', type(callback).__name__)}"
            for name, callback in callbacks
            if callback is not None
        )
        return f"ScoreMap({arguments})"


def _residual(predictions, outcomes):
    if predictions.shape[-1] != outcomes.shape[-1]:
        raise ValueError("residual scores need predictions and outcomes of equal width")
    return np.subtract(outcomes, predictions, dtype=np.float64)


def _residual_inverse(predictions, scores):
    return predictions + scores


def _residual_jacobian(predictions, outcomes):
    return np.eye(outcomes.shape[-1])


residual = ScoreMap(_residual, _residual_inverse, _residual_jacobian)


def _points(value, dimension, name):
    """Keep a point or row batch; scalar-coordinate vectors may denote a batch."""
    value = np.asarray(value)
    if value.ndim == 0:
        value = value.reshape(1)
    if dimension == 1 and value.ndim == 1 and value.size != 1:
        value = value[:, None]
    if value.ndim not in (1, 2) or value.shape[-1] != dimension:
        raise ValueError(
            f"{name} must have shape ({dimension},) or (q, {dimension}); "
            f"got {value.shape}"
        )
    return value


def conformalize(predictions, outcomes, *, score=residual, target=None):
    """Fit once to scores of paired calibration predictions and outcomes.

    Args:
        predictions: Array ``(n, p)``; scalar predictions may use ``(n,)``.
        outcomes: Array ``(n, d)``; scalar outcomes may use ``(n,)``.
        score: ScoreMap or batch callable. Defaults to ``outcomes - predictions``.
            Return one score per pair, with shape ``(n, s)`` or ``(n,)`` for
            scalar scores. Only these scores must be numeric; s may differ
            from the prediction and outcome widths.
        target: Optional target passed to :func:`yemale.ot.fit`.

    Return a Conformalizer. The predictor and score must be fixed independently
    of calibration. DH coverage assumptions apply to the resulting scores.
    """
    predictions = batch(predictions, "predictions")
    outcomes = batch(outcomes, "outcomes")
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have the same number of rows")
    if callable(score):
        score = ScoreMap(score)
    if not isinstance(score, ScoreMap):
        raise TypeError("score must be a ScoreMap or a batch callable")
    scores = batch(score.forward(predictions, outcomes), "scores")
    if len(scores) != len(outcomes):
        raise ValueError("score must return one score per outcome")
    return Conformalizer(
        ot.fit(scores, target=target), score, predictions.shape[1], outcomes.shape[1]
    )


@dataclass(frozen=True, eq=False, repr=False)
class Conformalizer:
    """A fitted score transport; use ``cp.predict(prediction)`` for its readouts.

    Create with ``yemale.conformalize(predictions, outcomes)``.

    Attributes:
        transport: Shared DH Transport fitted to the calibration scores.
        score: ScoreMap used for calibration and subsequent outcomes.
    """

    transport: ot.Transport
    score: ScoreMap
    _prediction_dimension: int
    _outcome_dimension: int

    def __repr__(self):
        return f"Conformalizer(transport={self.transport!r})"

    def predict(self, prediction, *, candidate=None, law=None):
        """Bind one prediction or a batch, without refitting; return a CPD.

        For prediction width p, ``(p,)`` is one prediction and ``(batch, p)``
        is a batch. For p = 1, a scalar or ``[value]`` is one prediction;
        a longer vector is a batch of scalar predictions. ``[[value]]`` is
        a one-member batch, so select ``cpd[0]`` for single-prediction density.

        Regions, ranks and the forward ``density`` readout use the fitted
        transport directly. For sampling and summaries, choose one of:

        * ``candidate``: one outcome fixing the augmented smooth law at
          S_x(candidate), using the default temperature. Requires reference
          cells and an inverse score. Supports sample, expect, mean, cov and
          moment; ``pdf``, ``logpdf`` and entropy require the ``law`` path below.
        * ``law``: a score-space Law Pi of your choice. The inverse score
          converts its samples to outcomes. ``pdf`` and ``logpdf`` also require
          the law's density callbacks and the score Jacobian.

        The candidate stays fixed across draws. For equal-cell calibration,
        choose a law assigning mass 1 / (n + 1) to each fitted score cell.
        """
        prediction = _points(prediction, self._prediction_dimension, "prediction")
        if law is not None and not isinstance(law, ot.Law):
            raise TypeError("law must be a score-space Law")
        if law is not None and law.dimension != self.transport._target.shape[1]:
            raise ValueError("law dimension must match the fitted scores")
        cpd = CPD(self, readonly(prediction, dtype=prediction.dtype), law)
        if candidate is not None:
            if law is not None:
                raise ValueError("supply candidate or law, not both")
            if len(np.atleast_2d(prediction)) != 1:
                raise ValueError(
                    "candidate requires one prediction; pass one prediction row"
                )
            law = self.transport.smooth().pullback(
                self.transport._require_reference(), candidate=cpd._scores(candidate)
            )
            return replace(cpd, _score_law=law)
        return cpd


@dataclass(frozen=True, eq=False, repr=False)
class CPD:
    """Conformal predictive distribution (CPD) and readouts at given predictions.

    Create with ``cp.predict(prediction)``. Queries pair prediction and outcome
    rows, or use one prediction for many outcomes. ``cpd[i]`` selects a batch
    member and shares the fitted transport. Geometric queries keep the outcome row axis:
    for q outcomes, s score coordinates, and d outcome coordinates, ``transform``
    and ``sign`` return ``(q, s)``, ``rank`` and ``potential`` return ``(q,)``,
    and ``potential_gradient`` returns ``(q, d)``. A single outcome has q = 1.

    Sampling and summaries need an inverse score and either a candidate outcome
    or an explicit score law. Their shapes are documented on each method.
    """

    _cp: Conformalizer
    _prediction: np.ndarray
    _score_law: ot.Law | None = None

    def __repr__(self):
        size = 1 if self._prediction.ndim == 1 else len(self._prediction)
        return f"CPD(predictions={size}, law={self._score_law is not None})"

    @property
    def transport(self):
        """The shared DH transport on calibration scores."""
        return self._cp.transport

    def __getitem__(self, index):
        """Select predictions from a batch, sharing its transport and score law."""
        if self._prediction.ndim == 1:
            raise TypeError("this CPD contains one prediction, not a batch")
        if isinstance(index, tuple):
            raise IndexError("index prediction rows, not their coordinates")
        prediction = self._prediction[index]
        if prediction.ndim not in (1, 2):
            raise IndexError("index prediction rows, not their coordinates")
        return replace(self, _prediction=prediction)

    def _scores(self, outcomes):
        outcomes = np.atleast_2d(
            _points(outcomes, self._cp._outcome_dimension, "outcomes")
        )
        predictions = np.atleast_2d(self._prediction)
        if len(predictions) not in (1, len(outcomes)):
            raise ValueError("pair outcome rows with predictions, or select one cpd[i]")
        scores, _ = rows(
            self._cp.score.forward(predictions, outcomes),
            self.transport._target.shape[1],
            "scores",
        )
        if len(scores) != len(outcomes):
            raise ValueError("score must return one score per outcome")
        return scores

    def transform(self, outcomes):
        r"""Map outcomes to reference coordinates, one vector per outcome.

        .. math:: G_x(y)=T(S_x(y)).

        The vector's norm is ``rank``; its unit direction is ``sign``.
        """
        return self.transport(self._scores(outcomes))

    def rank(self, outcomes):
        r"""Return one center-outward rank per outcome, on the reference scale.

        .. math:: \operatorname{Rank}_x(y)=\operatorname{Rank}(S_x(y)).
        """
        return self.transport.rank(self._scores(outcomes))

    def sign(self, outcomes):
        r"""Return each outcome's reference-space direction; zero maps give zero.

        .. math:: \operatorname{Sign}_x(y)=\operatorname{Sign}(S_x(y)).
        """
        return self.transport.sign(self._scores(outcomes))

    def potential(self, outcomes):
        r"""Evaluate the potential through the score, one value per outcome.

        .. math:: y\mapsto\Phi(S_x(y)).
        """
        return self.transport.potential(self._scores(outcomes))

    def potential_gradient(self, outcomes):
        r"""Return the potential gradient in outcome coordinates.

        .. math:: H_x(y)=DS_x(y)^\top T(S_x(y)).

        This equals the gradient of ``potential`` where the score and DH
        potential are differentiable. At ties, it uses the DH-selected target.
        Requires ``ScoreMap.jacobian``; returns one vector per outcome.
        """
        score = self._cp.score
        if score.jacobian is None:
            raise TypeError("potential_gradient requires ScoreMap.jacobian")
        outcomes = np.atleast_2d(
            _points(outcomes, self._cp._outcome_dimension, "outcomes")
        )
        transformed = self.transform(outcomes)
        jacobian = np.asarray(score.jacobian(np.atleast_2d(self._prediction), outcomes))
        shape = (transformed.shape[1], outcomes.shape[1])
        if jacobian.shape not in (shape, (len(outcomes), *shape)):
            raise ValueError(
                f"score Jacobian must have shape {shape} or "
                f"{(len(outcomes), *shape)}; got {jacobian.shape}"
            )
        return np.einsum("...ij,...i->...j", jacobian, transformed)

    def region(self, coverage=None, *, reference_set=None, randomized=True, rng=None):
        r"""Return an outcome region by composing a DH region with S_x.

        For a hard reference set B:

        .. math:: \mathcal R_B(x)=\{y:T(S_x(y))\in B\}.

        Supply ``coverage`` (default 0.9) or ``reference_set``, not both.
        Reference sets are predicates on target coordinates, as in ``T.region``.
        With ``coverage``, randomization is on by default: draw once inside each
        reference cell and include its source cell if the draw lies inside the
        reference ball. This selection stays fixed across membership calls.
        Use ``rng`` to reproduce it, or ``randomized=False`` to select by cell
        centres instead, with whole-shell rounding as in ``T.quantile_region``.
        """
        if reference_set is not None:
            if coverage is not None or rng is not None or randomized is not True:
                raise ValueError(
                    "reference_set replaces coverage and randomization options"
                )
            region = self.transport.region(reference_set)
        else:
            region = self.transport.quantile_region(
                0.9 if coverage is None else coverage, randomized=randomized, rng=rng
            )
        return Region(self, region)

    def reference_distribution(self, outcome):
        r"""Return K_x(y, .) = K(S_x(y), .) for one candidate outcome.

        The returned Law draws reference points, not future outcomes.
        """
        return self.transport.reference_distribution(self._scores(outcome))

    def log_density(self, outcomes, *, temperature=None):
        """Return the forward smooth log-density readout, one value per outcome.

        Requires ``ScoreMap.jacobian`` and equal score/outcome dimensions.
        No candidate or inverse score is needed. ``temperature`` is passed
        to ``transport.smooth``; prediction batches use paired outcomes.
        """
        score = self._cp.score
        if score.jacobian is None:
            raise TypeError("density requires ScoreMap.jacobian")
        outcomes = np.atleast_2d(
            _points(outcomes, self._cp._outcome_dimension, "outcomes")
        )
        scores = self._scores(outcomes)
        if scores.shape[1] != outcomes.shape[1]:
            raise ValueError("density requires equal score and outcome dimensions")
        logdet = np.linalg.slogdet(
            score.jacobian(np.atleast_2d(self._prediction), outcomes)
        )[1]
        return self.transport.smooth(temperature).log_density(scores) + logdet

    def density(self, outcomes, *, temperature=None):
        r"""Evaluate the forward smooth density through the score.

        .. math:: p_{\tau,x}(y)=p_\tau(S_x(y))|\det D_yS_x(y)|.

        Here p_tau is ``transport.smooth(temperature).density``. Evaluate on
        a domain where the score is one-to-one and differentiable. Return
        one value per outcome, as in ``log_density``. This readout need not
        integrate to one or equal the PDF of the chosen sampling law.
        """
        return np.exp(self.log_density(outcomes, temperature=temperature))

    @cached_property
    def _law(self):
        if self._score_law is None:
            raise TypeError(
                "supply candidate=... or law=... with cp.predict(prediction, ...)"
            )
        score = self._cp.score
        if score.inverse is None:
            raise TypeError("distribution readouts require ScoreMap.inverse")
        if self._cp._outcome_dimension != self._score_law.dimension:
            raise ValueError(
                "Law composition requires equal score and outcome dimensions"
            )
        prediction = np.atleast_2d(self._prediction)
        # Sampling uses Y = S_x^{-1}(Z). Density uses the opposite direction:
        # p_Y(y) = p_Z(S_x(y)) * |det D_y S_x(y)|.
        backward = None
        if score.jacobian is not None:

            def backward(outcomes):
                return (
                    score.forward(prediction, outcomes),
                    np.linalg.slogdet(score.jacobian(prediction, outcomes))[1],
                )

        return self._score_law._map(
            lambda scores: score.inverse(prediction, scores), backward
        )

    @cached_property
    def _laws(self):
        return tuple(self[i]._law for i in range(len(self._prediction)))

    def _readout(self, method, *args, **kwargs):
        """Apply each prediction's law separately, then stack on the prediction axis."""
        if self._prediction.ndim == 1:
            return getattr(self._law, method)(*args, **kwargs)
        if not len(self._prediction):
            raise ValueError("distribution readouts need at least one prediction")
        return np.stack([getattr(law, method)(*args, **kwargs) for law in self._laws])

    def sample(self, size=1, *, rng=None):
        """Draw outcomes: ``(size, d)`` or ``(batch, size, d)`` for a CPD batch."""
        # Create one stream so batch members advance it instead of reseeding.
        return self._readout("sample", size, rng=np.random.default_rng(rng))

    def expect(self, function, *, n_integration_points=64, rng=None):
        r"""Compute E_Pi_x[f(Y)] = E_Pi[f(S_x^{-1}(Z))] through the chosen Law.

        ``function`` receives ``(q, d)`` outcomes. Numerical integration and its
        controls are those of :meth:`yemale.ot.Law.expect`.
        """
        # None requests fixed integration points; random batches share one stream.
        generator = None if rng is None else np.random.default_rng(rng)
        return self._readout(
            "expect", function, n_integration_points=n_integration_points, rng=generator
        )

    def mean(self, *, n_integration_points=64):
        """Return the outcome mean, shape ``(d,)`` or ``(batch, d)``."""
        return self._readout("mean", n_integration_points=n_integration_points)

    def cov(self, *, n_integration_points=64):
        """Return outcome covariance, shape ``(d, d)`` or ``(batch, d, d)``."""
        return self._readout("cov", n_integration_points=n_integration_points)

    def moment(self, powers, *, n_integration_points=64):
        """Return E[prod(Y**powers)] using the chosen Law's moment method."""
        return self._readout(
            "moment", powers, n_integration_points=n_integration_points
        )

    def entropy(self, *, n_integration_points=64):
        """Return outcome differential entropy in nats; requires density."""
        return self._readout("entropy", n_integration_points=n_integration_points)

    def logpdf(self, outcomes):
        """Return outcome log density for one prediction; index a CPD batch first.

        A point ``(d,)`` returns a scalar; a batch ``(*batch_shape, d)`` returns
        ``batch_shape``. For d = 1, a scalar or one-element vector is a point,
        and a longer vector is a batch. Density does not keep a singleton
        outcome axis unless it is present in the input batch shape.
        """
        if self._prediction.ndim != 1:
            raise ValueError("select one cpd[i] before evaluating its density")
        return self._law.logpdf(outcomes)

    def pdf(self, outcomes):
        """Return outcome density for one prediction; requires the score Jacobian.

        Input and output shapes follow ``logpdf``; index a CPD batch first.
        """
        if self._prediction.ndim != 1:
            raise ValueError("select one cpd[i] before evaluating its density")
        return self._law.pdf(outcomes)

    def density_region(self, mass, *, n_integration_points=256):
        """Return a density-mass region of one completed outcome law, not coverage."""
        if self._prediction.ndim != 1:
            raise ValueError("select one cpd[i] before constructing its density region")
        return self._law.density_region(mass, n_integration_points=n_integration_points)


@dataclass(frozen=True, eq=False, repr=False)
class Region:
    """Candidate outcomes whose scores belong to a shared DH region.

    Create with ``cpd.region(coverage)`` or ``model.predict_region(inputs, coverage)``.
    """

    _cpd: CPD
    _region: ot.Region

    def __repr__(self):
        return f"Region(coverage={self.coverage:.6g})"

    @property
    def coverage(self):
        """Marginal coverage reported by the underlying DH region."""
        return self._region.coverage

    def contains(self, outcomes):
        """Return one Boolean per outcome, pairing rows for a prediction batch."""
        return self._region.contains(self._cpd._scores(outcomes))

    def select(self, candidates):
        """Filter candidates for one region, preserving their values and order.

        For a batch, select one region with ``regions[i]`` first.
        """
        if len(np.atleast_2d(self._cpd._prediction)) != 1:
            raise ValueError("select one region first: regions[i].select(candidates)")
        candidates = np.asarray(candidates)
        accepted = self.contains(candidates)
        return candidates[accepted]

    def __getitem__(self, index):
        """Select predictions, retaining the same DH region and randomization."""
        return replace(self, _cpd=self._cpd[index])
