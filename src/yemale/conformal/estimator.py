"""Conformal readouts for any predictor with a ``predict`` method."""

from .predict import conformalize, residual


class Extended:
    """Add calibration and regions to a predictor; delegate its other methods.

    Create with ``yemale.extend(model)`` after fitting the predictor.
    """

    def __init__(self, model, *, score=residual):
        self._model = model
        self._score = score

    def conformalize(self, inputs, outcomes, *, target=None, **predict_kwargs):
        """Calibrate from predictions on observations not used to train the model."""
        predictions = self._model.predict(inputs, **predict_kwargs)
        self.conformal_ = conformalize(
            predictions, outcomes, score=self._score, target=target
        )
        return self

    def predict_distribution(
        self, inputs, *, candidate=None, law=None, **predict_kwargs
    ):
        """Bind the calibrated readouts to a batch of new predictions.

        For one prediction, supply a candidate outcome for smooth sampling and
        moments. Alternatively, supply a score-space ``law``. Without either,
        the CPD supports regions and geometric readouts.
        """
        if "conformal_" not in self.__dict__:
            raise RuntimeError("call conformalize(inputs, outcomes) first")
        predictions = self._model.predict(inputs, **predict_kwargs)
        return self.conformal_.predict(predictions, candidate=candidate, law=law)

    def predict_region(
        self,
        inputs,
        coverage=None,
        *,
        reference_set=None,
        randomized=True,
        rng=None,
        **predict_kwargs,
    ):
        """Return regions for new inputs, by coverage or a reference-set predicate."""
        return self.predict_distribution(inputs, **predict_kwargs).region(
            coverage,
            reference_set=reference_set,
            randomized=randomized,
            rng=rng,
        )

    def __getattr__(self, name):
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)


def extend(model, *, score=residual):
    """Add conformal readouts to a fitted predictor without changing its methods.

    Fit the model before extending it. Delegated methods keep their return
    values: a model's self-returning ``fit`` returns that model, not this wrapper.
    ``score(predictions, outcomes)`` is evaluated on whole arrays.
    """
    return Extended(model, score=score)
