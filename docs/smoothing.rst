Smoothing
=========

``transport.smooth()`` blends target centers. Its ``density`` and ``log_density``
evaluate the forward map's unnormalized change-of-variables density.
``pullback(..., candidate=...)`` instead maps a reference law to source coordinates
using the fixed candidate. It returns a ``Law`` for sampling and moments.
These are distinct constructions: the forward density is not the pullback law's
PDF. See the :doc:`mathematical guide <usage>` for examples and formulas.

.. automethod:: yemale.ot.Transport.smooth

.. autoclass:: yemale.ot.smoothing.SmoothMap
   :members: __call__, potential, jacobian, map_jacobian, density, log_density, reference_distribution, inverse, pullback
   :special-members: __call__
