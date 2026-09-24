Distributions
=============

Choose a distribution, then use its ``sample``, ``mean``, ``cov``, ``moment``
and ``expect`` methods. The coordinates of those results depend on the law:

- ``transport.reference`` describes reference coordinates in the unit ball.
- ``transport.smooth().pullback(reference, candidate=z)`` builds a smooth law
  in source coordinates, with ``z`` fixed across draws. See the :doc:`OT example <ot>`.
- ``transport.predictive_distribution(map_from_reference=...)`` specifies a
  distribution within each hard source cell. The :doc:`mathematical guide <usage>`
  constructs one with uniform density between scalar observations.

For conformal prediction, source coordinates are scores; the inverse score
converts their draws to outcomes. See :doc:`conformal`.

Source-space distributions
--------------------------

.. automethod:: yemale.ot.Transport.predictive_distribution

Reference distribution
----------------------

``transport.reference_distribution(point)`` selects the reference law within
the point's assigned cell. Its samples stay in reference coordinates.

``n`` is the number of fitted source points and ``dimension`` is the number of
coordinates. ``centers``, ``ranks``, and ``signs`` have one entry per reference
cell.

.. autofunction:: yemale.ot.reference

.. automethod:: yemale.ot.Transport.reference_distribution

.. autoclass:: yemale.ot.Reference
   :members: sample, pdf, logpdf, density_region, expect, moment, mean, cov, entropy, centers, ranks, signs, locate

Cell mixtures and mapped distributions
--------------------------------------

``reference`` is the underlying Reference. ``cells`` are its selected
zero-based labels and ``weights`` are their probabilities.

.. autoclass:: yemale.ot.Law
   :class-doc-from: both
   :members: sample, pdf, logpdf, density_region, expect, moment, mean, cov, entropy, dimension
