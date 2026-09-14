Distributions
=============

``T.reference`` is the reference distribution. To obtain a distribution in
source space, supply a map to ``T.predictive_distribution``.
The :doc:`guide <usage>` walks through a complete example.

Construct a distribution
------------------------

.. autofunction:: yemale.ot.reference

.. automethod:: yemale.ot.Transport.reference_distribution

.. automethod:: yemale.ot.Transport.predictive_distribution

Reference distribution
----------------------

``n`` is the number of fitted source points and ``dimension`` is the number of
coordinates. ``centers``, ``ranks``, and ``signs`` have one entry per reference
cell.

.. autoclass:: yemale.ot.Reference
   :members: sample, pdf, logpdf, density_region, expect, moment, mean, covariance, entropy, centers, ranks, signs, locate

Cell mixtures and mapped distributions
--------------------------------------

``reference`` is the underlying Reference. ``cells`` are its selected
zero-based labels and ``weights`` are their probabilities.

.. autoclass:: yemale.ot.Law
   :class-doc-from: both
   :members: sample, pdf, logpdf, density_region, expect, moment, mean, covariance, entropy, dimension
