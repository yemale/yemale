Quantile regions
================

Fit source points with :func:`~yemale.ot.fit`, then request a region with
``T.quantile_region(coverage=0.9)``.
See the :doc:`guide <usage>` for the construction and a worked example.

``radius`` is the largest included reference-centre radius. ``labels`` are the
zero-based labels of the included cells.

.. automethod:: yemale.ot.Transport.quantile_region

.. autoclass:: yemale.ot.QuantileRegion
   :members: contains, coverage, halfspaces

Density regions
---------------

``law.density_region(mass=0.9)`` selects points by density. Its mass is a
numerical integral, not a future-data coverage guarantee. Smooth maps offer
the same operation on their unnormalized density; see :doc:`smoothing`.

``threshold`` is the density cutoff. ``mass`` is the numerical mass included
at that cutoff.

.. autoclass:: yemale.ot.DensityRegion
   :members: contains, threshold
