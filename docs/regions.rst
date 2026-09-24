Quantile regions
================

Fit source points with ``transport = ot.fit(source)``, then request a central
90% region with ``transport.quantile_region(0.9)``.
See the :doc:`guide <usage>` for the construction and a worked example.

.. automethod:: yemale.ot.Transport.quantile_region

.. autoclass:: yemale.ot.QuantileRegion
   :members: contains, select, coverage, halfspaces
   :inherited-members:

Reference sets
--------------

For a different shape, choose a reference set ``B`` through membership.
The function receives reference points ``(q, d)`` and returns one Boolean per row.
``region.contains`` still receives source points: it tests ``B(transport(points))``.

.. code-block:: python

   def B(u):
       return u[:, 0] >= 0

   region = transport.region(B)
   points = [[0.2, 0.4], [-0.5, -0.3]]
   region.contains(points)  # Boolean mask
   region.select(points)    # Accepted rows, in input order

``region.coverage`` is the fraction of target centers in ``B``, not generally
its continuous reference probability :math:`\nu(B)`. For the marginal coverage
guarantee, choose ``B`` independently of the fitted observations.

Reference coordinates versus probability levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a continuous one-dimensional population with CDF :math:`F^\star`,
the center-outward map and the conversion of probability levels :math:`[a,b]` are

.. math::

   T^\star(z)=2F^\star(z)-1,\qquad
   B=[2a-1,\,2b-1].

The reference uses coordinates in :math:`[-1,1]`. The 5th-to-95th percentile
range therefore gives :math:`B=[-0.9,0.9]`; use
``transport.quantile_region(0.9)`` for a central 90% region with the default
randomization. In multiple dimensions, specify ``B`` through membership in
reference space.

.. automethod:: yemale.ot.Transport.region

.. autoclass:: yemale.ot.Region
   :members: contains, select, coverage, labels, halfspaces

Density regions
---------------

``law.density_region(mass=0.9)`` selects a density superlevel set. ``threshold``
is its density cutoff; ``mass`` is the numerically integrated probability at or
above that cutoff under the chosen law. Quantile-region ``coverage`` describes
marginal future-data coverage under the exchangeability and assignment assumptions.

.. autoclass:: yemale.ot.DensityRegion
   :members: contains, threshold
