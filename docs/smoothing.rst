Smoothing
=========

Smoothing is optional. It is not needed for assignments or quantile regions.

.. automethod:: yemale.ot.Transport.smooth

.. autoclass:: yemale.ot.smoothing.SmoothMap
   :members: __call__, potential, jacobian, map_jacobian, density, log_density, density_region, reference_distribution, inverse, pullback
   :special-members: __call__
