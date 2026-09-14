Transport
=========

With the default target, ``T.reference`` is the Reference distribution used by
the fit. An array supplied through ``target=`` instead gives arbitrary-target
transport without reference-law or quantile-region features.

.. autofunction:: yemale.ot.fit

.. autoclass:: yemale.ot.Transport
   :members: __call__, label, rank, sign, evaluate, potential, phi, assignment, halfspaces
   :special-members: __call__

For regions and distributions, see :doc:`regions` and :doc:`distributions`.
For the candidate-augmented construction, see the :doc:`guide <usage>`.
