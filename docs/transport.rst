Transport
=========

``transport = ot.fit(source)`` returns a :class:`~yemale.ot.Transport`.
Call ``transport(points)`` to map points to target coordinates.

.. autofunction:: yemale.ot.fit

.. autoclass:: yemale.ot.Transport
   :members: __call__, label, rank, sign, evaluate, potential, phi, assignment, halfspaces
   :special-members: __call__

For regions and distributions, see :doc:`regions` and :doc:`distributions`.
For the candidate-augmented construction, see the :doc:`guide <usage>`.
