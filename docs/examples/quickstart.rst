Quick start
===========

Run this complete example in an empty directory after following :doc:`../INSTALL`.
The small training budget checks execution, not converged benchmark accuracy.

Problem and execution
---------------------

.. literalinclude:: quickstart.py
   :language: python
   :linenos:
   :end-before: # Check the declared physical/output contract

Check the result
----------------

These checks verify the physical/output contract. Where applicable they also
round-trip the trained model through the documented JAX metadata interface.

.. literalinclude:: quickstart.py
   :language: python
   :start-after: # Check the declared physical/output contract and model transport.

Download the complete script: :download:`quickstart.py <quickstart.py>`.
