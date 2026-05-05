Building
========

KKT-HardNet builds the internal JAX model when ``model()``, ``optimize()``, or
``estimate()`` is called. Users define symbols, objective, constraints, and data
before calling the selected training method.

.. code-block:: python

   model.dataset(parameters="parameters.csv", variables="variables.csv")
   result = model.model()
