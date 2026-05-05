Objective
=========

The package supports symbolic objectives built from registered parameters and
variables. Objectives are required for ``optimize()`` and optional for
``model()`` and ``estimate()``.

Examples
--------

.. code-block:: python

   model.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)

.. code-block:: python

   model.objective = (
       0.5 * (y.y1**2 + y.y2**2)
       + model.exp(y.y1)
       + x.x1 * y.y2
   )

Relevant methods
----------------

- :meth:`kkthn.builder.KKTHardNet.lin`
- :meth:`kkthn.builder.KKTHardNet.batch_lin`
- :meth:`kkthn.builder.KKTHardNet.quad`
- :meth:`kkthn.builder.KKTHardNet.batch_quad`
- :meth:`kkthn.builder.KKTHardNet.sin`
- :meth:`kkthn.builder.KKTHardNet.cos`
- :meth:`kkthn.builder.KKTHardNet.exp`
- :meth:`kkthn.builder.KKTHardNet.log`
- :meth:`kkthn.builder.KKTHardNet.sqrt`
- :meth:`kkthn.builder.KKTHardNet.abs`
