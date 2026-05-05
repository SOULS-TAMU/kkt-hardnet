Equality Constraints
====================

Equality constraints use Python ``==`` comparisons.

.. code-block:: python

   model.constraints.add(
       y.y1 + y.y2 - x.x1 == 0,
       y.y2 - y.y3 - x.x2 == 0,
   )

Equality residuals enter the KKT projection with Lagrange multipliers.

Matrix-based equalities can be added from registered or extracted constants.
When the comparison is vector-valued, KKT-HardNet expands it into scalar
equality constraints.

.. code-block:: python

   model.extract("problem_matrices.npz")

   model.constraints.add(
       model.lin(model.Aeq, y) == model.lin(model.Beq, x) + model.beq,
   )

For smaller problems, constants can be registered directly:

.. code-block:: python

   Aeq = model.matrix([[1.0, 1.0, 0.0], [0.0, 1.0, -1.0]])
   Beq = model.matrix([[1.0, 0.0], [0.0, 1.0]])

   model.constraints.add(model.lin(Aeq, y) == model.lin(Beq, x))
