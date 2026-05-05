Inequality Constraints
======================

Inequality constraints use Python ``<=`` or ``>=`` comparisons.

.. code-block:: python

   model.constraints.add(
       y.y1**2 + y.y3**2 <= 2.0,
       y.y1 >= 0.0,
   )

Inequality residuals are represented with slack and complementarity blocks in
the projection layer.

Inequality systems can also be built from extracted matrices or tensors.

.. code-block:: python

   model.extract("problem_matrices.npz")

   model.constraints.add(
       model.lin(model.Cineq, y) <= model.dineq + model.lin(model.Dineq, x),
       model.batch_quad(model.Qineq, y) + model.batch_lin(model.qineq, y) <= model.beta,
   )

The structured expressions above create one inequality per row or tensor slice.
