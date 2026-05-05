Box Constraints
===============

The current high-level KKT-HardNet API represents variable bounds as ordinary
inequality constraints.

.. code-block:: python

   model.constraints.add(
       y.y1 >= 0.0,
       y.y1 <= 1.0,
       y.y2 >= -1.0,
       y.y2 <= x.x1 + 2.0,
   )

These constraints are folded into the same projection system as other
inequalities.

Bounds can also come from extracted vectors, including parameter-dependent
affine bounds.

.. code-block:: python

   model.extract("bounds.npz")

   model.constraints.add(
       y.vector() >= model.lower,
       y.vector() <= model.upper,
       y.vector() <= model.upper_offset + model.lin(model.Upper_x, x),
   )
