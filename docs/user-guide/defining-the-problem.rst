Defining the Problem
====================

KKT-HardNet defines constrained problems symbolically using named parameters,
decision variables, optional inverse parameters, an optional objective, and a
set of equality or inequality constraints.

The general form is:

.. math::

   \begin{aligned}
   \min_y &\, f(x,y) \\
   \textrm{s.t.} &\, h(x,y) = 0, \\
   &\, g(x,y) \le 0.
   \end{aligned}

Creating the Model Object
-------------------------

.. code-block:: python

   from kkthn import KKTHardNet

   TRAIN = {
       "epochs": 1000,
       "batch_size": 32,
       "learning_rate": 1e-3,
       "hidden_size": 64,
       "hidden_layers": 2,
   }

   model = KKTHardNet(name="Example_Model", train=TRAIN)

Defining Parameters and Variables
---------------------------------

Parameters are the problem inputs :math:`x`; variables are the decision
variables :math:`y`.

.. code-block:: python

   x = model.add_parameter(["x1", "x2"])
   y = model.add_variable(["y1", "y2", "y3"])

For larger models:

.. code-block:: python

   parameter_names = [f"x{i+1}" for i in range(50)]
   variable_names = [f"y{i+1}" for i in range(100)]

   x = model.add_parameter(parameter_names)
   y = model.add_variable(variable_names)

Inverse Parameters
------------------

Inverse parameters are unknown scalar quantities learned jointly with the
network weights during ``estimate()``.

.. code-block:: python

   theta = model.add_inverse_parameter(["a0", "a1"], init_value=[1.0, 1.0])

   model.constraints.add(
       theta.a0 * y.y1 + y.y2 - x.x1 == 0,
       y.y2 - theta.a1 * y.y3 - x.x2 == 0,
   )

Defining the Objective
----------------------

The objective is required for ``optimize()`` and optional for supervised
``model()`` and inverse ``estimate()`` workflows.

.. code-block:: python

   model.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)

Common nonlinear expression helpers are available:

- ``model.sin(expr)``
- ``model.cos(expr)``
- ``model.exp(expr)``
- ``model.log(expr)``
- ``model.sqrt(expr)``
- ``model.abs(expr)``

.. code-block:: python

   model.objective = (
       0.5 * (y.y1**2 + y.y2**2)
       + model.exp(y.y1)
       + x.x1 * y.y2
   )

Using Extracted Matrices
------------------------

Large models can keep coefficients in arrays instead of spelling out every
scalar expression. Use ``matrix(...)``, ``vector(...)``, and ``tensor(...)`` for
in-memory constants, or load a ``.npz`` file with ``extract(...)``. Extracted
arrays are also exposed as model attributes using the array names from the
file.

.. code-block:: python

   constants = model.extract("ed_column_matrices.npz")

   # If the file contains arrays named Aeq, Beq, beq, Cineq, and cineq,
   # they can be used as model attributes.
   model.constraints.add(
       model.lin(model.Aeq, y) == model.lin(model.Beq, x) + model.beq,
       model.lin(model.Cineq, y) <= model.cineq,
   )

The structured helpers expand vector comparisons into scalar constraints:

- ``model.lin(A, y)`` forms ``A @ y`` for vector or matrix ``A``.
- ``model.quad(Q, y)`` forms ``y.T @ Q @ y``.
- ``model.batch_lin(A, y)`` and ``model.batch_quad(Qs, y)`` create vector-valued expressions.

Defining Constraints
--------------------

Constraints are added as Python comparison expressions. Equalities use
``==``. Inequalities can use ``<=`` or ``>=``.

.. code-block:: python

   model.constraints.add(
       y.y1 + y.y2 - x.x1 == 0,
       y.y2 - y.y3 - x.x2 == 0,
       y.y1**2 + y.y3**2 <= 2.0,
       y.y1 >= 0,
   )

KKT-HardNet stores inequalities internally in a common residual form and folds
them into the projection system with slack and complementarity variables.

Bounds
------

The current high-level API represents bounds as ordinary inequalities:

.. code-block:: python

   model.constraints.add(
       y.y1 >= 0.0,
       y.y1 <= 1.0,
       y.y2 >= -1.0,
       y.y2 <= x.x1 + 2.0,
   )

Bounds can also be expressed from extracted vectors or affine maps:

.. code-block:: python

   model.extract("bounds.npz")

   model.constraints.add(
       y.vector() >= model.lower,
       y.vector() <= model.upper + model.lin(model.Ux, x),
   )

This keeps all feasibility requirements in the same symbolic constraint list.
