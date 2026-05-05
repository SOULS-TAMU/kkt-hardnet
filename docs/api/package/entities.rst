Entities
========

In the package workflow, entities are the symbols that appear in a problem
definition.

Parameters and Variables
------------------------

- Parameters are registered with :meth:`kkthn.builder.KKTHardNet.add_parameter`.
- Variables are registered with :meth:`kkthn.builder.KKTHardNet.add_variable`.
- Access is expression-based, for example ``x.x1`` or ``y.y2``.
- Full vectors are available with ``x.vector()`` and ``y.vector()``.

Inverse Parameters
------------------

Inverse parameters are registered with
:meth:`kkthn.builder.KKTHardNet.add_inverse_parameter` and are learned during
``estimate()``.

Constants and Extracted Arrays
------------------------------

Constants are registered with :meth:`kkthn.builder.KKTHardNet.matrix`,
:meth:`kkthn.builder.KKTHardNet.vector`, and
:meth:`kkthn.builder.KKTHardNet.tensor`. A ``.npz`` file can be loaded with
:meth:`kkthn.builder.KKTHardNet.extract`; each array is then available as an
attribute on the model.

Autodoc
-------

.. autoclass:: kkthn.builder.Expression
   :members:
   :member-order: bysource
   :no-index:

.. autoclass:: kkthn.builder.VectorExpression
   :members:
   :member-order: bysource
   :no-index:

.. autoclass:: kkthn.builder.Constant
   :members:
   :member-order: bysource
   :no-index:

.. autoclass:: kkthn.builder._SymbolNamespace
   :members:
   :member-order: bysource
   :no-index:
