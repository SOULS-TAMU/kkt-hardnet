Object
======

The primary object in the package is :class:`kkthn.builder.KKTHardNet`.

For the configuration associated with this object, see :doc:`hyperparameters`.

Role of the object
------------------

``KKTHardNet`` is responsible for:

- storing symbolic parameters, inverse parameters, and variables,
- recording objectives and constraints,
- attaching CSV datasets,
- training supervised, unsupervised, and inverse-estimation workflows,
- saving model artifacts,
- loading saved runs,
- predicting with JAX or native projection backends.

Autodoc
-------

.. autoclass:: kkthn.builder.KKTHardNet
   :members:
   :special-members: __init__
   :member-order: bysource
   :no-index:
