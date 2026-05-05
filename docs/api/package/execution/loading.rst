Loading
=======

Saved runs are restored from ``metadata.json``.

.. code-block:: python

   from kkthn import KKTHardNet

   model = KKTHardNet().load("path/to/metadata.json")

Loading restores model metadata, trained weights, the JAX projection path, and
the native projection backend when a compatible artifact is available.
