Predicting
==========

Use ``predict(...)`` after training or loading.

.. code-block:: python

   y_native = model.predict(sample_x, projection_backend="native")
   y_jax = model.predict(sample_x, projection_backend="jax")

Available backends:

- ``auto``: native if available, otherwise JAX.
- ``jax``: JAX projection path.
- ``native``: compiled C projection path.
