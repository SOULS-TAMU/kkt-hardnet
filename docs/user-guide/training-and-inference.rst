Training and Inference
======================

After defining the symbolic problem and attaching data, choose one of the
training workflows.

Build and Train
---------------

.. code-block:: python

   result = model.model()      # supervised surrogate learning
   # result = model.optimize() # unsupervised optimization
   # result = model.estimate() # inverse parameter estimation

   run_dir = result["output_dir"]
   print("run_dir =", run_dir)

Each run writes a timestamped directory:

``<model_name>_<YYYYMMDD>_<HHMMSS>/``

Model Summary and Training History
----------------------------------

``summary()`` prints the model dimensions, dataset sizes, constraint violation,
training time, and estimated inference times.

.. code-block:: python

   model.summary()

.. note::

   Inference time estimations are based on microbenchmarking on the training
   hardware and may vary across hardware and runtime conditions.

``plot_history()`` saves and displays training curves.

.. code-block:: python

   model.plot_history(bg="white")

.. image:: ../figures/training_history.png
   :align: center
   :width: 100%

Saved Artifacts
---------------

The following files are generated inside the run directory:

- ``metadata.json`` — Complete problem definition and artifact references used for reload.
- ``summary.json`` — Final metrics, timing estimates, model dimensions, and native projection status.
- ``history.csv`` — Per-epoch training and validation metrics.
- ``parameters.csv`` — Parameter samples used for training and validation.
- ``variables.csv`` — Optional supervised target variables.
- ``predictions.csv`` — Raw and projected model predictions.
- ``model_weights.npz`` — Trained MLP weights and inverse parameters when present.
- ``projection_native.json`` — Manifest for the native C projection backend.
- ``native_projection/<source_version>/<platform>/projection_native.so`` — Platform-specific native library on Linux, with corresponding ``.dll`` or ``.dylib`` names on Windows/macOS.

Reloading the Model
-------------------

.. code-block:: python

   from kkthn import KKTHardNet

   reloaded = KKTHardNet().load("path/to/metadata.json")

   sample_pred_native = reloaded.predict(sample_x, projection_backend="native")
   sample_pred_jax = reloaded.predict(sample_x, projection_backend="jax")

Projection Backends
-------------------

- ``auto`` — Uses the native backend when a compatible artifact is available, otherwise falls back to JAX.
- ``jax`` — Uses the JAX implementation of the projection layer.
- ``native`` — Uses the compiled C projection layer and raises an error if no native artifact can be loaded.

The native backend compiles a platform-specific shared library into the run
folder. If the model is reloaded on a different supported OS/architecture,
KKT-HardNet compiles a matching native binary for the current system into the
same run-local ``native_projection/`` directory.

.. note::

   To use the native backend, a C compiler such as ``cc``, ``gcc``, or ``clang``
   must be installed.
