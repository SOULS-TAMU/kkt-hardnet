Configuration Options
=====================

KKT-HardNet accepts training options either when constructing
``KKTHardNet(name=..., train=...)`` or when calling ``model()``,
``optimize()``, or ``estimate()``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Default Value
     - Required
   * - ``epochs``
     - Number of training epochs.
     - 1200
     - No
   * - ``batch_size``
     - Number of samples per training batch.
     - 32
     - No
   * - ``learning_rate``
     - Learning rate for Adam.
     - 1e-3
     - No
   * - ``hidden_size``
     - Width of hidden MLP layers.
     - 64
     - No
   * - ``hidden_layers``
     - Number of hidden MLP layers.
     - 2
     - No
   * - ``train_frac``
     - Fraction of samples used for training.
     - 0.8
     - No
   * - ``seed``
     - Random seed for splitting and initialization.
     - 42
     - No
   * - ``dtype``
     - Numerical precision, usually ``"float64"``.
     - ``"float64"``
     - No
   * - ``print_every``
     - Frequency of printed epoch logs.
     - 1
     - No
   * - ``drop_last``
     - Whether to drop incomplete mini-batches.
     - False
     - No
   * - ``eta``
     - Optional loss threshold for switching from MLP-only to projection training.
     - None
     - No
   * - ``epoch_mlp``
     - Optional epoch at which projection training begins.
     - None
     - No
   * - ``cons_alpha``
     - Weight for the consistency loss between raw and projected predictions.
     - 0.0
     - No

Projection settings can be supplied with the ``projection`` argument or nested
training configuration.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Default Value
   * - ``fb_eps``
     - Fischer-Burmeister smoothing value.
     - 1e-8
   * - ``gn_max_iters``
     - Maximum Gauss-Newton iterations in the projection solve.
     - 30
   * - ``gn_tol``
     - Projection residual tolerance.
     - 1e-6
   * - ``gn_reg``
     - Regularization in the normal-equation solve.
     - 1e-3
   * - ``newton_step_length``
     - Initial line-search step length.
     - 0.5
   * - ``armijo_alpha``
     - Armijo sufficient decrease parameter.
     - 1e-4
   * - ``armijo_beta``
     - Backtracking contraction factor.
     - 0.5
   * - ``max_backtrack_iter``
     - Maximum line-search backtracking steps.
     - 10
   * - ``backward_reg``
     - Regularization used by the custom VJP backward solve.
     - 1e-8

Example
-------

.. code-block:: python

   from kkthn import KKTHardNet

   TRAIN = {
       "epochs": 1000,
       "batch_size": 32,
       "learning_rate": 1e-3,
       "train_frac": 0.8,
       "hidden_size": 64,
       "hidden_layers": 2,
       "seed": 42,
       "dtype": "float64",
       "print_every": 50,
       "cons_alpha": 1.0,
   }

   PROJECTION = {
       "gn_max_iters": 30,
       "gn_tol": 1e-6,
       "gn_reg": 1e-3,
       "newton_step_length": 0.5,
   }

   model = KKTHardNet(name="Example_Model", train=TRAIN)

The output directory is named:

``<model_name>_<YYYYMMDD>_<HHMMSS>``

For example:

- ``Example_Model_20260505_104512``
- ``kkthardnet_20260505_104512``
