Hyperparameters
===============

``KKTHardNet`` accepts a training dictionary that is normalized into
:class:`kkthn.training.KKTTrainConfig`.

Common keys
-----------

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``epochs``
     - Number of training epochs.
   * - ``batch_size``
     - Training batch size.
   * - ``learning_rate``
     - Adam learning rate.
   * - ``hidden_size``
     - Width of hidden MLP layers.
   * - ``hidden_layers``
     - Number of hidden MLP layers.
   * - ``train_frac``
     - Fraction of data used for training.
   * - ``seed``
     - Random seed for splitting and initialization.
   * - ``dtype``
     - Numeric precision, typically ``float64``.
   * - ``print_every``
     - Console logging frequency.
   * - ``eta``
     - Optional loss threshold for starting projection training.
   * - ``epoch_mlp``
     - Optional epoch for starting projection training.
   * - ``cons_alpha``
     - Consistency-loss weight.

Projection keys
---------------

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``fb_eps``
     - Fischer-Burmeister smoothing value.
   * - ``gn_max_iters``
     - Maximum projection solver iterations.
   * - ``gn_tol``
     - Projection residual tolerance.
   * - ``gn_reg``
     - Projection solve regularization.
   * - ``newton_step_length``
     - Initial step length for line search.
   * - ``armijo_alpha``
     - Armijo sufficient-decrease coefficient.
   * - ``armijo_beta``
     - Backtracking contraction coefficient.
   * - ``max_backtrack_iter``
     - Maximum backtracking steps.
   * - ``backward_reg``
     - Regularization in the implicit backward solve.

Core code reference
-------------------

.. autoclass:: kkthn.training.KKTTrainConfig
   :members:
   :member-order: bysource
   :no-index:

.. autoclass:: kkthn.projection.ProjectionSettings
   :members:
   :member-order: bysource
   :no-index:
