Pooling Problem
===============

This example trains KKT-HardNet on a pooling problem with nonlinear blending
constraints and quality limits.

Setup
-----

.. code-block:: python

   import os
   import pandas as pd
   from pathlib import Path
   import sys

   ROOT = next(
       path for path in [Path.cwd(), *Path.cwd().parents]
       if (path / "kkthn").is_dir() and (path / "notebooks").is_dir()
   )
   SRC = ROOT / "kkthn" / "src"
   if str(SRC) not in sys.path:
       sys.path.insert(0, str(SRC))

   from kkthn import KKTHardNet

Configuration
-------------

.. code-block:: python

   DATA_PATH = "dataset/Pooling_dataset.csv"

   PARAMETERS = ["x1", "x2", "x3", "x4"]
   VARIABLES = ["y1", "y2", "y3", "y4", "y5"]

   TRAIN = {
       "epochs": 1200,
       "batch_size": 40,
       "learning_rate": 1e-3,
       "train_frac": 0.8,
       "hidden_size": 64,
       "hidden_layers": 2,
       "seed": 42,
       "dtype": "float64",
       "print_every": 100,
       "newton_step_length": 0.5,
       "newton_tol": 1e-6,
       "newton_reg_factor": 1e-2,
       "max_newton_iter": 100,
       "max_backtrack_iter": 10,
       "eta": 75,
       "epoch_mlp": 100,
       "cons_alpha": 10,
   }

Prepare Data
------------

.. code-block:: python

   df = pd.read_csv(DATA_PATH)

   required_cols = PARAMETERS + VARIABLES
   missing = [c for c in required_cols if c not in df.columns]
   if missing:
       raise ValueError(f"Missing columns: {missing}")

   df_2000 = (
       df[required_cols]
       .dropna()
       .sample(n=2000, random_state=TRAIN["seed"])
       .reset_index(drop=True)
   )

   os.makedirs("dataset", exist_ok=True)

   param_path = "dataset/pooling_parameters_2000.csv"
   var_path = "dataset/pooling_variables_2000.csv"

   df_2000[PARAMETERS].to_csv(param_path, index=False)
   df_2000[VARIABLES].to_csv(var_path, index=False)

Build Model
-----------

.. code-block:: python

   model = KKTHardNet(name="Pooling", train=TRAIN)

   x = model.add_parameter(PARAMETERS)
   y = model.add_variable(VARIABLES)

   model.constraints.add(
       y.y2 + y.y3 - x.x1 - x.x2 == 0,
       x.x3 - y.y2 - y.y4 == 0,
       x.x4 - y.y3 - y.y5 == 0,
       y.y1 * y.y2 + y.y1 * y.y3 - 3 * x.x1 - x.x2 == 0,
       y.y1 * y.y2 + 2 * y.y4 - 2.5 * x.x3 <= 0,
       y.y1 * y.y3 + 2 * y.y5 - 1.5 * x.x4 <= 0,
   )

   model.dataset(
       parameters=param_path,
       variables=var_path,
   )

Train
-----

.. code-block:: python

   result = model.model()

Load and Use a Trained Model
----------------------------

.. code-block:: python

   run_dirs = [
       d for d in os.listdir(".")
       if os.path.isdir(d) and d.startswith("Pooling_20260505_105644")
   ]

   latest_run_dir = max(run_dirs, key=os.path.getmtime)
   metadata_path = os.path.join(latest_run_dir, "metadata.json")

   loaded_model = KKTHardNet()
   loaded_model.load(metadata_path)

   PARAMETERS = ["x1", "x2", "x3", "x4"]
   VARIABLES = ["y1", "y2", "y3", "y4", "y5"]

   single_x = df_2000[PARAMETERS].iloc[0].tolist()
   batch_x = df_2000[PARAMETERS].iloc[:50].values.tolist()

   loaded_model.predict(
       [41.79668639504713, 205.6995824265593, 95.92034507046218, 156.35598700458974]
   )
   loaded_model.predict(batch_x, projection_backend="jax")

Example Summary
---------------

.. code-block:: python

   model.summary()

.. code-block:: text

   📊 KKT-HardNet Summary
   ------------------------------------------------------------
   Model Name                        : Pooling
   No. of Parameters                 : 4
   No. of Variables                  : 5
   No. of Equalities                 : 4
   No. of Inequalities               : 2
   No. of Train Samples              : 1600
   No. of Validation Samples         : 400
   Maximum Constraint Violation      : 0.0019
   Training Time                     : 671.11 s
   Est. JAX Single Inference Time    : 0.17 ms
   Est. JAX Batch Inference Time     : 2.44 ms
   ------------------------------------------------------------
   Note: Inference time estimations are based on
   microbenchmarking on the hardware used during
   training and may vary across different hardware
   and runtime conditions.
