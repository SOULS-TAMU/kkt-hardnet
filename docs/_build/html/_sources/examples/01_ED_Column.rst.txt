Extractive Distillation Column
==============================

This example trains a supervised KKT-HardNet model for an extractive
distillation column dataset. The parameters are feed or operating quantities,
and the variables represent column outputs constrained by material-balance and
composition equations.

Setup
-----

.. code-block:: python

   import os
   from pathlib import Path
   import time
   import numpy as np
   import pandas as pd
   import sys

   from kkthn import KKTHardNet

Configuration
-------------

.. code-block:: python

   DATA_PATH = "dataset/ED_Col_Data.csv"
   WORK_PATH = "dataset/ED_Col_Data_2000.csv"

   PARAMETERS = ["x1", "x2", "x3"]
   VARIABLES = ["y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9"]

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
       "newton_reg_factor": 1e-3,
       "max_newton_iter": 30,
       "max_backtrack_iter": 10,
       "eta": 1e-4,
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
       raise ValueError(f"Missing columns in CSV: {missing}")

   df_2000 = (
       df[required_cols]
       .dropna()
       .sample(n=2000, random_state=TRAIN["seed"])
       .reset_index(drop=True)
   )

   os.makedirs("dataset", exist_ok=True)
   df_2000.to_csv(WORK_PATH, index=False)

   parameters_csv = "dataset/parameters_2000.csv"
   variables_csv = "dataset/variables_2000.csv"

   df_2000[PARAMETERS].to_csv(parameters_csv, index=False)
   df_2000[VARIABLES].to_csv(variables_csv, index=False)

Build Model
-----------

.. code-block:: python

   model = KKTHardNet(name="ED_Column", train=TRAIN)

   x = model.add_parameter(PARAMETERS)
   y = model.add_variable(VARIABLES)

   model.constraints.add(
       x.x1 + x.x2 - y.y1 - y.y2 == 0,
       x.x1 * 0.697616946 - y.y1 * y.y3 - y.y2 * y.y6 == 0,
       x.x1 * 0.302383054 - y.y1 * y.y4 - y.y2 * y.y7 == 0,
       y.y3 + y.y4 + y.y5 - 1 == 0,
       y.y6 + y.y7 + y.y8 - 1 == 0,
       x.x3 * y.y1 - y.y9 == 0,
   )

   model.dataset(
       parameters=parameters_csv,
       variables=variables_csv,
   )

Train
-----

.. code-block:: python

   result = model.model()

   print("Training finished.")

   model.summary()
   model.plot_history(bg="white")

   pred_native = model.predict(
       df_2000[PARAMETERS].iloc[0].to_numpy(),
       projection_backend="native",
   )

Load and Use a Trained Model
----------------------------

.. code-block:: python

   run_dirs = [
       d for d in os.listdir(".")
       if os.path.isdir(d) and d.startswith("ED_Column_")
   ]

   latest_run_dir = max(run_dirs, key=os.path.getmtime)
   metadata_path = os.path.join(latest_run_dir, "metadata.json")

   loaded_model = KKTHardNet()
   loaded_model.load(metadata_path)

   loaded_model.predict([0.363557425, 1.312977767, 2.5])

The projection backend can be:

- ``"auto"``: use native projection if available, otherwise JAX.
- ``"jax"``: force the JAX projection path.
- ``"native"``: force the compiled native C projection path.

Example Summary
---------------

.. code-block:: python

   model.summary()

.. code-block:: text

   📊 KKT-HardNet Summary
   ------------------------------------------------------------
   Model Name                        : ED_Column_test
   No. of Parameters                 : 3
   No. of Variables                  : 9
   No. of Equalities                 : 6
   No. of Inequalities               : 0
   No. of Train Samples              : 1600
   No. of Validation Samples         : 400
   Maximum Constraint Violation      : 7.4773e-07
   Training Time                     : 197.92 s
   Est. JAX Single Inference Time    : 0.14 ms
   Est. JAX Batch Inference Time     : 1.85 ms
   ------------------------------------------------------------
   Note: Inference time estimations are based on
   microbenchmarking on the hardware used during
   training and may vary across different hardware
   and runtime conditions.

.. code-block:: python

   model.plot_history(bg="white")

.. image:: ../figures/training_history.png
   :align: center
   :width: 100%
