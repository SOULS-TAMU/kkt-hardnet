Defining Parameter Data
=======================

KKT-HardNet currently uses CSV datasets for parameter samples and, when
needed, supervised variable targets.

CSV-based Dataset
-----------------

Attach data with:

.. code-block:: python

   model.dataset(parameters="parameters.csv", variables="variables.csv")

The ``parameters.csv`` columns must match the names passed to
``add_parameter(...)``. The ``variables.csv`` columns must match the names
passed to ``add_variable(...)``.

Data Requirements by Workflow
-----------------------------

.. list-table::
   :header-rows: 1

   * - Workflow
     - Required CSV files
     - Description
   * - ``model()``
     - ``parameters.csv`` and ``variables.csv``
     - Supervised surrogate learning from known solutions.
   * - ``estimate()``
     - ``parameters.csv`` and ``variables.csv``
     - Inverse estimation using observed variables.
   * - ``optimize()``
     - ``parameters.csv``
     - Unsupervised optimization over supplied parameter samples.

Example
-------

.. code-block:: text

   parameters.csv

   x1,x2
   0.0,0.0
   0.5,-0.25
   -0.8,0.4

.. code-block:: text

   variables.csv

   y1,y2,y3
   0.2,0.1,-0.1
   0.4,0.0,-0.3
   -0.1,0.5,0.2

Then attach the files:

.. code-block:: python

   model.dataset(
       parameters="parameters.csv",
       variables="variables.csv",
   )

Notes
-----

- CSV headers are required.
- Header names should be unique.
- The row counts of ``parameters.csv`` and ``variables.csv`` must match when both files are provided.
- The model does not automatically reject infeasible parameter samples before training.
