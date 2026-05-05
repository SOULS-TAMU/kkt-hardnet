Training
========

Training methods:

- ``model()``: supervised surrogate learning from parameter-variable pairs.
- ``optimize()``: unsupervised optimization using only parameter samples.
- ``estimate()``: inverse parameter estimation using observed variables.

Each run writes ``metadata.json``, ``summary.json``, ``history.csv``,
``predictions.csv``, ``model_weights.npz``, and native projection metadata.
