# KKTHardNet Version Tracker

## 0.1.0 - 2026-04-11

Initial packaged KKTHardNet codespace.

- Added `kkthn` as a `src`-layout installable package under `KKTHardNet/kkthn`.
- Added KKT-HardNet training, backbone, projection, problem, and string-problem modules.
- Added runner support for QP, QCQP, NLP, nonconvex, model-definition general, and string-defined general cases.
- Added CPU and GPU requirement files.
- Added per-epoch and per-batch timing plus component profiling for backbone, projection, backprop, and optimizer update.
- Added automatic broadcasting for single-entry `x_L`/`x_U` bounds to the parameter dimension.
- Added package metadata, README, LICENSE, and build placeholder.
