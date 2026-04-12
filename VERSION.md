# KKTHardNet Version Tracker

## 0.1.4 - 2026-04-12

Standalone utility bundle.

- Bundled the required `jaxmodel`, `solgen`, `scripts.factory`, and
  `scripts.misc` utilities inside KKTHardNet.
- Removed sibling NLPOptNet path discovery from `main.py`, `run_general.py`,
  `kkthn.problems`, `kkthn.string_problem`, and the general model definition.
- Mirrored utility modules under both root `scripts` and `kkthn/src/scripts` so
  the codespace and editable package install are self-contained.
- Updated documentation to describe KKTHardNet as an independent codespace.

## 0.1.3 - 2026-04-12

Sibling relocation support.

- Added NLPOptNet discovery for sibling layouts such as `D:/Projects/KKTHardNet` next to `D:/Projects/NLPOptNet`.
- Added `NLP_OPT_NET_ROOT` / `NLPOPTNET_ROOT` environment-variable override for dependency discovery.
- Updated runner scripts, package helpers, and the general case model definition to use the same discovery behavior.

## 0.1.2 - 2026-04-12

Native Windows install fallback.

- Changed default dependency detection to use CPU requirements on native Windows.
- Avoids `jax[cuda12]` install failures from unavailable `jax-cuda12-plugin` Windows wheels.
- Added install-time message with selected CPU/GPU dependency mode and reason.
- Added `install_info.py` for explicitly printing the selected dependency mode before running pip.
- Preserved `KKTHN_REQUIREMENTS=gpu` override for explicit advanced use.

## 0.1.1 - 2026-04-12

Packaging install dependency selection.

- Added dynamic editable-install dependency selection for `pip install -e kkthn`.
- Uses `kkthn/requirements.gpu.txt` when CUDA is detected via `CUDA_HOME`, `CUDA_PATH`, `nvidia-smi`, or `nvcc`.
- Uses `kkthn/requirements.txt` otherwise.
- Added `KKTHN_REQUIREMENTS=cpu|gpu` override.
- Mirrored CPU/GPU requirement files inside the package project for install-time metadata generation.

## 0.1.0 - 2026-04-11

Initial packaged KKTHardNet codespace.

- Added `kkthn` as a `src`-layout installable package under `KKTHardNet/kkthn`.
- Added KKT-HardNet training, backbone, projection, problem, and string-problem modules.
- Added runner support for QP, QCQP, NLP, nonconvex, model-definition general, and string-defined general cases.
- Added CPU and GPU requirement files.
- Added per-epoch and per-batch timing plus component profiling for backbone, projection, backprop, and optimizer update.
- Added automatic broadcasting for single-entry `x_L`/`x_U` bounds to the parameter dimension.
- Added package metadata, README, LICENSE, and build placeholder.
