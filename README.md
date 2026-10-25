# KKT-HardNet

[![CI](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![Validation](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/validation.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/validation.yml?query=branch%3Adevelopment)
[![CodeQL](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/codeql.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/codeql.yml?query=branch%3Adevelopment)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/kkt-hardnet/codex/coverage-development/coverage.svg)](https://github.com/SOULS-TAMU/kkt-hardnet/tree/codex/coverage-development)

A constrained neural network frontend that projects predictions onto nonlinear equality and inequality constraints.

## Installation

Install this development version with its tested shared dependencies:

```shell
python -m pip install "purin @ git+https://github.com/SOULS-TAMU/Purin.git@9ab708b470c4d9359e57f7bd214b0b5515aa4115"
python -m pip install "ohagi @ git+https://github.com/SOULS-TAMU/Ohagi.git@e28f687e840d6840b259e687323029c3cc49bd59"
python -m pip install "git+https://github.com/SOULS-TAMU/kkt-hardnet.git@development"
```

Python 3.10 or later is required. Availability on a particular Python version
and accelerator also depends on JAX and optional dependencies providing wheels
for that platform. Install accelerator-enabled JAX using the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Shared architecture

PURIN owns problem compilation, projection composition, sparse operators,
automatic differentiation, and numerical solvers. OHAGI owns backbone adapters,
the projected model, generic losses, gradient/update construction, batch and
epoch execution, checkpoint transport, and deployment runtime.

Each frontend owns its problem API, task loss, defaults, reporting, training
phase policy, and shared JAX serialization adapters. DAEHN also owns differential
processing, discretization, and reconstruction in `daehn.dae`. Frontends supply
epoch plans to `ohagi.training.Engine`; they do not implement separate numerical
training loops. The engine supports host batches and compiled JAX scans so
existing execution and precision policies remain intact.

OHAGI and PURIN never import a frontend. Framework adapters supply problem
definitions and pure prediction graphs. DAEHN and NLPOPTNet expose their current public interfaces directly, without
legacy implementation trees or alias catalogs.
Metadata inference loads through OHAGI without importing frontend code.

## Workflow

Keep the existing `KKTHardNet` and problem-builder API: define parameters,
variables and constraints, select the backbone and projection, provide the
dataset, and train. The distance projection uses an identity quadratic metric.
Custom JAX backbones remain supported through the existing callbacks. The
functional MLP is the default; Flax backbones and Orbax checkpoints are optional.
The existing resume interface restores optimizer state as well as weights.

See the [upstream modeling tutorial](https://kkt-hardnet.readthedocs.io/en/latest/)
for problem definitions and benchmark conventions. Historical documentation
may describe deployment backends that are not part of this stack.
## Define a constrained surrogate

```python
from kkthn import KKTHardNet, Projection, MLP, Loss

projection = Projection(method="chambolle-pock")
p = projection.add_parameter(["p"])
y = projection.add_variable(["u", "v"])
projection.constraints.add(y.u + y.v == p.p, y.u >= 0, y.v >= 0)
model = KKTHardNet(
    "surrogate", MLP(hidden_sizes=(64, 64)), Loss.mse(), projection,
    name="example", train={"epochs": 100, "batch_size": 32})
# Provide a CSV with columns p, u, v, then train:
model.dataset(parameters="data.csv", variables="data.csv")
result = model.model()
```

Use `backend="flax"` on `MLP` to select the optional Flax implementation.
Dataset preparation and training options retain their existing meanings.

## Export and repeated inference

Export a trained frontend model once:

```python
model.export_inference("deployment/metadata.json")
```

Load through the shared interface:

```python
from ohagi.inference import load

predictor = load("deployment/metadata.json")
y = predictor(x)
```

Supply only the metadata path to the loader; keep its referenced
`.jax_export` artifact with the bundle. Metadata alone does not contain the
weights or executable graph. Loading prepares and warms a JAX function for
repeated single-instance calls. Reuse the loaded predictor across calls;
loading and compilation are one-time costs. `predictor.predict_jax(x)` keeps
the result as a JAX array, while `predictor.predict(x)` returns synchronized
NumPy output. Exported inference does not require Flax, Orbax, or frontend
problem factories. The supported deployment backend is JAX.

## Packaging and development

This is a pure Python package with a `src` layout and a PEP 517 build defined
in `pyproject.toml`. Build with `python -m build`; no parallel `setup.py` or
custom wheel script is needed. Contributions should preserve the shared
engine and benchmark call signatures. See [CONTRIBUTING.md](CONTRIBUTING.md),
[CHANGELOG.md](CHANGELOG.md), [CITATION.cff](CITATION.cff), and [NOTICE](NOTICE).

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

Original authorship and applicable upstream notices are retained in [NOTICE](NOTICE).

## Package layout and model format

Each frontend uses the same source organization:

```text
src/<frontend>/
├── __init__.py
├── _aliases.py          # Retained only in KKTHN
├── api/                 # Problem definitions and task policies
└── serialization/       # Graph/definition adapters to OHAGI
```

DAEHN owns differential processing in `daehn.api.differential`; existing
`from daehn.dae import ...` imports remain supported. There are no physical
`_legacy`, `_compat`, `_api`, or top-level `adapters` implementation directories
in either DAEHN or NLPOPTNet.

All full-model saves and readers use OHAGI's `ohagi_model_v1` metadata schema and
`ohagi_pytree_npz_v1` numeric parameter codec. Deployment uses the same shared
single-instance JAX export/load interface for all three frontends:

```python
model.export_inference("deployment/metadata.json")
from ohagi.inference import load
predictor = load("deployment/metadata.json")
y = predictor(x)
```

Keep the referenced `.jax_export` graph alongside `metadata.json`. Deployment
bundles do not need frontend code, Flax, or Orbax. Full-model saves additionally
retain `weights.npz` and formulation data needed by the frontend's model reload
API; Orbax training checkpoints remain separate.

New saves do not produce `model.json`, duplicate backbone/model-weight archives,
standalone `.stablehlo` derivative files, or export-manifest sidecars. There are
no native/portable save-format switches. JAX uses StableHLO internally through
`jax.export`; no standalone StableHLO package or separate export backend is
required. JAX export's `flatbuffers` and `absl-py` dependencies remain required.

Model loaders accept only shared OHAGI JAX bundles. Legacy JSON/NPZ model
readers, format-specific plugins, and graph reconstruction fallbacks are removed. The two KKTHN
benchmark imports of `load_native_predictor` retain an import-only rejection
shim so the scripts stay unchanged; no native compiler or native writer exists.

## Compatibility and checks

Development CI tests Python 3.10–3.14 on Linux, Windows and macOS, plus Linux
ARM64 and a pinned JAX baseline. Accelerator support follows JAX's available
wheels; these checks exercise CPU execution. Coverage combines statements and
branches across the whole frontend. Optional Flax and Orbax integrations are
tested separately. See the [validation policy](.github/validation/README.md)
for numerical tolerances, current quality baselines and performance limitations.

### Quality gates
+
+CI requires at least **90% combined statement and branch coverage** over the
+entire frontend, cognitive complexity of at most **15** per function, and zero
+errors under the documented type-check policy. API snapshots, numerical
+references, exact checkpoint resume and isolated metadata inference protect
+compatibility. See the [validation policy](.github/validation/README.md).
+
