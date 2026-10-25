# KKT-HardNet

[![CI](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![Validation](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/validation.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/validation.yml?query=branch%3Adevelopment)
[![CodeQL](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/codeql.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/codeql.yml?query=branch%3Adevelopment)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/kkt-hardnet/codex/coverage-development/coverage.svg)](https://github.com/SOULS-TAMU/kkt-hardnet/tree/codex/coverage-development)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/SOULS-TAMU/kkt-hardnet/blob/development/LICENSE)
[![Tested Python: 3.10–3.14](https://img.shields.io/badge/Python-3.10%E2%80%933.14-blue?logo=python&logoColor=white)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![CPU platforms: Linux, Windows, macOS](https://img.shields.io/badge/CPU_platforms-Linux%20%7C%20Windows%20%7C%20macOS-blue)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![Dependency review](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/dependency-review.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/kkt-hardnet/actions/workflows/dependency-review.yml?query=branch%3Adevelopment)

A JAX framework for constrained surrogate modeling, inverse parameter
estimation and unsupervised optimization with differentiable hard projections.

The frontend defines the problem and task policy.
[OHAGI](https://github.com/SOULS-TAMU/Ohagi) provides the shared learning and
inference engine; [PURIN](https://github.com/SOULS-TAMU/Purin) provides projection
algorithms and sparse kernels. Existing benchmark APIs remain supported.

## Install

Install the development version with its tested shared dependencies:

```shell
python -m pip install "purin @ git+https://github.com/SOULS-TAMU/Purin.git@9ab708b470c4d9359e57f7bd214b0b5515aa4115"
python -m pip install "ohagi @ git+https://github.com/SOULS-TAMU/Ohagi.git@e28f687e840d6840b259e687323029c3cc49bd59"
python -m pip install "git+https://github.com/SOULS-TAMU/kkt-hardnet.git@development"
```

See [installation options](docs/INSTALL.md) for editable installs and optional
Flax backbones/Orbax checkpoints.

## Python versions and platforms

Python **3.10–3.14** is tested on **Linux, Windows and macOS**, with additional
Linux ARM64 and pinned JAX baseline checks. Package installation and optional
Flax/Orbax integrations are tested across all three OSes. These are CPU checks;
accelerator support follows the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Basic API

### Constrained surrogate

```python
from pathlib import Path
import numpy as np
from kkthn import KKTHardNet

p_values = np.linspace(0.4, 1.2, 8)
data = Path("data.csv")
np.savetxt(data, np.column_stack((p_values, 0.6*p_values, 0.4*p_values)),
           delimiter=",", header="p,u,v", comments="")

model = KKTHardNet(name="sum_model", train={
    "epochs": 2, "batch_size": 2, "seed": 7,
    "pretrain": False, "print_every": 0,
})
p = model.add_parameter(["p"])
y = model.add_variable(["u", "v"])
model.constraints.add(y.u + y.v == p.p, y.u >= 0, y.v >= 0)
model.dataset(parameters=data, variables=data)
result = model.model()
prediction = model.predict([0.7])
```

### Optimization and inverse estimation

| Task | Required resources | Entry point |
|---|---|---|
| Surrogate | Parameters and observed variables | `model.model()` |
| Unsupervised optimization | Parameters and symbolic objective | `model.optimize()` |
| Inverse estimation | Parameters, observations and unknown coefficients | `model.estimate()` |

See [problem definitions](docs/PROBLEM.md) for examples and backbone options.

## Load and predict

Reload the full model returned by training:

```python
model = KKTHardNet().load(result["metadata_path"])
prediction = model.predict([0.7])
```

For repeated single-instance deployment, export once:

```python
model.export_inference("deployment/metadata.json")
```

The shared loader is available to both frameworks:

```python
from ohagi.inference import load

predictor = load("deployment/metadata.json")
prediction = predictor.predict([0.7])
```

Pass the metadata path and keep its referenced `.jax_export` file with the
bundle. Load once and reuse the predictor. `predict` returns synchronized NumPy
output; `predict_jax` returns a JAX array. See [inference](docs/INFERENCE.md).

## Data convention

CSV headers must match declared parameter and variable names. Surrogate and
inverse tasks use parameter and variable observations; optimization uses only
parameter observations. One CSV may contain both sets of columns.

## Examples and documentation

- [Install](docs/INSTALL.md): dependencies, optional integrations and packaging.
- [Problem definitions](docs/PROBLEM.md): modeling, backbones and training.
- [Inference](docs/INFERENCE.md): export, model bundles and repeated prediction.
- [Version notes](docs/VERSION.md) and [changelog](CHANGELOG.md): development changes.
- [Contributing](CONTRIBUTING.md) and [validation policy](.github/validation/README.md): development checks.

CI requires **90% combined statement and branch coverage**, cognitive complexity
at most **15** per function and the documented type-check policy. API snapshots,
numerical regression and exact checkpoint-resume checks protect compatibility.

The [historical tutorials](https://kkt-hardnet.readthedocs.io/en/latest/) provide
additional modeling examples. Use this branch's guides for current installation
and JAX deployment.

## Citation and license

Use [CITATION.bib](CITATION.bib) for BibTeX or GitHub's **Cite this repository**
menu. Licensed under [Apache 2.0](LICENSE); attribution is retained in [NOTICE](NOTICE).
