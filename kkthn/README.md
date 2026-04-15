# kkt-hardnet

`kkt-hardnet` is the publishable Python package for KKT-HardNet.

## Install

Editable install from this repository:

```bash
pip install -e kkthn
```

Editable install with CUDA 12 support:

```bash
pip install -e "kkthn[cuda12]"
```

Or use pip after publishing:

```bash
pip install kkt-hardnet
```

```bash
pip install "kkt-hardnet[cuda12]"
```

## Import

```python
from kkthn import KKTHardNet
```

## Core Methods

- `model()` for supervised surrogate learning from `parameters.csv` and `variables.csv`
- `optimize()` for unsupervised optimization from `parameters.csv`
- `estimate()` for inverse parameter estimation from `parameters.csv` and `variables.csv`
- `load(metadata_path)` to reload a trained run
- `predict(x)` to infer variables for new parameter values

## Packaging

The PyPI project name is `kkt-hardnet`, while the import path remains `kkthn`. The default install is CPU-oriented; CUDA support is selected explicitly with the `cuda12` extra instead of automatic device detection.

# Modeling Workflow

Use `KKTHardNet` to define symbolic constrained problems with:

- named parameters, decision variables, and optional inverse parameters
- equality and inequality constraints written as Python expressions
- optional objectives for optimization tasks
- hard KKT projection during training and inference
- saved run artifacts including weights, predictions, history, summary, and metadata

Available methods:

- `dataset(parameters=..., variables=...)` to attach training data
- `model()` for supervised surrogate learning
- `optimize()` for unsupervised optimization
- `estimate()` for inverse parameter estimation
- `load(metadata_path)` to reload a trained run
- `predict(x)` to evaluate a trained or loaded model on new inputs

The example below shows one complete workflow. The CSV column names must match the declared parameter and variable names. Each run writes `<model_name>_<timestamp>/` with `parameters.csv`, optional `variables.csv`, `history.csv`, `predictions.csv`, `model_weights.npz`, `summary.json`, and `metadata.json`.

```python
from kkthn import KKTHardNet

TRAIN = {
    "epochs": 1200,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "train_frac": 0.8,
    "hidden_size": 64,
    "hidden_layers": 2,
    "seed": 42,
    "dtype": "float64",
    "print_every": 1,
    "newton_step_length": 0.5,
    "newton_tol": 1e-6,
    "newton_reg_factor": 1e-3,
    "max_newton_iter": 30,
    "max_backtrack_iter": 10,
}

# Build a symbolic problem.
model = KKTHardNet(name="demo_model", train=TRAIN)
x = model.add_parameter(["x1", "x2"])
theta = model.add_inverse_parameter(["a0", "a1"], init_value=[10.0, -10.0])
y = model.add_variable(["y1", "y2", "y3"])

# Objectives are optional for surrogate modeling and inverse estimation,
# but required for optimize().
model.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)
model.constraints.add(
    theta.a0 * y.y1 + y.y2 - x.x1 == 0,
    y.y2 - theta.a1 * y.y3 - x.x2 == 0,
    y.y1**2 + y.y3**2 <= 2.0,
    y.y1 >= 0,
)

# Attach data.
# For model() or estimate(): provide both parameters and variables.
# For optimize(): provide only parameters.
model.dataset(parameters="parameters.csv", variables="variables.csv")

# Choose one training mode.
surrogate_result = model.model()
# optimize_result = model.optimize()
# estimate_result = model.estimate()

# Reload a saved run later and predict on a new parameter vector.
reloaded = KKTHardNet()
reloaded.load("demo_model_20260414_120000/metadata.json")
prediction = reloaded.predict([1.0, 2.0])
```
⚠️ Please cite our work if you use this code in your research.
Citation formats are provided below.

arXiv Preprint: https://arxiv.org/pdf/2507.08124
Journal: https://doi.org/10.1016/j.compchemeng.2025.109418

```bibtex
@article{iftakher2025physics,
  title={Physics-informed neural networks with hard nonlinear equality and inequality constraints},
  author={Iftakher, Ashfaq and Golder, Rahul and Nath Roy, Bimol and Hasan, MM Faruque},
  journal={Computers \& Chemical Engineering},
  pages={109418},
  year={2025},
  publisher={Elsevier}
}
