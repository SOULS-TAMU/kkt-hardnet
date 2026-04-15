# KKT-HardNet

KKT-HardNet is a JAX-based framework for constrained surrogate modeling, inverse parameter estimation, and unsupervised optimization with a hard KKT projection layer.

## Install

Stable PIP Install:

```bash
pip install kkt-hardnet
```

For users who want CUDA 12 support:

```bash
pip install "kkt-hardnet[cuda12]"
```


For local install:

```bash
git clone https://github.com/SOULS-TAMU/kkt-hardnet.git
cd kkt-hardnet
python -m venv env
source env/bin/activate
python -m pip install -e kkthn
```

For local install with CUDA 12 dependencies:

```bash
git clone https://github.com/SOULS-TAMU/kkt-hardnet.git
cd kkt-hardnet
python -m venv env
source env/bin/activate
python -m pip install -e "kkthn[cuda12]"
```

The published package name is `kkt-hardnet`. The import name stays:

```python
from kkthn import KKTHardNet
```

## Python Versions

The package keeps Python-version-specific dependency pins through environment markers. CPU is the default install, and GPU support is enabled only when the user requests the `cuda12` extra.

## Basic API

For usage use these parameters.csv and variables.csv files

### Surrogate Model (Extractive Distillation Column):

```python
from kkthn import KKTHardNet

TRAIN = {
    "epochs": 1000,
    "batch_size": 20,
    "learning_rate": 1e-3,
    "train_frac": 0.8,
    "hidden_size": 32,
    "hidden_layers": 2,
    "seed": 42,
    "dtype": "float64",
    "print_every": 100,
    "newton_step_length": 0.5,
    "newton_tol": 1e-6,
    "newton_reg_factor": 1e-3,
    "max_newton_iter": 30,
    "max_backtrack_iter": 10,
}

model = KKTHardNet(name='ED_Column', train=TRAIN)
x = model.add_parameter(['x1', 'x2', 'x3'])
y = model.add_variable(['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9'])
model.constraints.add(
    x.x1 + x.x2 - y.y1 - y.y2 == 0,
    x.x1*0.697616946 - y.y1*y.y3 - y.y2*y.y6 == 0,
    x.x1*0.302383054 - y.y1*y.y4 - y.y2*y.y7 == 0,
    y.y3 + y.y4 + y.y5 - 1 == 0,
    y.y6 + y.y7 + y.y8 - 1 == 0,
    x.x3*y.y1 - y.y9 == 0
)
model.dataset(parameters='notebooks/ED_Column/parameters.csv', variables='notebooks/ED_Column/variables.csv')

# Surrogate Model
result = model.model()

```

### Unsupervised Optimization (Toy Optimization Problem):

```python
from kkthn import KKTHardNet

# Set the training configuration here
# TRAIN = {}

model = KKTHardNet(name='general_optimize', train=TRAIN)
x = model.add_parameter(['x1', 'x2'])
y = model.add_variable(['y1', 'y2', 'y3'])
model.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)
model.constraints.add(
    y.y1 + y.y2 - x.x1 == 0,
    y.y2 - y.y3 - x.x2 == 0,
    y.y1**2 + y.y3**2 <= 2.0,
    y.y1 >= 0,
)
model.dataset(parameters='notebooks/optimization/parameters.csv')

# Unsupervised Optimization
result = model.optimize()

```
### Inverse Model (Extractive Distillation Column):

```python
from kkthn import KKTHardNet

# Set the training configuration here
# TRAIN = {}

model = KKTHardNet(name='ED_Column', train=TRAIN)
theta = model.add_inverse_parameter(["a0", "a1"], init_value=[1.0, 1.0])
x = model.add_parameter(['x1', 'x2', 'x3'])
y = model.add_variable(['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9'])
model.constraints.add(
    x.x1 + x.x2 - y.y1 - y.y2 == 0,
    x.x1*theta.a0 - y.y1*y.y3 - y.y2*y.y6 == 0,
    x.x1*theta.a1 - y.y1*y.y4 - y.y2*y.y7 == 0,
    y.y3 + y.y4 + y.y5 - 1 == 0,
    y.y6 + y.y7 + y.y8 - 1 == 0,
    x.x3*y.y1 - y.y9 == 0
)
model.dataset(parameters='notebooks/ED_Column/parameters.csv', variables='notebooks/ED_Column/variables.csv')

# Inverse Model
result = model.estimate()

```

Each run creates a folder in the working directory named:

```text
<model_name>_<timestamp>
```

That folder contains `parameters.csv`, optional `variables.csv`, `history.csv`, `predictions.csv`, `model_weights.npz`, `summary.json`, and `metadata.json`.

## Load And Predict

```python
model = KKTHardNet()
model.load("<model_name>_<timestamp>/metadata.json")
predicted = model.predict([0.390345867,1.378626656,3.1])
```

If `predict()` is called before training or loading, the package raises:

```text
Please train or load the model before calling predict().
```

## Data Convention

- Surrogate model: `parameters.csv` and `variables.csv`
- Inverse estimation: `parameters.csv` and `variables.csv`
- Optimization: `parameters.csv` only

The CSV headers must match the parameter and variable names declared in the model.

## Examples And Docs

- `notebooks/`: example workflows for QP, QCQP, NLP, nonconvex, and general cases
- `docs/INSTALL.md`: install notes
- `docs/PROBLEM.md`: modeling workflow
- `docs/PUBLISH.md`: build and publish instructions
- `docs/VERSION.md`: version notes
