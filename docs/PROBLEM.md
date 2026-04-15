# Modeling Workflow

## Public Class

```python
from kkthn import KKTHardNet
```

## Define A Problem

```python
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


model = KKTHardNet(name="demo_model", train=TRAIN)
x = model.add_parameter(["x1", "x2"])
theta = model.add_inverse_parameter(["a0", "a1"], init_value=[10.0, -10.0])
y = model.add_variable(["y1", "y2", "y3"])

model.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)
model.constraints.add(
    theta.a0 * y.y1 + y.y2 - x.x1 == 0,
    y.y2 - theta.a1 * y.y3 - x.x2 == 0,
    y.y1**2 + y.y3**2 <= 2.0,
    y.y1 >= 0,
)
```

## Attach Data

Surrogate model or inverse estimation:

```python
model.dataset(parameters="parameters.csv", variables="variables.csv")
```

Optimization:

```python
model.dataset(parameters="parameters.csv")
```

The CSV column names must match the declared parameter and variable names.

## Train

Supervised surrogate:

```python
result = model.model()
```

Unsupervised optimization:

```python
result = model.optimize()
```

Inverse parameter estimation:

```python
result = model.estimate()
```

## Artifacts

Every run creates:

```text
<model_name>_<timestamp>/
```

with:

- `parameters.csv`
- `variables.csv` when labels are available
- `history.csv`
- `predictions.csv`
- `model_weights.npz`
- `summary.json`
- `metadata.json`

## Reload

```python
model = KKTHardNet()
model.load("demo_model_20260414_120000/metadata.json")
prediction = model.predict([1.0, 2.0])
```
