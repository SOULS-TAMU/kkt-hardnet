# Problem definitions and training

## Workflow

Declare parameters and variables, add constraints, select a backbone/projection
and loss, attach observations, then train. The distance projection uses an
identity quadratic metric. Constraint satisfaction depends on solver convergence
and tolerances.

## Explicit backbone and projection

```python
from kkthn import KKTHardNet, Projection, MLP, Loss

projection = Projection(method="chambolle-pock")
p = projection.add_parameter(["p"])
y = projection.add_variable(["u", "v"])
projection.constraints.add(y.u + y.v == p.p, y.u >= 0, y.v >= 0)
model = KKTHardNet(
    "surrogate", MLP(hidden_sizes=(32, 32)), Loss.mse(), projection,
    name="example", train={"epochs": 100, "batch_size": 32},
)
```

Use `MLP(hidden_sizes=(32, 32), backend="flax")` for the optional Flax adapter.
Custom JAX backbones retain the existing initialization/application interface.

## Unsupervised optimization

```python
model = KKTHardNet(name="allocation", train={"epochs": 100})
p = model.add_parameter(["p"])
y = model.add_variable(["u", "v"])
model.objective = 0.5 * (y.u**2 + y.v**2)
model.constraints.add(y.u + y.v == p.p, y.u >= 0, y.v >= 0)
model.dataset(parameters="data.csv")  # Header includes p.
result = model.optimize()
```

## Inverse parameter estimation

```python
model = KKTHardNet(name="inverse", train={"epochs": 100})
a = model.add_inverse_parameter(["a"], init_value=[1.0])
p = model.add_parameter(["p"])
y = model.add_variable(["u"])
model.constraints.add(y.u == a.a * p.p)
model.dataset(parameters="observations.csv", variables="observations.csv")
result = model.estimate()
```

The observations CSV contains `p,u`. Choose observations that identify the
unknown coefficients; constraints alone do not guarantee identifiability.

## Checkpoint and resume

After training, `model.save_checkpoint("checkpoints/latest")` saves optional
Orbax state. Recreate the same problem, backbone, data and training settings,
then call `resumed.resume_training(checkpoint, epochs=total_epochs)`.
Here `epochs` is the total training horizon, not an additional epoch count.
Keep the seed, batching and phase settings consistent for reproducibility.

Training returns history and paths including `result["metadata_path"]`. Keep
referenced weights and formulation files when moving a full-model bundle.

[Back to README](../README.md)
