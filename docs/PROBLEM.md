# Defining General Problems

KKTHardNet supports two styles for custom general problems:

- Symbolic builder problems in `run_general.py`.
- Block-structured JAX problems in `case/general/model_definition.py`.

Use the symbolic builder for small and readable systems. Use the block style
when the problem has many repeated constraints or matrix/tensor structure.

## Symbolic Builder Problem

`run_general.py` is the editable entry point for symbolic general problems.
The problem is defined in `build_problem()`:

```python
from kkthn import ProblemBuilder


def build_problem() -> ProblemBuilder:
    builder = ProblemBuilder(y_bound=4.0)

    x = builder.add_parameter(["x1", "x2"])
    theta = builder.add_inverse_parameter(DATA["inv_param"])
    y = builder.add_variable(["y1", "y2", "y3"])

    builder.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)

    builder.constraints.add(
        theta.a0 * y.y1 + y.y2 - x.x1 == 0,
        y.y2 - theta.a1 * y.y3 - x.x2 == 0,
        y.y1**2 + y.y3**2 <= 2.0,
    )

    builder.bounds.set(lower=-4.0, upper=4.0)
    return builder
```

The builder compiles this symbolic problem into the internal
`HighLevelNLPBuilder` model. Constraints are grouped into equality and
inequality blocks before JAX tracing, so preprocessing is faster than parsing
strings repeatedly.

### Forward Mode

Forward mode is the default:

```sh
python run_general.py --mode forward
```

If the symbolic problem uses inverse parameters, forward mode treats them as
fixed constants from `DATA["inv_param_label"]`. No inverse parameters are
estimated in this mode.

Example data block:

```python
DATA = {
    "num_samples": 100,
    "seed": 42,
    "noise_scale": 0.0,
    "x_L": [-1.0, -1.0],
    "x_U": [1.0, 1.0],
    "inv_param": ["a0", "a1"],
    "inv_param_label": [1.0, -1.0],
    "inv_param_init": [0.0, 0.0],
}
```

`noise_scale` controls Gaussian noise added to generated labels after the clean
optimizer solution is computed. A value of `0.1` adds
`0.1 * N(0, 1)` noise to every generated `Y` entry. Leave it as `0.0` for clean
labels.

### Inverse Mode

Inverse mode estimates the entries declared with `add_inverse_parameter(...)`:

```sh
python run_general.py --mode inverse
```

The labels are generated with `DATA["inv_param_label"]`. Training starts from
`DATA["inv_param_init"]`, or zeros if the init list is empty.

At the end of training, KKTHardNet prints and saves:

```text
inverse_comparison.json
```

That file contains the actual values, estimated values, signed errors, and
absolute errors.

### Symbolic Builder API

Create symbols:

```python
x = builder.add_parameter(["x1", "x2"])
theta = builder.add_inverse_parameter(["a0", "a1"])
y = builder.add_variable(["y1", "y2", "y3"])
```

Set the objective:

```python
builder.objective = 0.5 * (y.y1**2 + y.y2**2)
```

Add constraints:

```python
builder.constraints.add(
    y.y1 + y.y2 - x.x1 == 0,
    theta.a0 * y.y1 + y.y2 <= 1.0,
    y.y3 >= -2.0,
)
```

Supported comparison operators:

- `==` creates equality residuals.
- `<=` creates inequality residuals of the form `left - right <= 0`.
- `>=` creates inequality residuals of the form `right - left <= 0`.

Set variable bounds:

```python
builder.bounds.set(lower=-4.0, upper=4.0)
```

Bounds may be scalars or vectors with one entry per decision variable:

```python
builder.bounds.set(
    lower=[-4.0, -2.0, 0.0],
    upper=[4.0, 2.0, 5.0],
)
```

Available nonlinear helpers:

```python
builder.sin(expr)
builder.cos(expr)
builder.exp(expr)
builder.log(expr)
builder.sqrt(expr)
builder.abs(expr)
```

Example:

```python
builder.objective = 0.5 * y.y1**2 + builder.sin(y.y2)
builder.constraints.add(builder.exp(y.y1) + y.y2 <= theta.a0 + x.x1)
```

## Block-Structured Problem

Use `case/general/model_definition.py` when the problem has matrix, tensor, or
repeated block structure. This route builds a `JaxNLPModel` directly with
`HighLevelNLPBuilder`.

The required entry point is:

```python
def build_model(*, dtype=jnp.float64):
    ...
    return model
```

Minimal block skeleton:

```python
import jax.numpy as jnp

from jaxmodel import HighLevelNLPBuilder


PARAM_NAME = "x"
N_X = 5
N_Y = 10


def objective_fun(params, vars_dict):
    y = vars_dict["y"]
    return 0.5 * jnp.dot(y, y)


def eq_block(params, vars_dict):
    x = params[PARAM_NAME]
    y = vars_dict["y"]
    return jnp.array([
        y[0] + y[1] - x[0],
        y[2] ** 2 + y[3] - x[1],
    ])


def ineq_block(params, vars_dict):
    y = vars_dict["y"]
    return jnp.array([
        y[0] ** 2 + y[4] ** 2 - 2.0,
    ])


def build_model(*, dtype=jnp.float64):
    zeros = jnp.zeros((N_Y, N_X), dtype=dtype)
    lower = -4.0 * jnp.ones((N_Y,), dtype=dtype)
    upper = 4.0 * jnp.ones((N_Y,), dtype=dtype)

    return (
        HighLevelNLPBuilder(dtype=dtype)
        .add_parameter(PARAM_NAME, N_X)
        .add_variable("y", N_Y)
        .set_objective(objective_fun)
        .add_nonlinear_equality(eq_block, name="eq_block")
        .add_nonlinear_inequality(ineq_block, name="ineq_block")
        .set_affine_lower_bound(var_name="y", param_name=PARAM_NAME, M=zeros, c=lower)
        .set_affine_upper_bound(var_name="y", param_name=PARAM_NAME, M=zeros, c=upper)
        .build(example_params={PARAM_NAME: jnp.zeros((N_X,), dtype=dtype)}, jit_compile=True)
    )
```

Run the block general case through `main.py`:

```sh
python main.py --type general --action run
```

The block style is the fastest route for large structured systems because the
constraints are already native JAX functions and no symbolic expression tree is
constructed.

## Choosing A Style

Use `run_general.py` with `ProblemBuilder` when:

- The problem is small or medium sized.
- You want a readable algebraic definition.
- You want inverse parameters estimated from data.

Use `case/general/model_definition.py` with block functions when:

- Constraint blocks are naturally vectorized.
- The problem has many similar equations or inequalities.
- You want maximum preprocessing and JAX tracing speed.
