from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxmodel import HighLevelNLPBuilder  # noqa: E402


def _safe_env():
    return {
        "jnp": jnp,
        "sin": jnp.sin,
        "cos": jnp.cos,
        "tan": jnp.tan,
        "exp": jnp.exp,
        "log": jnp.log,
        "sqrt": jnp.sqrt,
        "abs": jnp.abs,
        "maximum": jnp.maximum,
        "minimum": jnp.minimum,
        "pi": jnp.pi,
    }


def parse_constraint(constraint: str) -> tuple[str, str]:
    c = str(constraint).strip()
    if "==" in c:
        left, right = c.split("==", 1)
        return f"({left.strip()}) - ({right.strip()})", "eq"
    if "<=" in c:
        left, right = c.split("<=", 1)
        return f"({left.strip()}) - ({right.strip()})", "ineq"
    if ">=" in c:
        left, right = c.split(">=", 1)
        return f"({right.strip()}) - ({left.strip()})", "ineq"
    raise ValueError(f"Unsupported constraint format: {constraint}")


def make_scalar_expr_fn(
    expr: str,
    parameter_names: list[str],
    variable_names: list[str],
    inverse_parameter_names: list[str] | None = None,
    inverse_values=None,
    train_inverse: bool = False,
) -> Callable:
    env0 = _safe_env()
    inv_names = list(inverse_parameter_names or [])
    inv_values = jnp.asarray([] if inverse_values is None else inverse_values, dtype=jnp.float64).reshape(-1)
    forward_dim = len(parameter_names)

    def f(y, x):
        env = dict(env0)
        y_vec = jnp.ravel(y)
        x_vec = jnp.ravel(x)
        for idx, name in enumerate(variable_names):
            env[name] = y_vec[idx]
        for idx, name in enumerate(parameter_names):
            env[name] = x_vec[idx]
        for idx, name in enumerate(inv_names):
            env[name] = x_vec[forward_dim + idx] if train_inverse else inv_values[idx]
        return eval(expr, {"__builtins__": {}}, env)

    return f


def _objective_expr(problem: dict) -> str:
    expr = str(problem.get("objective", "")).strip()
    if expr:
        return expr
    terms = [f"{name}**2" for name in problem["variables"]]
    return "0.5*(" + " + ".join(terms) + ")"


def build_model_from_string_problem(
    problem: dict,
    *,
    dtype=jnp.float64,
    inverse_parameter_names: list[str] | None = None,
    inverse_values=None,
    train_inverse: bool = False,
):
    parameter_names = list(problem["parameters"])
    variable_names = list(problem["variables"])
    inv_names = list(problem.get("inverse_parameters", inverse_parameter_names or []))
    constraints = list(problem.get("constraints", []))
    objective_expr = _objective_expr(problem)
    total_parameter_dim = len(parameter_names) + (len(inv_names) if train_inverse else 0)

    eq_exprs: list[str] = []
    ineq_exprs: list[str] = []
    for constraint in constraints:
        expr, sense = parse_constraint(constraint)
        if sense == "eq":
            eq_exprs.append(expr)
        else:
            ineq_exprs.append(expr)

    eq_fns = [
        make_scalar_expr_fn(
            expr,
            parameter_names,
            variable_names,
            inv_names,
            inverse_values=inverse_values,
            train_inverse=train_inverse,
        )
        for expr in eq_exprs
    ]
    ineq_fns = [
        make_scalar_expr_fn(
            expr,
            parameter_names,
            variable_names,
            inv_names,
            inverse_values=inverse_values,
            train_inverse=train_inverse,
        )
        for expr in ineq_exprs
    ]
    obj_fn = make_scalar_expr_fn(
        objective_expr,
        parameter_names,
        variable_names,
        inv_names,
        inverse_values=inverse_values,
        train_inverse=train_inverse,
    )

    def objective(params, vars_dict):
        return obj_fn(vars_dict["y"], params["x"])

    builder = (
        HighLevelNLPBuilder(dtype=dtype)
        .add_parameter("x", total_parameter_dim)
        .add_variable("y", len(variable_names))
        .set_objective(objective)
    )

    if eq_fns:

        def eq_block(params, vars_dict):
            y = vars_dict["y"]
            x = params["x"]
            return jnp.stack([f(y, x) for f in eq_fns], axis=0)

        builder = builder.add_nonlinear_equality(eq_block, name="string_eq_block")

    if ineq_fns:

        def ineq_block(params, vars_dict):
            y = vars_dict["y"]
            x = params["x"]
            return jnp.stack([f(y, x) for f in ineq_fns], axis=0)

        builder = builder.add_nonlinear_inequality(ineq_block, name="string_ineq_block")

    model = builder.build(example_params={"x": jnp.zeros((total_parameter_dim,), dtype=dtype)}, jit_compile=True)
    return model, {
        "eq_exprs": eq_exprs,
        "ineq_exprs": ineq_exprs,
        "objective_expr": objective_expr,
        "parameter_names": parameter_names,
        "inverse_parameter_names": inv_names,
        "total_parameter_dim": total_parameter_dim,
        "train_inverse": bool(train_inverse),
    }
