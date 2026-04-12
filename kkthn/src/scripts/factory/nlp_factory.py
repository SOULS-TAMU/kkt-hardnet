from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jaxmodel import HighLevelNLPBuilder

from scripts.misc.nlp_generator import NLPGenerator
from scripts.misc.solver_config import resolve_solver_name

jax.config.update("jax_enable_x64", True)

SUPPORTED_PROBLEM_TYPES = ("nlp",)


def normalize_problem_type(problem_type: str) -> str:
    normalized = str(problem_type).strip().lower()
    if normalized not in SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            f"Unsupported problem type '{problem_type}'. "
            f"Supported types: {', '.join(SUPPORTED_PROBLEM_TYPES)}."
        )
    return normalized


def build_problem_generator(data_cfg: Mapping[str, object]) -> NLPGenerator:
    problem_type = normalize_problem_type(str(data_cfg["type"]))
    if problem_type != "nlp":
        raise ValueError(f"Unsupported NLP problem type '{problem_type}'")

    n_x = int(data_cfg["n_x"])
    n_y = int(data_cfg["n_y"])
    n_eq = int(data_cfg["n_eq"])
    n_ineq = int(data_cfg["n_ineq"])
    seed = int(data_cfg["seed"])
    x_L = np.asarray(data_cfg["x_L"], dtype=float)
    x_U = np.asarray(data_cfg["x_U"], dtype=float)
    solver_name = resolve_solver_name(data_cfg, default="SCS")

    gen = NLPGenerator(
        n_y=n_y,
        n_x=n_x,
        n_eq=n_eq,
        n_ineq=n_ineq,
        seed=seed,
        is_diag_Q=bool(data_cfg.get("is_diag_Q", False)),
    )
    gen.set_solver(solver_name)
    gen.build_problem_data(
        x_L=x_L,
        x_U=x_U,
        q_diag_shift=float(data_cfg.get("q_diag_shift", 0.5)),
        nl_margin=float(data_cfg.get("nl_margin", 1.0)),
        bound_margin=float(data_cfg.get("bound_margin", 1.0)),
        bound_scale=float(data_cfg.get("bound_scale", 0.2)),
        param_scale=float(data_cfg.get("param_scale", 0.4)),
        preview_num_samples=int(data_cfg.get("N_points", data_cfg.get("N_samples", 0))),
    )

    if getattr(gen, "requested_solver", "cvxpy") == "cvxpy":
        gen.build_cvxpy_problem(solver=solver_name)
    return gen


def build_problem_data(data_cfg: Mapping[str, object]):
    return build_problem_generator(data_cfg).get_problem_data()


def build_problem_model_from_data(problem_data: Mapping[str, object], *, dtype=jnp.float64):
    n_x = int(problem_data["n_x"])
    n_y = int(problem_data["n_y"])
    n_eq = int(problem_data["n_eq"])
    n_ineq = int(problem_data["n_ineq"])

    Q = jnp.asarray(problem_data["Q"], dtype=dtype)
    c = jnp.asarray(problem_data["c"], dtype=dtype)
    A = jnp.asarray(problem_data["A"], dtype=dtype)
    b = jnp.asarray(problem_data["b"], dtype=dtype)
    B = jnp.asarray(problem_data["B"], dtype=dtype)
    a = jnp.asarray(problem_data["a"], dtype=dtype)
    W = jnp.asarray(problem_data["W"], dtype=dtype)
    beta = jnp.asarray(problem_data["beta"], dtype=dtype)
    E = jnp.asarray(problem_data["E"], dtype=dtype)
    l = jnp.asarray(problem_data["l"], dtype=dtype)
    L = jnp.asarray(problem_data["L"], dtype=dtype)
    u = jnp.asarray(problem_data["u"], dtype=dtype)
    U = jnp.asarray(problem_data["U"], dtype=dtype)

    builder = (
        HighLevelNLPBuilder(dtype=dtype)
        .add_variable("y", n_y)
        .add_parameter("x", n_x)
        .set_quadratic_objective(Q=Q, c=c)
    )

    if n_eq > 0:
        builder = builder.add_affine_equality(
            var_name="y",
            A=A,
            rhs_const=b,
            param_terms=[(B, "x")],
            name="eq_block",
        )

    if n_ineq > 0:

        def nonlinear_ineq(params, vars, *, a=a, W=W, beta=beta, E=E):
            y = vars["y"]
            x = jnp.ravel(params["x"])
            exp_term = a @ jnp.exp(y)
            quad_term = jnp.einsum("i,mij,j->m", y, W, y)
            rhs = beta + E @ x
            return exp_term + quad_term - rhs

        builder = builder.add_nonlinear_inequality(nonlinear_ineq, name="ineq_nlp")

    params = {"x": jnp.zeros((n_x,), dtype=dtype)}
    return (
        builder
        .set_affine_lower_bound(var_name="y", param_name="x", M=L, c=l)
        .set_affine_upper_bound(var_name="y", param_name="x", M=U, c=u)
        .build(example_params=params, jit_compile=True)
    )


def build_problem_model(data_cfg: Mapping[str, object], *, dtype=jnp.float64):
    return build_problem_model_from_data(build_problem_data(data_cfg), dtype=dtype)
