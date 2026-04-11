from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "nlpopt" / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from jaxmodel import HighLevelNLPBuilder

PARAM_NAME = "x"

N_X = 5
N_Y = 10
N_EQ = 5
N_INEQ = 5

SEED = 42
BOUND_RADIUS = 2.0


def _rng():
    return np.random.default_rng(SEED)


def _make_spd_matrix(rng: np.random.Generator, n: int, *, dtype) -> jnp.ndarray:
    raw = rng.normal(size=(n, n))
    spd = (raw.T @ raw) / max(1, n)
    spd = spd + np.diag(1.0 + rng.uniform(0.2, 0.8, size=(n,)))
    return jnp.asarray(spd, dtype=dtype)


def _make_objective_vector(rng: np.random.Generator, n: int, *, dtype) -> jnp.ndarray:
    return jnp.asarray(rng.uniform(-0.2, 0.2, size=(n,)), dtype=dtype)


def make_structured_nonlinear_eq_block(A_stack, c_stack, d_vec, *, dtype=jnp.float64):
    A_stack = jnp.asarray(A_stack, dtype=dtype)
    c_stack = jnp.asarray(c_stack, dtype=dtype)
    d_vec = jnp.asarray(d_vec, dtype=dtype)

    def block(params, vars_dict):
        x = params[PARAM_NAME]
        y = vars_dict["y"]
        quad = jnp.einsum("mij,i,j->m", A_stack, y, y)
        linear = c_stack @ y
        return quad + linear + d_vec - x[: A_stack.shape[0]] ** 3

    return block


def make_structured_nonlinear_ineq_block(H_stack, g_stack, r_vec, *, dtype=jnp.float64):
    H_stack = jnp.asarray(H_stack, dtype=dtype)
    g_stack = jnp.asarray(g_stack, dtype=dtype)
    r_vec = jnp.asarray(r_vec, dtype=dtype)

    def block(params, vars_dict):
        del params
        y = vars_dict["y"]
        quad = jnp.einsum("mij,i,j->m", H_stack, y, y)
        linear = g_stack @ y
        return quad + linear - r_vec

    return block


def objective_fun_factory(Q, c_obj):
    Q = jnp.asarray(Q)
    c_obj = jnp.asarray(c_obj)

    def objective_fun(params, vars_dict):
        del params
        y = vars_dict["y"]
        return 0.5 * y @ (Q @ y) + c_obj @ jnp.sin(y)

    return objective_fun


def _make_equality_data(rng: np.random.Generator, *, dtype):
    A_stack = np.zeros((N_EQ, N_Y, N_Y), dtype=np.float64)
    c_stack = np.zeros((N_EQ, N_Y), dtype=np.float64)
    d_vec = np.zeros((N_EQ,), dtype=np.float64)
    for idx in range(N_EQ):
        slack_idx = N_EQ + idx
        A_stack[idx, slack_idx, slack_idx] = 0.08 + 0.02 * rng.uniform()
        c_stack[idx, idx] = 1.0
        c_stack[idx, slack_idx] = 0.05 * rng.normal()
        d_vec[idx] = 0.02 * rng.normal()
    return jnp.asarray(A_stack, dtype=dtype), jnp.asarray(c_stack, dtype=dtype), jnp.asarray(d_vec, dtype=dtype)


def _make_inequality_data(rng: np.random.Generator, *, dtype):
    H_stack = np.zeros((N_INEQ, N_Y, N_Y), dtype=np.float64)
    g_stack = np.zeros((N_INEQ, N_Y), dtype=np.float64)
    r_vec = np.zeros((N_INEQ,), dtype=np.float64)
    for idx in range(N_INEQ):
        main_idx = idx
        slack_idx = N_EQ + idx
        H_stack[idx, main_idx, main_idx] = 0.10 + 0.02 * rng.uniform()
        H_stack[idx, slack_idx, slack_idx] = 1.00 + 0.10 * rng.uniform()
        g_stack[idx, slack_idx] = 0.03 * rng.normal()
        r_vec[idx] = 1.0 + 0.1 * rng.uniform()
    return jnp.asarray(H_stack, dtype=dtype), jnp.asarray(g_stack, dtype=dtype), jnp.asarray(r_vec, dtype=dtype)


def build_model(*, dtype=jnp.float64):
    rng = _rng()
    Q = _make_spd_matrix(rng, N_Y, dtype=dtype)
    c_obj = _make_objective_vector(rng, N_Y, dtype=dtype)
    eq_A, eq_c, eq_d = _make_equality_data(rng, dtype=dtype)
    ineq_H, ineq_g, ineq_r = _make_inequality_data(rng, dtype=dtype)

    objective_fun = objective_fun_factory(Q, c_obj)
    eq_block = make_structured_nonlinear_eq_block(eq_A, eq_c, eq_d, dtype=dtype)
    ineq_block = make_structured_nonlinear_ineq_block(ineq_H, ineq_g, ineq_r, dtype=dtype)

    zeros_bounds = jnp.zeros((N_Y, N_X), dtype=dtype)
    lower_bounds = -BOUND_RADIUS * jnp.ones((N_Y,), dtype=dtype)
    upper_bounds = BOUND_RADIUS * jnp.ones((N_Y,), dtype=dtype)
    params0 = {PARAM_NAME: jnp.zeros((N_X,), dtype=dtype)}

    return (
        HighLevelNLPBuilder(dtype=dtype)
        .add_parameter(PARAM_NAME, N_X)
        .add_variable("y", N_Y)
        .set_objective(objective_fun)
        .add_nonlinear_equality(eq_block, name="nonlinear_eq_block")
        .add_nonlinear_inequality(ineq_block, name="quadratic_ineq_block")
        .set_affine_lower_bound(var_name="y", param_name=PARAM_NAME, M=zeros_bounds, c=lower_bounds)
        .set_affine_upper_bound(var_name="y", param_name=PARAM_NAME, M=zeros_bounds, c=upper_bounds)
        .build(example_params=params0, jit_compile=True)
    )
