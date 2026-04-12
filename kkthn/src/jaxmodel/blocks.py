from typing import Optional, Sequence, Tuple, Callable
import jax.numpy as jnp
from .variables import VariableSpec
from .constraints import eq_block, ineq_block


def _build_affine_jacobian_matrix(
    var_spec: VariableSpec,
    y_blocks: Sequence[Tuple[jnp.ndarray, str]],
) -> jnp.ndarray:
    n = var_spec.total_size
    if len(y_blocks) == 0:
        return jnp.zeros((0, n), dtype=jnp.float64)

    m = int(jnp.asarray(y_blocks[0][0]).shape[0])
    full = jnp.zeros((m, n), dtype=jnp.asarray(y_blocks[0][0]).dtype)

    for mat, yname in y_blocks:
        block = jnp.asarray(mat)
        sl = var_spec.slices[yname]
        full = full.at[:, sl].add(block)
    return full


def affine_eq_from_parts(
    var_spec: VariableSpec,
    y_blocks: Sequence[Tuple[jnp.ndarray, str]],
    rhs_const: Optional[jnp.ndarray] = None,
    x_blocks: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    name: str = "affine_eq",
):
    rhs_const = 0.0 if rhs_const is None else rhs_const
    x_blocks = [] if x_blocks is None else list(x_blocks)

    def fun(params, vars):
        lhs = 0.0
        for A, yname in y_blocks:
            lhs = lhs + A @ jnp.ravel(vars[yname])

        rhs = jnp.asarray(rhs_const)
        for D, xname in x_blocks:
            rhs = rhs + D @ jnp.ravel(params[xname])

        return lhs - rhs

    jac_matrix = _build_affine_jacobian_matrix(var_spec, y_blocks)
    jac_y_fun = lambda params, y, J=jac_matrix: J

    return eq_block(
        fun,
        var_spec,
        name=name,
        structure="affine",
        jac_y_fun=jac_y_fun,
        metadata={
            "type": "affine_block",
            "jac_matrix": jac_matrix,
            "rhs_const": jnp.asarray(rhs_const),
            "x_blocks": list(x_blocks),
        },
    )


def affine_ineq_from_parts(
    var_spec: VariableSpec,
    y_blocks: Sequence[Tuple[jnp.ndarray, str]],
    rhs_const: Optional[jnp.ndarray] = None,
    x_blocks: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    name: str = "affine_ineq",
):
    rhs_const = 0.0 if rhs_const is None else rhs_const
    x_blocks = [] if x_blocks is None else list(x_blocks)

    def fun(params, vars):
        lhs = 0.0
        for C, yname in y_blocks:
            lhs = lhs + C @ jnp.ravel(vars[yname])

        rhs = jnp.asarray(rhs_const)
        for E, xname in x_blocks:
            rhs = rhs + E @ jnp.ravel(params[xname])

        return lhs - rhs

    jac_matrix = _build_affine_jacobian_matrix(var_spec, y_blocks)
    jac_y_fun = lambda params, y, J=jac_matrix: J

    return ineq_block(
        fun,
        var_spec,
        name=name,
        structure="affine",
        jac_y_fun=jac_y_fun,
        metadata={
            "type": "affine_block",
            "jac_matrix": jac_matrix,
            "rhs_const": jnp.asarray(rhs_const),
            "x_blocks": list(x_blocks),
        },
    )


def nonlinear_eq_from_parts(
    var_spec: VariableSpec,
    linear_y_terms: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    nonlinear_y_terms: Optional[Sequence[Tuple[jnp.ndarray, Callable, str]]] = None,
    rhs_const: Optional[jnp.ndarray] = None,
    x_blocks: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    x_direct_name: Optional[str] = None,
    name: str = "nonlinear_eq",
):
    linear_y_terms = [] if linear_y_terms is None else list(linear_y_terms)
    nonlinear_y_terms = [] if nonlinear_y_terms is None else list(nonlinear_y_terms)
    x_blocks = [] if x_blocks is None else list(x_blocks)
    rhs_const = 0.0 if rhs_const is None else rhs_const

    def fun(params, vars):
        lhs = 0.0
        for A, yname in linear_y_terms:
            lhs = lhs + A @ jnp.ravel(vars[yname])

        for B, phi, yname in nonlinear_y_terms:
            lhs = lhs + B @ phi(jnp.ravel(vars[yname]))

        rhs = jnp.asarray(rhs_const)
        for D, xname in x_blocks:
            rhs = rhs + D @ jnp.ravel(params[xname])

        if x_direct_name is not None:
            rhs = rhs + jnp.ravel(params[x_direct_name])

        return lhs - rhs

    return eq_block(fun, var_spec, name=name)


def nonlinear_ineq_from_parts(
    var_spec: VariableSpec,
    linear_y_terms: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    nonlinear_y_terms: Optional[Sequence[Tuple[jnp.ndarray, Callable, str]]] = None,
    rhs_const: Optional[jnp.ndarray] = None,
    x_blocks: Optional[Sequence[Tuple[jnp.ndarray, str]]] = None,
    name: str = "nonlinear_ineq",
):
    linear_y_terms = [] if linear_y_terms is None else list(linear_y_terms)
    nonlinear_y_terms = [] if nonlinear_y_terms is None else list(nonlinear_y_terms)
    x_blocks = [] if x_blocks is None else list(x_blocks)
    rhs_const = 0.0 if rhs_const is None else rhs_const

    def fun(params, vars):
        lhs = 0.0
        for C, yname in linear_y_terms:
            lhs = lhs + C @ jnp.ravel(vars[yname])

        for F, phi, yname in nonlinear_y_terms:
            lhs = lhs + F @ phi(jnp.ravel(vars[yname]))

        rhs = jnp.asarray(rhs_const)
        for E, xname in x_blocks:
            rhs = rhs + E @ jnp.ravel(params[xname])

        return lhs - rhs

    return ineq_block(fun, var_spec, name=name)
