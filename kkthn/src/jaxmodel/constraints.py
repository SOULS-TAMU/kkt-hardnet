from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union
import jax
import jax.numpy as jnp
from .variables import VariableSpec
from .types import ParamsDict, VarsDict, VectorModelFun


def _as_1d(z: jnp.ndarray) -> jnp.ndarray:
    z = jnp.asarray(z)
    if z.ndim == 0:
        return z[None]
    if z.ndim != 1:
        raise ValueError(f"Expected 1D output, got shape {z.shape}")
    return z


def wrap_vector_constraint(var_spec: VariableSpec, fun: VectorModelFun):
    def wrapped(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        vars_dict = var_spec.unpack(y_flat)
        return _as_1d(fun(params, vars_dict))
    return wrapped


@dataclass
class ConstraintEntry:
    name: str
    kind: str   # "eq" or "ineq"
    fun: callable
    structure: str = "nonlinear"
    jac_y_fun: Optional[callable] = None
    metadata: Optional[dict[str, Any]] = None


def _validate_quadratic_shapes(var_spec: VariableSpec, Q: jnp.ndarray, c: jnp.ndarray):
    n = var_spec.total_size
    Q = jnp.asarray(Q)
    c = jnp.asarray(c)
    if Q.shape != (n, n):
        raise ValueError(f"Q must have shape {(n, n)}, got {Q.shape}")
    if c.shape != (n,):
        raise ValueError(f"c must have shape {(n,)}, got {c.shape}")
    return Q, c


def eq_scalar(
    fun,
    var_spec: VariableSpec,
    name: str = "eq_scalar",
    structure: str = "nonlinear",
    jac_y_fun: Optional[callable] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConstraintEntry:
    wrapped = wrap_vector_constraint(var_spec, lambda p, v: jnp.array([fun(p, v)]))
    return ConstraintEntry(
        name=name,
        kind="eq",
        fun=wrapped,
        structure=structure,
        jac_y_fun=jac_y_fun,
        metadata=metadata,
    )


def ineq_scalar(
    fun,
    var_spec: VariableSpec,
    name: str = "ineq_scalar",
    structure: str = "nonlinear",
    jac_y_fun: Optional[callable] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConstraintEntry:
    wrapped = wrap_vector_constraint(var_spec, lambda p, v: jnp.array([fun(p, v)]))
    return ConstraintEntry(
        name=name,
        kind="ineq",
        fun=wrapped,
        structure=structure,
        jac_y_fun=jac_y_fun,
        metadata=metadata,
    )


def eq_block(
    fun,
    var_spec: VariableSpec,
    name: str = "eq_block",
    structure: str = "nonlinear",
    jac_y_fun: Optional[callable] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConstraintEntry:
    wrapped = wrap_vector_constraint(var_spec, fun)
    return ConstraintEntry(
        name=name,
        kind="eq",
        fun=wrapped,
        structure=structure,
        jac_y_fun=jac_y_fun,
        metadata=metadata,
    )


def ineq_block(
    fun,
    var_spec: VariableSpec,
    name: str = "ineq_block",
    structure: str = "nonlinear",
    jac_y_fun: Optional[callable] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConstraintEntry:
    wrapped = wrap_vector_constraint(var_spec, fun)
    return ConstraintEntry(
        name=name,
        kind="ineq",
        fun=wrapped,
        structure=structure,
        jac_y_fun=jac_y_fun,
        metadata=metadata,
    )


def quadratic_eq_scalar(
    var_spec: VariableSpec,
    Q: jnp.ndarray,
    c: jnp.ndarray,
    rhs_const: Union[float, jnp.ndarray] = 0.0,
    x_coeff: Optional[jnp.ndarray] = None,
    x_name: Optional[str] = None,
    name: str = "quadratic_eq",
) -> ConstraintEntry:
    Q, c = _validate_quadratic_shapes(var_spec, Q, c)
    Qsym = 0.5 * (Q + Q.T)
    rhs_const = jnp.asarray(rhs_const)

    if (x_coeff is None) != (x_name is None):
        raise ValueError("x_coeff and x_name must be provided together.")

    def fun(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        rhs = rhs_const
        if x_coeff is not None:
            rhs = rhs + jnp.asarray(x_coeff) @ jnp.ravel(params[x_name])
        val = 0.5 * y_flat @ Qsym @ y_flat + c @ y_flat - rhs
        return jnp.array([val])

    def jac_y_fun(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        return (Qsym @ y_flat + c)[None, :]

    return ConstraintEntry(
        name=name,
        kind="eq",
        fun=fun,
        structure="quadratic",
        jac_y_fun=jac_y_fun,
        metadata={
            "type": "quadratic_scalar",
            "Q": Qsym,
            "c": c,
            "rhs_const": rhs_const,
            "x_coeff": x_coeff,
            "x_name": x_name,
        },
    )


def quadratic_ineq_scalar(
    var_spec: VariableSpec,
    Q: jnp.ndarray,
    c: jnp.ndarray,
    rhs_const: Union[float, jnp.ndarray] = 0.0,
    x_coeff: Optional[jnp.ndarray] = None,
    x_name: Optional[str] = None,
    name: str = "quadratic_ineq",
) -> ConstraintEntry:
    Q, c = _validate_quadratic_shapes(var_spec, Q, c)
    Qsym = 0.5 * (Q + Q.T)
    rhs_const = jnp.asarray(rhs_const)

    if (x_coeff is None) != (x_name is None):
        raise ValueError("x_coeff and x_name must be provided together.")

    def fun(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        rhs = rhs_const
        if x_coeff is not None:
            rhs = rhs + jnp.asarray(x_coeff) @ jnp.ravel(params[x_name])
        val = 0.5 * y_flat @ Qsym @ y_flat + c @ y_flat - rhs
        return jnp.array([val])

    def jac_y_fun(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        return (Qsym @ y_flat + c)[None, :]

    return ConstraintEntry(
        name=name,
        kind="ineq",
        fun=fun,
        structure="quadratic",
        jac_y_fun=jac_y_fun,
        metadata={
            "type": "quadratic_scalar",
            "Q": Qsym,
            "c": c,
            "rhs_const": rhs_const,
            "x_coeff": x_coeff,
            "x_name": x_name,
        },
    )


def aggregate_constraints(entries: Sequence[ConstraintEntry], dtype=jnp.float64):
    def agg(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        if len(entries) == 0:
            return jnp.zeros((0,), dtype=dtype)
        vals = [e.fun(params, y_flat) for e in entries]
        return jnp.concatenate([_as_1d(v) for v in vals], axis=0)
    return agg


def aggregate_constraint_jacobian(
    entries: Sequence[ConstraintEntry],
    n_vars: int,
    dtype=jnp.float64,
):
    jac_parts = []
    for entry in entries:
        if entry.jac_y_fun is not None:
            jac_parts.append(lambda y, p, jf=entry.jac_y_fun: jf(p, y))
        else:
            jac_parts.append(jax.jacrev(lambda y, p, f=entry.fun: f(p, y), argnums=0))

    def agg_jac(y: jnp.ndarray, params: ParamsDict) -> jnp.ndarray:
        if len(jac_parts) == 0:
            return jnp.zeros((0, n_vars), dtype=dtype)
        mats = [jp(y, params) for jp in jac_parts]
        return jnp.concatenate(mats, axis=0)

    return agg_jac
