from dataclasses import dataclass
from typing import Callable, Dict
import jax.numpy as jnp
from .variables import VariableSpec
from .types import ParamsDict, VarsDict, ScalarModelFun, FlatScalarFun


@dataclass
class QuadraticObjective:
    Q: jnp.ndarray
    c: jnp.ndarray
    constant: float = 0.0

    def __post_init__(self):
        self.Q = jnp.asarray(self.Q)
        self.c = jnp.asarray(self.c)
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError(f"Q must be square, got shape {self.Q.shape}")
        if self.c.ndim != 1 or self.c.shape[0] != self.Q.shape[0]:
            raise ValueError(
                f"c shape {self.c.shape} must match Q dimension {self.Q.shape[0]}"
            )

    def value_from_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * y_flat @ self.Q @ y_flat + self.c @ y_flat + self.constant

    def grad_from_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        return self.Q @ y_flat + self.c

    def hess_from_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        return self.Q

    def diag_hess_from_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        return jnp.diag(self.Q)


def wrap_scalar_objective(
    var_spec: VariableSpec,
    fun: ScalarModelFun,
) -> FlatScalarFun:
    def wrapped(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
        vars_dict = var_spec.unpack(y_flat)
        return jnp.asarray(fun(params, vars_dict))
    return wrapped


def resolve_objective(var_spec: VariableSpec, objective_fun):
    if isinstance(objective_fun, QuadraticObjective):
        n = var_spec.total_size
        if objective_fun.Q.shape != (n, n):
            raise ValueError(
                f"Quadratic objective Q shape must be {(n, n)}, got {objective_fun.Q.shape}"
            )
        if objective_fun.c.shape != (n,):
            raise ValueError(
                f"Quadratic objective c shape must be {(n,)}, got {objective_fun.c.shape}"
            )

        def obj_flat(params: ParamsDict, y_flat: jnp.ndarray) -> jnp.ndarray:
            return objective_fun.value_from_flat(y_flat)

        grad_fun = lambda y, params: objective_fun.grad_from_flat(y)
        hess_fun = lambda y, params: objective_fun.hess_from_flat(y)
        diag_hess_fun = lambda params, y: objective_fun.diag_hess_from_flat(y)
        return "quadratic", obj_flat, grad_fun, hess_fun, diag_hess_fun

    obj_flat = wrap_scalar_objective(var_spec, objective_fun)
    return "nonlinear", obj_flat, None, None, None
