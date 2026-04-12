from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union
import jax.numpy as jnp

from .blocks import affine_eq_from_parts, affine_ineq_from_parts
from .builder import NLPBuilder
from .constraints import (
    eq_block,
    ineq_block,
    quadratic_eq_scalar,
    quadratic_ineq_scalar,
)
from .objective import QuadraticObjective
from .parameters import ParameterSpec
from .variables import VariableBuilder, VariableSpec


@dataclass
class _AffineBound:
    param_name: str
    M: jnp.ndarray
    c: jnp.ndarray


class HighLevelNLPBuilder:
    """High-level modeling interface built on top of NLPBuilder.

    This class keeps the same functionality as the low-level API but provides
    easier methods for users to define common model structures.
    """

    def __init__(self, dtype=jnp.float64):
        self.dtype = dtype
        self._vb = VariableBuilder()
        self._param_sizes: dict[str, int] = {}
        self._objective_fun = None
        self._constraints = []
        self._lower_bounds: dict[str, _AffineBound] = {}
        self._upper_bounds: dict[str, _AffineBound] = {}

    def add_variable(self, name: str, size: int) -> "HighLevelNLPBuilder":
        self._vb.add_vector(name, size)
        return self

    def add_parameter(self, name: str, size: int) -> "HighLevelNLPBuilder":
        if name in self._param_sizes:
            raise ValueError(f"Parameter '{name}' already exists")
        self._param_sizes[name] = int(size)
        return self

    def set_objective(self, objective_fun) -> "HighLevelNLPBuilder":
        self._objective_fun = objective_fun
        return self

    def set_quadratic_objective(
        self,
        Q: jnp.ndarray,
        c: jnp.ndarray,
        constant: float = 0.0,
    ) -> "HighLevelNLPBuilder":
        self._objective_fun = QuadraticObjective(Q=Q, c=c, constant=constant)
        return self

    def add_affine_equality(
        self,
        *,
        var_name: str,
        A: jnp.ndarray,
        rhs_const: Union[jnp.ndarray, float] = 0.0,
        param_terms: Optional[list[tuple[jnp.ndarray, str]]] = None,
        name: str = "affine_eq",
    ) -> "HighLevelNLPBuilder":
        param_terms = [] if param_terms is None else param_terms
        self._constraints.append(
            ("affine_eq", dict(var_name=var_name, A=A, rhs_const=rhs_const, param_terms=param_terms, name=name))
        )
        return self

    def add_affine_inequality(
        self,
        *,
        var_name: str,
        C: jnp.ndarray,
        rhs_const: Union[jnp.ndarray, float] = 0.0,
        param_terms: Optional[list[tuple[jnp.ndarray, str]]] = None,
        name: str = "affine_ineq",
    ) -> "HighLevelNLPBuilder":
        param_terms = [] if param_terms is None else param_terms
        self._constraints.append(
            ("affine_ineq", dict(var_name=var_name, C=C, rhs_const=rhs_const, param_terms=param_terms, name=name))
        )
        return self

    def add_nonlinear_equality(
        self,
        fun: Callable,
        name: str = "nonlinear_eq",
    ) -> "HighLevelNLPBuilder":
        self._constraints.append(("eq_block", dict(fun=fun, name=name)))
        return self

    def add_nonlinear_inequality(
        self,
        fun: Callable,
        name: str = "nonlinear_ineq",
    ) -> "HighLevelNLPBuilder":
        self._constraints.append(("ineq_block", dict(fun=fun, name=name)))
        return self

    def add_quadratic_equality(
        self,
        *,
        Q: jnp.ndarray,
        c: jnp.ndarray,
        rhs_const: Union[jnp.ndarray, float] = 0.0,
        x_coeff: Optional[jnp.ndarray] = None,
        x_name: Optional[str] = None,
        name: str = "quadratic_eq",
    ) -> "HighLevelNLPBuilder":
        self._constraints.append(
            (
                "quadratic_eq",
                dict(Q=Q, c=c, rhs_const=rhs_const, x_coeff=x_coeff, x_name=x_name, name=name),
            )
        )
        return self

    def add_quadratic_inequality(
        self,
        *,
        Q: jnp.ndarray,
        c: jnp.ndarray,
        rhs_const: Union[jnp.ndarray, float] = 0.0,
        x_coeff: Optional[jnp.ndarray] = None,
        x_name: Optional[str] = None,
        name: str = "quadratic_ineq",
    ) -> "HighLevelNLPBuilder":
        self._constraints.append(
            (
                "quadratic_ineq",
                dict(Q=Q, c=c, rhs_const=rhs_const, x_coeff=x_coeff, x_name=x_name, name=name),
            )
        )
        return self

    def set_affine_lower_bound(
        self,
        *,
        var_name: str,
        param_name: str,
        M: jnp.ndarray,
        c: jnp.ndarray,
    ) -> "HighLevelNLPBuilder":
        self._lower_bounds[var_name] = _AffineBound(param_name=param_name, M=jnp.asarray(M), c=jnp.asarray(c))
        return self

    def set_affine_upper_bound(
        self,
        *,
        var_name: str,
        param_name: str,
        M: jnp.ndarray,
        c: jnp.ndarray,
    ) -> "HighLevelNLPBuilder":
        self._upper_bounds[var_name] = _AffineBound(param_name=param_name, M=jnp.asarray(M), c=jnp.asarray(c))
        return self

    def _build_parameter_spec(self, example_params: dict[str, jnp.ndarray]) -> ParameterSpec:
        missing = [k for k in self._param_sizes if k not in example_params]
        if missing:
            raise KeyError(f"Missing parameter examples for: {missing}")

        shapes = {}
        for name, size in self._param_sizes.items():
            shape = tuple(jnp.asarray(example_params[name]).shape)
            if shape != (size,):
                raise ValueError(f"Parameter '{name}' expected shape {(size,)}, got {shape}")
            shapes[name] = shape
        return ParameterSpec(names=list(self._param_sizes.keys()), shapes=shapes)

    def _build_bounds(self, var_spec: VariableSpec):
        if len(self._lower_bounds) == 0 and len(self._upper_bounds) == 0:
            return None, None

        for var_name, bnd in {**self._lower_bounds, **self._upper_bounds}.items():
            if var_name not in var_spec.shapes:
                raise KeyError(f"Unknown variable block '{var_name}' in bounds")
            n = var_spec.sizes[var_name]
            if bnd.M.shape[0] != n:
                raise ValueError(f"Bound matrix for '{var_name}' must have {n} rows, got {bnd.M.shape[0]}")
            if bnd.c.shape != (n,):
                raise ValueError(f"Bound vector for '{var_name}' must have shape {(n,)}, got {bnd.c.shape}")

        def lower_fun(params):
            blocks = {}
            for name in var_spec.names:
                n = var_spec.sizes[name]
                if name in self._lower_bounds:
                    bnd = self._lower_bounds[name]
                    blocks[name] = bnd.M @ jnp.ravel(params[bnd.param_name]) + bnd.c
                else:
                    blocks[name] = -jnp.inf * jnp.ones((n,), dtype=self.dtype)
            return var_spec.pack(blocks, dtype=self.dtype)

        def upper_fun(params):
            blocks = {}
            for name in var_spec.names:
                n = var_spec.sizes[name]
                if name in self._upper_bounds:
                    bnd = self._upper_bounds[name]
                    blocks[name] = bnd.M @ jnp.ravel(params[bnd.param_name]) + bnd.c
                else:
                    blocks[name] = jnp.inf * jnp.ones((n,), dtype=self.dtype)
            return var_spec.pack(blocks, dtype=self.dtype)

        return lower_fun, upper_fun

    def build(self, *, example_params: dict[str, jnp.ndarray], jit_compile: bool = True):
        if self._objective_fun is None:
            raise ValueError("Objective must be set before build().")

        var_spec = self._vb.build()
        parameter_spec = self._build_parameter_spec(example_params)
        lower_fun, upper_fun = self._build_bounds(var_spec)

        builder = NLPBuilder(var_spec=var_spec, parameter_spec=parameter_spec, dtype=self.dtype)
        builder.set_objective(self._objective_fun)

        for kind, payload in self._constraints:
            if kind == "affine_eq":
                builder.add_constraint(
                    affine_eq_from_parts(
                        var_spec=var_spec,
                        y_blocks=[(jnp.asarray(payload["A"]), payload["var_name"])],
                        rhs_const=payload["rhs_const"],
                        x_blocks=payload["param_terms"],
                        name=payload["name"],
                    )
                )
            elif kind == "affine_ineq":
                builder.add_constraint(
                    affine_ineq_from_parts(
                        var_spec=var_spec,
                        y_blocks=[(jnp.asarray(payload["C"]), payload["var_name"])],
                        rhs_const=payload["rhs_const"],
                        x_blocks=payload["param_terms"],
                        name=payload["name"],
                    )
                )
            elif kind == "eq_block":
                builder.add_constraint(eq_block(payload["fun"], var_spec, name=payload["name"]))
            elif kind == "ineq_block":
                builder.add_constraint(ineq_block(payload["fun"], var_spec, name=payload["name"]))
            elif kind == "quadratic_eq":
                builder.add_constraint(
                    quadratic_eq_scalar(
                        var_spec=var_spec,
                        Q=jnp.asarray(payload["Q"]),
                        c=jnp.asarray(payload["c"]),
                        rhs_const=payload["rhs_const"],
                        x_coeff=None if payload["x_coeff"] is None else jnp.asarray(payload["x_coeff"]),
                        x_name=payload["x_name"],
                        name=payload["name"],
                    )
                )
            elif kind == "quadratic_ineq":
                builder.add_constraint(
                    quadratic_ineq_scalar(
                        var_spec=var_spec,
                        Q=jnp.asarray(payload["Q"]),
                        c=jnp.asarray(payload["c"]),
                        rhs_const=payload["rhs_const"],
                        x_coeff=None if payload["x_coeff"] is None else jnp.asarray(payload["x_coeff"]),
                        x_name=payload["x_name"],
                        name=payload["name"],
                    )
                )
            else:
                raise ValueError(f"Unknown constraint kind: {kind}")

        if lower_fun is not None or upper_fun is not None:
            builder.set_bounds(lower_fun=lower_fun, upper_fun=upper_fun)

        return builder.build(jit_compile=jit_compile)
