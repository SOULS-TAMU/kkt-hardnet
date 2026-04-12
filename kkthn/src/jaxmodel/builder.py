from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import jax.numpy as jnp

from .bounds import BoundSpec
from .constraints import ConstraintEntry, eq_block, eq_scalar, ineq_block, ineq_scalar
from .model import JaxNLPModel
from .parameters import ParameterSpec
from .variables import VariableSpec


@dataclass
class NLPBuilder:
    var_spec: VariableSpec
    parameter_spec: Optional[ParameterSpec] = None
    dtype: jnp.dtype = jnp.float64
    _objective_fun: Optional[Callable] = None
    _constraints: list[ConstraintEntry] = field(default_factory=list)
    _bounds: BoundSpec = field(default_factory=BoundSpec)

    def set_objective(self, objective_fun) -> "NLPBuilder":
        self._objective_fun = objective_fun
        return self

    def add_constraint(self, constraint: ConstraintEntry) -> "NLPBuilder":
        self._constraints.append(constraint)
        return self

    def add_constraints(self, constraints: Sequence[ConstraintEntry]) -> "NLPBuilder":
        self._constraints.extend(constraints)
        return self

    def add_eq_scalar(self, fun, name: str = "eq_scalar") -> "NLPBuilder":
        return self.add_constraint(eq_scalar(fun, self.var_spec, name=name))

    def add_ineq_scalar(self, fun, name: str = "ineq_scalar") -> "NLPBuilder":
        return self.add_constraint(ineq_scalar(fun, self.var_spec, name=name))

    def add_eq_block(self, fun, name: str = "eq_block") -> "NLPBuilder":
        return self.add_constraint(eq_block(fun, self.var_spec, name=name))

    def add_ineq_block(self, fun, name: str = "ineq_block") -> "NLPBuilder":
        return self.add_constraint(ineq_block(fun, self.var_spec, name=name))

    def set_bounds(
        self,
        lower_fun=None,
        upper_fun=None,
    ) -> "NLPBuilder":
        self._bounds = BoundSpec(lower_fun=lower_fun, upper_fun=upper_fun)
        return self

    def build(self, jit_compile: bool = True) -> JaxNLPModel:
        if self._objective_fun is None:
            raise ValueError("Objective function must be set before build().")

        return JaxNLPModel(
            var_spec=self.var_spec,
            objective_fun=self._objective_fun,
            constraints=self._constraints,
            bounds=self._bounds,
            parameter_spec=self.parameter_spec,
            dtype=self.dtype,
            jit_compile=jit_compile,
        )
