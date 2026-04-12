from __future__ import annotations
from typing import Sequence, Optional
import jax
import jax.numpy as jnp

from .variables import VariableSpec
from .objective import resolve_objective
from .constraints import (
    ConstraintEntry,
    aggregate_constraint_jacobian,
    aggregate_constraints,
)
from .bounds import BoundSpec
from .parameters import ParameterSpec
from .autodiff import (
    build_grad_y_objective,
    build_hess_y_objective,
    build_diag_hess_y_objective,
)
from .approximations import (
    build_sqp_data,
    build_sqp_subproblem_data,
    linearize_constraints,
    linearize_constraints_data,
    quadraticize_objective,
    quadraticize_objective_data,
)


class JaxNLPModel:
    def __init__(
        self,
        var_spec: VariableSpec,
        objective_fun,
        constraints: Optional[Sequence[ConstraintEntry]] = None,
        bounds: Optional[BoundSpec] = None,
        parameter_spec: Optional[ParameterSpec] = None,
        dtype=jnp.float64,
        jit_compile: bool = True,
    ):
        self.var_spec = var_spec
        self.parameter_spec = parameter_spec
        self.dtype = dtype
        self.bounds = BoundSpec() if bounds is None else bounds
        self.constraints = [] if constraints is None else list(constraints)

        self.eq_constraints = [c for c in self.constraints if c.kind == "eq"]
        self.ineq_constraints = [c for c in self.constraints if c.kind == "ineq"]

        (
            self.objective_structure,
            self._obj_flat,
            grad_fun,
            hess_fun,
            diag_hess_fun,
        ) = resolve_objective(var_spec, objective_fun)

        self._eq_fun = aggregate_constraints(self.eq_constraints, dtype=dtype)
        self._ineq_fun = aggregate_constraints(self.ineq_constraints, dtype=dtype)

        if self.objective_structure == "quadratic":
            self._grad_f_y = grad_fun
            self._hess_f_y = hess_fun
            self._diag_hess_f_y = diag_hess_fun
        else:
            self._grad_f_y = build_grad_y_objective(self._obj_flat)
            self._hess_f_y = build_hess_y_objective(self._obj_flat)
            self._diag_hess_f_y = build_diag_hess_y_objective(self._grad_f_y)

        self.eq_structure = self._detect_constraint_structure(self.eq_constraints)
        self.ineq_structure = self._detect_constraint_structure(self.ineq_constraints)

        self._jac_h_y = aggregate_constraint_jacobian(
            self.eq_constraints,
            n_vars=self.var_spec.total_size,
            dtype=dtype,
        )
        self._jac_g_y = aggregate_constraint_jacobian(
            self.ineq_constraints,
            n_vars=self.var_spec.total_size,
            dtype=dtype,
        )

        if jit_compile:
            self._compile()

    @staticmethod
    def _detect_constraint_structure(constraints: Sequence[ConstraintEntry]) -> str:
        if len(constraints) == 0:
            return "empty"
        structures = {c.structure for c in constraints}
        if len(structures) == 1:
            s = next(iter(structures))
            if s in ("affine", "quadratic", "nonlinear"):
                return s
        return "mixed"

    def _compile(self):
        self.objective_value = jax.jit(self.objective_value)
        self.grad_y_objective = jax.jit(self.grad_y_objective)
        self.hess_y_objective = jax.jit(self.hess_y_objective)
        self.diag_hess_y_objective = jax.jit(self.diag_hess_y_objective)

        self.eq_residual = jax.jit(self.eq_residual)
        self.ineq_residual = jax.jit(self.ineq_residual)
        self.jac_y_eq = jax.jit(self.jac_y_eq)
        self.jac_y_ineq = jax.jit(self.jac_y_ineq)

        self.lower_bounds = jax.jit(self.lower_bounds)
        self.upper_bounds = jax.jit(self.upper_bounds)
        self.bound_residuals = jax.jit(self.bound_residuals)

        self.linearize_constraints = jax.jit(self.linearize_constraints)
        self.linearization_data = jax.jit(self.linearization_data)
        self.quadraticize_objective = jax.jit(
            self.quadraticize_objective,
            static_argnames=("use_diagonal_hessian", "diag_floor"),
        )
        self.quadratic_objective_data = jax.jit(
            self.quadratic_objective_data,
            static_argnames=("use_diagonal_hessian", "diag_floor"),
        )
        self.sqp_data = jax.jit(
            self.sqp_data,
            static_argnames=("use_diagonal_hessian", "diag_floor"),
        )
        self.sqp_subproblem_data = jax.jit(
            self.sqp_subproblem_data,
            static_argnames=("use_diagonal_hessian", "diag_floor"),
        )

    def pack_vars(self, values):
        return self.var_spec.pack(values, dtype=self.dtype)

    def unpack_vars(self, y):
        return self.var_spec.unpack(y)

    def objective_value(self, params, y):
        return self._obj_flat(params, y)

    def validate_params(self, params):
        if self.parameter_spec is not None:
            self.parameter_spec.validate(params)

    def grad_y_objective(self, params, y):
        return self._grad_f_y(y, params)

    def hess_y_objective(self, params, y):
        return self._hess_f_y(y, params)

    def diag_hess_y_objective(self, params, y):
        return self._diag_hess_f_y(params, y)

    def eq_residual(self, params, y):
        return self._eq_fun(params, y)

    def ineq_residual(self, params, y):
        return self._ineq_fun(params, y)

    def jac_y_eq(self, params, y):
        return self._jac_h_y(y, params)

    def jac_y_ineq(self, params, y):
        return self._jac_g_y(y, params)

    def lower_bounds(self, params):
        if self.bounds.lower_fun is None:
            return None
        return self.bounds.lower_fun(params)

    def upper_bounds(self, params):
        if self.bounds.upper_fun is None:
            return None
        return self.bounds.upper_fun(params)

    def bound_residuals(self, params, y):
        lb = jnp.zeros((0,), dtype=y.dtype)
        ub = jnp.zeros((0,), dtype=y.dtype)

        if self.bounds.lower_fun is not None:
            lb = self.bounds.lower_fun(params) - y
        if self.bounds.upper_fun is not None:
            ub = y - self.bounds.upper_fun(params)

        return lb, ub

    def linearize_constraints(self, params, y):
        return linearize_constraints(
            eq_fun=self._eq_fun,
            ineq_fun=self._ineq_fun,
            jac_eq=self._jac_h_y,
            jac_ineq=self._jac_g_y,
            params=params,
            y=y,
        )

    def linearization_data(self, params, y):
        return linearize_constraints_data(
            eq_fun=self._eq_fun,
            ineq_fun=self._ineq_fun,
            jac_eq=self._jac_h_y,
            jac_ineq=self._jac_g_y,
            params=params,
            y=y,
        )

    def quadraticize_objective(
        self,
        params,
        y,
        rho: float = 1.0,
        use_diagonal_hessian: bool = True,
        diag_floor=None,
    ):
        return quadraticize_objective(
            grad_fun=self._grad_f_y,
            hess_fun=self._hess_f_y,
            diag_hess_fun=self._diag_hess_f_y,
            params=params,
            y=y,
            rho=rho,
            use_diagonal_hessian=use_diagonal_hessian,
            diag_floor=diag_floor,
        )

    def quadratic_objective_data(
        self,
        params,
        y,
        rho: float = 1.0,
        use_diagonal_hessian: bool = True,
        diag_floor=None,
    ):
        return quadraticize_objective_data(
            grad_fun=self._grad_f_y,
            hess_fun=self._hess_f_y,
            diag_hess_fun=self._diag_hess_f_y,
            params=params,
            y=y,
            rho=rho,
            use_diagonal_hessian=use_diagonal_hessian,
            diag_floor=diag_floor,
        )

    def sqp_data(
        self,
        params,
        y,
        rho: float = 1.0,
        use_diagonal_hessian: bool = True,
        diag_floor=None,
    ):
        return build_sqp_data(
            eq_fun=self._eq_fun,
            ineq_fun=self._ineq_fun,
            jac_eq=self._jac_h_y,
            jac_ineq=self._jac_g_y,
            grad_fun=self._grad_f_y,
            hess_fun=self._hess_f_y,
            diag_hess_fun=self._diag_hess_f_y,
            lower_fun=self.bounds.lower_fun,
            upper_fun=self.bounds.upper_fun,
            params=params,
            y=y,
            rho=rho,
            use_diagonal_hessian=use_diagonal_hessian,
            diag_floor=diag_floor,
        )

    def sqp_subproblem_data(
        self,
        params,
        y,
        rho: float = 1.0,
        use_diagonal_hessian: bool = True,
        diag_floor=None,
    ):
        return build_sqp_subproblem_data(
            eq_fun=self._eq_fun,
            ineq_fun=self._ineq_fun,
            jac_eq=self._jac_h_y,
            jac_ineq=self._jac_g_y,
            grad_fun=self._grad_f_y,
            hess_fun=self._hess_f_y,
            diag_hess_fun=self._diag_hess_f_y,
            lower_fun=self.bounds.lower_fun,
            upper_fun=self.bounds.upper_fun,
            params=params,
            y=y,
            rho=rho,
            use_diagonal_hessian=use_diagonal_hessian,
            diag_floor=diag_floor,
        )
