from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

from jaxmodel import JaxNLPModel

from .cvxpy_solver import solve_jaxmodel_direct, solve_quadratic_program
from .types import QuadraticProgramData, SolveResult


def _to_numpy(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _project_box(y: np.ndarray, l: Optional[np.ndarray], u: Optional[np.ndarray]) -> np.ndarray:
    out = np.asarray(y, dtype=np.float64).copy()
    if l is not None:
        out = np.maximum(out, l)
    if u is not None:
        out = np.minimum(out, u)
    return out


def _default_initial_point(model: JaxNLPModel, params) -> np.ndarray:
    n = model.var_spec.total_size
    lb = _to_numpy(model.lower_bounds(params))
    ub = _to_numpy(model.upper_bounds(params))

    if lb is None and ub is None:
        return np.zeros((n,), dtype=np.float64)

    if lb is None:
        lb = -np.inf * np.ones((n,), dtype=np.float64)
    if ub is None:
        ub = np.inf * np.ones((n,), dtype=np.float64)

    finite_mid = np.where(
        np.isfinite(lb) & np.isfinite(ub),
        0.5 * (lb + ub),
        np.where(np.isfinite(lb), lb + 1.0, np.where(np.isfinite(ub), ub - 1.0, 0.0)),
    )
    return _project_box(finite_mid, lb, ub)


def _primal_residual(model: JaxNLPModel, params, y: np.ndarray) -> float:
    y_jax = jnp.asarray(y, dtype=model.dtype)
    eq = np.asarray(model.eq_residual(params, y_jax), dtype=np.float64)
    ineq = np.asarray(model.ineq_residual(params, y_jax), dtype=np.float64)

    eq_inf = 0.0 if eq.size == 0 else float(np.max(np.abs(eq)))
    ineq_inf = 0.0 if ineq.size == 0 else float(np.max(np.maximum(ineq, 0.0)))

    lb = _to_numpy(model.lower_bounds(params))
    ub = _to_numpy(model.upper_bounds(params))
    lb_inf = 0.0 if lb is None else float(np.max(np.maximum(lb - y, 0.0)))
    ub_inf = 0.0 if ub is None else float(np.max(np.maximum(y - ub, 0.0)))

    return max(eq_inf, ineq_inf, lb_inf, ub_inf)


def extract_exact_qp(model: JaxNLPModel, params) -> QuadraticProgramData:
    if model.objective_structure != "quadratic":
        raise ValueError("Exact CVXPY extraction requires a quadratic objective.")
    if model.eq_structure not in ("empty", "affine"):
        raise ValueError("Exact CVXPY extraction requires affine equalities.")
    if model.ineq_structure not in ("empty", "affine"):
        raise ValueError("Exact CVXPY extraction requires affine inequalities.")

    y_ref = jnp.asarray(_default_initial_point(model, params), dtype=model.dtype)
    Q = _to_numpy(model.hess_y_objective(params, y_ref))
    grad = _to_numpy(model.grad_y_objective(params, y_ref))
    A = _to_numpy(model.jac_y_eq(params, y_ref))
    C = _to_numpy(model.jac_y_ineq(params, y_ref))
    h = _to_numpy(model.eq_residual(params, y_ref))
    g = _to_numpy(model.ineq_residual(params, y_ref))

    b = A @ np.asarray(y_ref, dtype=np.float64) - h
    d = C @ np.asarray(y_ref, dtype=np.float64) - g
    c = grad - Q @ np.asarray(y_ref, dtype=np.float64)

    return QuadraticProgramData(
        Q=Q,
        c=c,
        A=A,
        b=b,
        C=C,
        d=d,
        l=_to_numpy(model.lower_bounds(params)),
        u=_to_numpy(model.upper_bounds(params)),
    )


def explain_cvxpy_support(model: JaxNLPModel) -> str:
    reasons = []
    if model.objective_structure != "quadratic":
        reasons.append("objective is not stored as an exact CVXPY-translatable quadratic form")
    for entry in model.constraints:
        meta = entry.metadata or {}
        if entry.structure == "affine" and meta.get("type") == "affine_block":
            continue
        if entry.structure == "quadratic" and meta.get("type") == "quadratic_scalar":
            continue
        reasons.append(f"constraint '{entry.name}' is not directly representable from stored block metadata")

    if not reasons:
        return "Model can be solved directly by CVXPY."
    return (
        "Model cannot be solved directly by solgen/CVXPY: "
        + "; ".join(reasons)
        + ". Current JaxNLPModel nonlinear callables do not preserve symbolic CVXPY expressions."
    )


@dataclass
class SolGenModel:
    model: JaxNLPModel

    def supports_direct_cvxpy(self) -> bool:
        if self.model.objective_structure != "quadratic":
            return False
        for entry in self.model.constraints:
            meta = entry.metadata or {}
            if entry.structure == "affine" and meta.get("type") == "affine_block":
                continue
            if entry.structure == "quadratic" and meta.get("type") == "quadratic_scalar":
                continue
            return False
        return True

    def solve(
        self,
        params,
        *,
        mode: str = "auto",
        solver: Optional[str] = None,
        y0: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> SolveResult:
        chosen_mode = mode
        if chosen_mode == "auto":
            chosen_mode = "direct"

        if chosen_mode != "direct":
            raise ValueError(f"Unknown solve mode '{mode}'")
        if not self.supports_direct_cvxpy():
            raise NotImplementedError(explain_cvxpy_support(self.model))

        y_init = _default_initial_point(self.model, params) if y0 is None else np.asarray(y0, dtype=np.float64)
        status, y, objective, mu, solve_time_sec = solve_jaxmodel_direct(
            self.model,
            params,
            solver=solver,
            warm_start=y_init,
            verbose=verbose,
        )
        residual = _primal_residual(self.model, params, y)
        return SolveResult(
            status=status,
            y=y,
            objective=objective,
            iterations=1,
            mode="direct",
            primal_residual=residual,
            step_norm=0.0,
            mu=mu,
            solve_time_sec=solve_time_sec,
        )
