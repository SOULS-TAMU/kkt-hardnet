from __future__ import annotations

from typing import Any, Mapping
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from jaxmodel import HighLevelNLPBuilder
from scripts.misc.inequality_multipliers import default_ineq_multipliers

jax.config.update("jax_enable_x64", True)

DATASET_KIND = "nonconvex"


def normalize_problem_type(problem_type: str) -> str:
    normalized = str(problem_type).strip().lower()
    if normalized != DATASET_KIND:
        raise ValueError(f"case/nonconvx only supports type='{DATASET_KIND}', got '{problem_type}'.")
    return normalized


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class DC3StyleNonconvexGenerator:
    def __init__(
        self,
        *,
        n_y: int,
        n_eq: int,
        n_ineq: int,
        num_samples: int,
        seed: int,
        is_diag_Q: bool,
    ) -> None:
        self.n_y = int(n_y)
        self.n_x = int(n_eq)
        self.n_eq = int(n_eq)
        self.n_ineq = int(n_ineq)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.is_diag_Q = bool(is_diag_Q)
        self.requested_solver = "ipopt" if (_has_module("ipopt") or _has_module("cyipopt")) else "scipy_slsqp"
        self._rng = np.random.default_rng(self.seed)
        self._param_rng = np.random.default_rng(self.seed + 1)
        self._build_problem_data()

    def _build_problem_data(self) -> None:
        if self.is_diag_Q:
            Q = np.diag(self._rng.random(self.n_y))
        else:
            mat = self._rng.normal(loc=0.0, scale=1.0, size=(self.n_y, self.n_y))
            Q = (mat.T @ mat) / max(1, self.n_y)
            diag_boost = 0.1 + self._rng.random(self.n_y)
            Q = 0.5 * (Q + Q.T) + np.diag(diag_boost)

        p = self._rng.random(self.n_y)
        A = self._rng.normal(loc=0.0, scale=1.0, size=(self.n_eq, self.n_y))
        G = self._rng.normal(loc=0.0, scale=1.0, size=(self.n_ineq, self.n_y))
        h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1) if self.n_ineq > 0 else np.zeros((0,), dtype=np.float64)

        self.Q = np.asarray(Q, dtype=np.float64)
        self.p = np.asarray(p, dtype=np.float64)
        self.A = np.asarray(A, dtype=np.float64)
        self.G = np.asarray(G, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)

    def sample_parameters(self, count: int) -> np.ndarray:
        return self._param_rng.uniform(-1.0, 1.0, size=(int(count), self.n_x)).astype(np.float64)

    def get_problem_data(self) -> dict[str, np.ndarray]:
        return {
            "Q": np.asarray(self.Q, dtype=np.float64),
            "p": np.asarray(self.p, dtype=np.float64),
            "A": np.asarray(self.A, dtype=np.float64),
            "G": np.asarray(self.G, dtype=np.float64),
            "h": np.asarray(self.h, dtype=np.float64),
            "n_x": np.asarray(self.n_x, dtype=np.int64),
            "n_y": np.asarray(self.n_y, dtype=np.int64),
            "n_eq": np.asarray(self.n_eq, dtype=np.int64),
            "n_ineq": np.asarray(self.n_ineq, dtype=np.int64),
            "is_diag_Q": np.asarray(int(self.is_diag_Q), dtype=np.int64),
        }

    def solve_for_x(self, x: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x, dtype=np.float64)
        y0 = np.linalg.pinv(self.A) @ x
        result = self._solve_slsqp(x, y0)
        status = "optimal" if bool(result.success) else "solver_failed"
        y = np.asarray(result.x, dtype=np.float64) if bool(result.success) else None
        objective = float(result.fun) if bool(result.success) else None
        return {
            "status": status,
            "y": y,
            "objective": objective,
            "mu": self._extract_ineq_multipliers(result),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "solver": self.requested_solver,
        }

    def _extract_ineq_multipliers(self, result) -> np.ndarray:
        if self.n_ineq <= 0:
            return np.zeros((0,), dtype=np.float64)
        raw = getattr(result, "multipliers", None)
        if raw is None:
            return default_ineq_multipliers(self.n_ineq)
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        expected = self.n_eq + self.n_ineq
        if arr.size < expected:
            return default_ineq_multipliers(self.n_ineq)
        out = arr[self.n_eq:self.n_eq + self.n_ineq].astype(np.float64, copy=True)
        finite = np.isfinite(out)
        out[finite] = np.maximum(out[finite], 0.0)
        return out

    def _solve_slsqp(self, x: np.ndarray, y0: np.ndarray):
        Q = self.Q
        p = self.p
        A = self.A
        G = self.G
        h = self.h

        def objective(y: np.ndarray) -> float:
            return float(0.5 * y @ Q @ y + p @ np.sin(y))

        def gradient(y: np.ndarray) -> np.ndarray:
            return Q @ y + p * np.cos(y)

        constraints: list[LinearConstraint] = [LinearConstraint(A, x, x)]
        if G.shape[0] > 0:
            constraints.append(LinearConstraint(G, -np.inf * np.ones_like(h), h))

        return minimize(
            objective,
            np.asarray(y0, dtype=np.float64),
            jac=gradient,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
        )


def _build_generator_from_any_cfg(data_cfg: Mapping[str, Any]) -> DC3StyleNonconvexGenerator:
    if "n_y" in data_cfg:
        normalize_problem_type(str(data_cfg["type"]))
        n_y = int(data_cfg["n_y"])
        n_eq = int(data_cfg["n_eq"])
        n_ineq = int(data_cfg["n_ineq"])
        num_samples = int(data_cfg["num_samples"])
        seed = int(data_cfg["seed"])
        is_diag_q = bool(data_cfg.get("is_diag_Q", True))
    else:
        n_y = int(data_cfg["n"])
        n_eq = int(data_cfg["me"])
        n_ineq = int(data_cfg["mi"])
        num_samples = int(data_cfg["num_samples"])
        seed = int(data_cfg["seed"])
        is_diag_q = bool(data_cfg.get("is_diag_Q", True))
    return DC3StyleNonconvexGenerator(
        n_y=n_y,
        n_eq=n_eq,
        n_ineq=n_ineq,
        num_samples=num_samples,
        seed=seed,
        is_diag_Q=is_diag_q,
    )


def build_problem_generator(data_cfg: Mapping[str, Any]) -> DC3StyleNonconvexGenerator:
    return _build_generator_from_any_cfg(data_cfg)


def build_problem_model_from_data(problem_data: Mapping[str, Any], *, dtype=jnp.float64):
    Q = jnp.asarray(problem_data["Q"], dtype=dtype)
    p = jnp.asarray(problem_data["p"], dtype=dtype)
    A = jnp.asarray(problem_data["A"], dtype=dtype)
    G = jnp.asarray(problem_data["G"], dtype=dtype)
    h = jnp.asarray(problem_data["h"], dtype=dtype)
    n_y = int(np.asarray(problem_data["n_y"]).item())
    n_eq = int(np.asarray(problem_data["n_eq"]).item())
    bound_radius = float(np.asarray(problem_data.get("bound_radius", 10.0)).item()) if "bound_radius" in problem_data else 10.0

    builder = (
        HighLevelNLPBuilder(dtype=dtype)
        .add_variable("y", n_y)
        .add_parameter("x", n_eq)
    )

    def objective(_params, vars_dict):
        y = vars_dict["y"]
        return 0.5 * y @ Q @ y + p @ jnp.sin(y)

    builder = builder.set_objective(objective)
    builder = builder.add_affine_equality(
        var_name="y",
        A=A,
        rhs_const=jnp.zeros((n_eq,), dtype=dtype),
        param_terms=[(jnp.eye(n_eq, dtype=dtype), "x")],
        name="dc3style_affine_eq",
    )
    if G.shape[0] > 0:
        builder = builder.add_affine_inequality(
            var_name="y",
            C=G,
            rhs_const=h,
            param_terms=[],
            name="dc3style_affine_ineq",
        )
    builder = builder.set_affine_lower_bound(
        var_name="y",
        param_name="x",
        M=jnp.zeros((n_y, n_eq), dtype=dtype),
        c=-bound_radius * jnp.ones((n_y,), dtype=dtype),
    )
    builder = builder.set_affine_upper_bound(
        var_name="y",
        param_name="x",
        M=jnp.zeros((n_y, n_eq), dtype=dtype),
        c=bound_radius * jnp.ones((n_y,), dtype=dtype),
    )
    return builder.build(example_params={"x": jnp.zeros((n_eq,), dtype=dtype)}, jit_compile=True)


def build_problem_model(data_cfg: Mapping[str, Any], *, dtype=jnp.float64):
    return build_problem_model_from_data(build_problem_generator(data_cfg).get_problem_data(), dtype=dtype)
