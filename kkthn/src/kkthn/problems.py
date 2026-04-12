from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

jax.config.update("jax_enable_x64", True)

from scripts.misc.solver_config import resolve_solver_name  # noqa: E402
from solgen import SolGenModel  # noqa: E402


@dataclass(frozen=True)
class ProblemBundle:
    problem_type: str
    model: Any
    X: np.ndarray
    Y: np.ndarray
    metadata: dict[str, Any]
    problem_data: Mapping[str, Any] | None = None


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, sort_keys=True)


def normalize_problem_type(problem_type: str) -> str:
    normalized = str(problem_type).strip().lower()
    if normalized == "nonconvex":
        return "nonconvx"
    if normalized not in {"qp", "qcqp", "nlp", "nonconvx", "general"}:
        raise ValueError("type must be one of qp, qcqp, nlp, nonconvex/nonconvx, or general.")
    return normalized


def case_folder_name(problem_type: str) -> str:
    return "nonconvx" if normalize_problem_type(problem_type) == "nonconvx" else normalize_problem_type(problem_type)


def _infer_parameter_dim(data_cfg: Mapping[str, Any]) -> int | None:
    if "p" in data_cfg:
        return int(data_cfg["p"])
    if "n_x" in data_cfg:
        return int(data_cfg["n_x"])
    return None


def _broadcast_parameter_bounds(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Allow x_L/x_U = [value] as shorthand for all parameter dimensions."""

    data = copy.deepcopy(dict(data_cfg))
    param_dim = _infer_parameter_dim(data)
    if param_dim is None:
        return data
    for key in ("x_L", "x_U"):
        if key not in data:
            continue
        vals = list(data[key])
        if len(vals) == 1 and param_dim != 1:
            data[key] = vals * param_dim
        elif len(vals) != param_dim:
            raise ValueError(f"{key} must have length 1 or {param_dim}; got {len(vals)}.")
    return data


def apply_overrides(
    data_cfg: Mapping[str, Any],
    cfg_dict: Mapping[str, Any],
    *,
    p: int | None = None,
    n: int | None = None,
    me: int | None = None,
    mi: int | None = None,
    samples: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    solver: str | None = None,
    train_frac: float | None = None,
    hidden_size: int | None = None,
    hidden_layers: int | None = None,
    seed: int | None = None,
    noise_scale: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = copy.deepcopy(dict(data_cfg))
    cfg = copy.deepcopy(dict(cfg_dict))

    if p is not None:
        if "p" in data:
            data["p"] = int(p)
        elif "n_x" in data:
            data["n_x"] = int(p)
    if n is not None:
        if "n" in data:
            data["n"] = int(n)
        elif "n_y" in data:
            data["n_y"] = int(n)
    if me is not None:
        if "me" in data:
            data["me"] = int(me)
        elif "n_eq" in data:
            data["n_eq"] = int(me)
    if mi is not None:
        if "mi" in data:
            data["mi"] = int(mi)
        elif "n_ineq" in data:
            data["n_ineq"] = int(mi)

    if samples is not None:
        if "num_samples" in data:
            data["num_samples"] = int(samples)
        if "N_samples" in data:
            data["N_samples"] = int(samples)
        if "N_points" in data:
            data["N_points"] = int(samples)
    if solver is not None:
        data["solver"] = str(solver)
    if seed is not None:
        data["seed"] = int(seed)
        cfg["seed"] = int(seed)
    if noise_scale is not None:
        data["noise_scale"] = float(noise_scale)

    if epochs is not None:
        cfg["epochs"] = int(epochs)
    if batch_size is not None:
        cfg["batch_size"] = int(batch_size)
    if learning_rate is not None:
        cfg["learning_rate"] = float(learning_rate)
    if train_frac is not None:
        cfg["train_frac"] = float(train_frac)
    if hidden_size is not None:
        cfg["hidden_size"] = int(hidden_size)
    if hidden_layers is not None:
        cfg["hidden_layers"] = int(hidden_layers)
        cfg["hidden_dim"] = int(hidden_layers)
    return _broadcast_parameter_bounds(data), cfg


def _sample_box(data_cfg: Mapping[str, Any], *, count: int, dim_key: str, seed_offset: int = 0) -> np.ndarray:
    n_x = int(data_cfg[dim_key])
    x_l = np.asarray(data_cfg["x_L"], dtype=np.float64)
    x_u = np.asarray(data_cfg["x_U"], dtype=np.float64)
    if x_l.shape == (1,):
        x_l = np.repeat(x_l, n_x)
    if x_u.shape == (1,):
        x_u = np.repeat(x_u, n_x)
    if x_l.shape != (n_x,) or x_u.shape != (n_x,):
        raise ValueError(f"x_L/x_U must have shape ({n_x},).")
    rng = np.random.default_rng(int(data_cfg["seed"]) + int(seed_offset))
    return rng.uniform(x_l, x_u, size=(int(count), n_x)).astype(np.float64)


def apply_label_noise(Y: np.ndarray, data_cfg: Mapping[str, Any], metadata: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    scale = float(data_cfg.get("noise_scale", 0.0))
    meta = dict(metadata or {})
    meta["noise_scale"] = scale
    meta["label_noise_distribution"] = "normal_0_mean_1_variance_scaled" if scale != 0.0 else "none"
    if scale == 0.0:
        meta["label_noise_seed"] = None
        meta["label_noise_empirical_std"] = 0.0
        return np.asarray(Y, dtype=np.float64), meta

    seed = int(data_cfg.get("seed", 0)) + int(data_cfg.get("noise_seed_offset", 7919))
    rng = np.random.default_rng(seed)
    noise = scale * rng.normal(loc=0.0, scale=1.0, size=np.asarray(Y).shape)
    meta["label_noise_seed"] = seed
    meta["label_noise_empirical_mean"] = float(np.mean(noise))
    meta["label_noise_empirical_std"] = float(np.std(noise))
    return np.asarray(Y, dtype=np.float64) + noise.astype(np.float64), meta


def _solve_with_solgen(model, X: np.ndarray, data_cfg: Mapping[str, Any], *, n_ineq: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    solver_name = resolve_solver_name(data_cfg, default="SCS")
    solver = SolGenModel(model)
    ys: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    statuses: dict[str, int] = {}
    fallback_count = 0
    solve_time = 0.0
    t0 = time.perf_counter()
    for x in X:
        try:
            result = solver.solve({"x": jnp.asarray(x, dtype=model.dtype)}, solver=solver_name)
        except Exception:
            if solver_name == "SCS":
                raise
            fallback_count += 1
            result = solver.solve({"x": jnp.asarray(x, dtype=model.dtype)}, solver="SCS")
        statuses[str(result.status)] = statuses.get(str(result.status), 0) + 1
        ys.append(np.asarray(result.y, dtype=np.float64))
        mu = np.asarray(result.mu, dtype=np.float64) if result.mu is not None else np.zeros((n_ineq,), dtype=np.float64)
        if mu.size != n_ineq:
            mu = np.resize(mu, (n_ineq,)).astype(np.float64)
        mus.append(mu)
        solve_time += float(result.solve_time_sec or 0.0)
    metadata = {
        "label_source": "optimizer",
        "solver": solver_name,
        "solver_fallback_to_scs_count": fallback_count,
        "status_counts": statuses,
        "optimizer_generation_wall_time_sec": solve_time,
        "label_generation_total_wall_time_sec": time.perf_counter() - t0,
    }
    return np.asarray(ys, dtype=np.float64), np.asarray(mus, dtype=np.float64), metadata


def _solve_with_generator(generator, data_cfg: Mapping[str, Any], *, target: int, oversample: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    kept_mu: list[np.ndarray] = []
    statuses: dict[str, int] = {}
    solver_counts: dict[str, int] = {}
    objectives: list[float] = []
    solve_time = 0.0
    n_ineq = int(getattr(generator, "n_ineq", data_cfg.get("n_ineq", data_cfg.get("mi", 0))))
    t0 = time.perf_counter()
    attempts = 0
    max_attempts = max(int(target) * 5, int(target) + 64)

    while len(kept_x) < int(target) and attempts < max_attempts:
        batch_size = int(oversample or max(16, int(target) - len(kept_x)))
        for x in generator.sample_parameters(batch_size):
            attempts += 1
            result = generator.solve_for_x(np.asarray(x, dtype=np.float64))
            status = str(result.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
            solver_used = result.get("solver")
            if solver_used is not None:
                solver_name = str(solver_used)
                solver_counts[solver_name] = solver_counts.get(solver_name, 0) + 1
            solve_time += float(result.get("solve_time_sec") or 0.0)
            if status in {"optimal", "optimal_inaccurate"} and result.get("y") is not None:
                kept_x.append(np.asarray(x, dtype=np.float64))
                kept_y.append(np.asarray(result["y"], dtype=np.float64))
                mu = result.get("mu")
                mu_arr = np.zeros((n_ineq,), dtype=np.float64) if mu is None else np.asarray(mu, dtype=np.float64).reshape(-1)
                if mu_arr.size != n_ineq:
                    mu_arr = np.resize(mu_arr, (n_ineq,)).astype(np.float64)
                kept_mu.append(mu_arr)
                if result.get("objective") is not None:
                    objectives.append(float(result["objective"]))
            if len(kept_x) >= int(target) or attempts >= max_attempts:
                break

    if len(kept_x) < int(target):
        raise RuntimeError(f"Only collected {len(kept_x)} successful labels out of requested {target}.")

    metadata = {
        "label_source": "optimizer",
        "status_counts": statuses,
        "solver": next(iter(solver_counts)) if len(solver_counts) == 1 else None,
        "solver_counts": solver_counts,
        "attempts": attempts,
        "objective_min": float(np.nanmin(objectives)) if objectives else None,
        "objective_max": float(np.nanmax(objectives)) if objectives else None,
        "objective_mean": float(np.nanmean(objectives)) if objectives else None,
        "optimizer_generation_wall_time_sec": solve_time,
        "label_generation_total_wall_time_sec": time.perf_counter() - t0,
    }
    return (
        np.asarray(kept_x, dtype=np.float64),
        np.asarray(kept_y, dtype=np.float64),
        np.asarray(kept_mu, dtype=np.float64),
        metadata,
    )


def _bounds_for_model(model, params) -> tuple[np.ndarray, np.ndarray]:
    n_y = int(model.var_spec.total_size)
    lb = model.lower_bounds(params)
    ub = model.upper_bounds(params)
    lower = -np.inf * np.ones((n_y,), dtype=np.float64) if lb is None else np.asarray(lb, dtype=np.float64).reshape(n_y)
    upper = np.inf * np.ones((n_y,), dtype=np.float64) if ub is None else np.asarray(ub, dtype=np.float64).reshape(n_y)
    return lower, upper


def solve_jaxmodel_slsqp(model, x: np.ndarray, *, param_name: str = "x", maxiter: int = 1000) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    params = {param_name: jnp.asarray(x, dtype=model.dtype)}
    n_y = int(model.var_spec.total_size)
    lower, upper = _bounds_for_model(model, params)
    finite_mid = np.where(
        np.isfinite(lower) & np.isfinite(upper),
        0.5 * (lower + upper),
        np.where(np.isfinite(lower), lower + 0.1, np.where(np.isfinite(upper), upper - 0.1, 0.0)),
    )
    y0 = np.asarray(finite_mid, dtype=np.float64).reshape(n_y)

    def obj_np(y_np):
        y_j = jnp.asarray(y_np, dtype=model.dtype)
        return float(model.objective_value(params, y_j))

    def grad_np(y_np):
        y_j = jnp.asarray(y_np, dtype=model.dtype)
        return np.asarray(model.grad_y_objective(params, y_j), dtype=np.float64)

    constraints = []
    y_zero = jnp.zeros((n_y,), dtype=model.dtype)
    n_eq = int(model.eq_residual(params, y_zero).shape[0])
    n_ineq = int(model.ineq_residual(params, y_zero).shape[0])
    if n_eq > 0:
        constraints.append(
            NonlinearConstraint(
                lambda y: np.asarray(model.eq_residual(params, jnp.asarray(y, dtype=model.dtype)), dtype=np.float64),
                np.zeros((n_eq,), dtype=np.float64),
                np.zeros((n_eq,), dtype=np.float64),
                jac=lambda y: np.asarray(model.jac_y_eq(params, jnp.asarray(y, dtype=model.dtype)), dtype=np.float64),
            )
        )
    if n_ineq > 0:
        constraints.append(
            NonlinearConstraint(
                lambda y: np.asarray(model.ineq_residual(params, jnp.asarray(y, dtype=model.dtype)), dtype=np.float64),
                -np.inf * np.ones((n_ineq,), dtype=np.float64),
                np.zeros((n_ineq,), dtype=np.float64),
                jac=lambda y: np.asarray(model.jac_y_ineq(params, jnp.asarray(y, dtype=model.dtype)), dtype=np.float64),
            )
        )

    result = minimize(
        obj_np,
        y0,
        jac=grad_np,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": int(maxiter), "disp": False},
    )
    status = "optimal" if bool(result.success) else "solver_failed"
    return {
        "status": status,
        "y": np.asarray(result.x, dtype=np.float64),
        "objective": float(result.fun) if np.isfinite(result.fun) else None,
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
    }


def generate_labels_with_slsqp(
    model,
    X: np.ndarray,
    *,
    param_name: str = "x",
    accept_failed_if_feasible: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    ys: list[np.ndarray] = []
    statuses: dict[str, int] = {}
    objectives: list[float] = []
    t0 = time.perf_counter()
    for x in np.asarray(X, dtype=np.float64):
        result = solve_jaxmodel_slsqp(model, x, param_name=param_name)
        status = str(result["status"])
        statuses[status] = statuses.get(status, 0) + 1
        y = np.asarray(result["y"], dtype=np.float64)
        if status != "optimal" and not accept_failed_if_feasible:
            raise RuntimeError(f"SLSQP failed for x={x}: {result.get('message')}")
        ys.append(y)
        if result.get("objective") is not None:
            objectives.append(float(result["objective"]))
    metadata = {
        "label_source": "synthetic_slsqp",
        "solver": "scipy_slsqp",
        "status_counts": statuses,
        "objective_min": float(np.nanmin(objectives)) if objectives else None,
        "objective_max": float(np.nanmax(objectives)) if objectives else None,
        "objective_mean": float(np.nanmean(objectives)) if objectives else None,
        "label_generation_total_wall_time_sec": time.perf_counter() - t0,
    }
    return np.asarray(ys, dtype=np.float64), metadata


def _build_poly(problem_type: str, data_cfg: Mapping[str, Any]) -> ProblemBundle:
    from scripts.factory.poly_factory import build_problem_data, build_problem_model_from_data

    problem_data = build_problem_data(data_cfg)
    model = build_problem_model_from_data(problem_data, dtype=jnp.float64)
    X = _sample_box(data_cfg, count=int(data_cfg["num_samples"]), dim_key="p")
    Y, _Mu, meta = _solve_with_solgen(model, X, data_cfg, n_ineq=int(data_cfg["mi"]))
    meta.update({"problem_type": problem_type, "n_x": int(data_cfg["p"]), "n_y": int(data_cfg["n"])})
    Y, meta = apply_label_noise(Y, data_cfg, meta)
    return ProblemBundle(problem_type=problem_type, model=model, X=X, Y=Y, metadata=meta, problem_data=problem_data)


def _build_nlp(data_cfg: Mapping[str, Any]) -> ProblemBundle:
    from scripts.factory.nlp_factory import build_problem_generator, build_problem_model_from_data

    generator = build_problem_generator(data_cfg)
    problem_data = generator.get_problem_data()
    model = build_problem_model_from_data(problem_data, dtype=jnp.float64)
    target = int(data_cfg.get("N_points", data_cfg.get("N_samples", 0)))
    X, Y, _Mu, meta = _solve_with_generator(generator, data_cfg, target=target)
    meta.update({"problem_type": "nlp", "n_x": int(data_cfg["n_x"]), "n_y": int(data_cfg["n_y"])})
    Y, meta = apply_label_noise(Y, data_cfg, meta)
    return ProblemBundle(problem_type="nlp", model=model, X=X, Y=Y, metadata=meta, problem_data=problem_data)


def _build_nonconvx(data_cfg: Mapping[str, Any]) -> ProblemBundle:
    from scripts.factory.nonconvx_factory import build_problem_generator, build_problem_model_from_data

    local = copy.deepcopy(dict(data_cfg))
    if "n_y" not in local:
        local["n_y"] = int(local["n"])
        local["n_eq"] = int(local["me"])
        local["n_ineq"] = int(local["mi"])
    if "p" in local and int(local["p"]) != int(local["n_eq"]):
        raise ValueError("nonconvex requires p == me/n_eq in the DC3-style family.")
    generator = build_problem_generator(local)
    problem_data = generator.get_problem_data()
    model = build_problem_model_from_data(problem_data, dtype=jnp.float64)
    X, Y, _Mu, meta = _solve_with_generator(generator, local, target=int(local["num_samples"]))
    meta.update({"problem_type": "nonconvex", "n_x": int(generator.n_x), "n_y": int(generator.n_y)})
    Y, meta = apply_label_noise(Y, local, meta)
    return ProblemBundle(problem_type="nonconvx", model=model, X=X, Y=Y, metadata=meta, problem_data=problem_data)


def load_model_definition(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Expected model definition at {path}")
    module_name = f"_kkthn_general_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model definition from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_general_from_model_definition(case_dir: Path, data_cfg: Mapping[str, Any]) -> ProblemBundle:
    module = load_model_definition(Path(case_dir) / "model_definition.py")
    if not hasattr(module, "build_model"):
        raise AttributeError("case/general/model_definition.py must define build_model(...).")
    model = module.build_model(dtype=jnp.float64)
    n_x = int(np.asarray(data_cfg["x_L"], dtype=np.float64).size)
    X = _sample_box({"x_L": data_cfg["x_L"], "x_U": data_cfg["x_U"], "seed": data_cfg["seed"], "n_x": n_x}, count=int(data_cfg["num_samples"]), dim_key="n_x")
    Y, meta = generate_labels_with_slsqp(model, X)
    y0 = jnp.zeros((model.var_spec.total_size,), dtype=model.dtype)
    x0 = jnp.zeros((n_x,), dtype=model.dtype)
    dims = {
        "n_x": n_x,
        "n_y": int(model.var_spec.total_size),
        "n_eq": int(model.eq_residual({"x": x0}, y0).shape[0]),
        "n_ineq": int(model.ineq_residual({"x": x0}, y0).shape[0]),
    }
    meta.update({"problem_type": "general", **dims, "model_definition": str(Path(case_dir) / "model_definition.py")})
    Y, meta = apply_label_noise(Y, data_cfg, meta)
    return ProblemBundle(problem_type="general", model=model, X=X, Y=Y, metadata=meta, problem_data=dims)


def build_problem_bundle(problem_type: str, case_dir: Path, data_cfg: Mapping[str, Any]) -> ProblemBundle:
    normalized = normalize_problem_type(problem_type)
    data_cfg = _broadcast_parameter_bounds(data_cfg)
    if normalized in {"qp", "qcqp"}:
        return _build_poly(normalized, data_cfg)
    if normalized == "nlp":
        return _build_nlp(data_cfg)
    if normalized == "nonconvx":
        return _build_nonconvx(data_cfg)
    return build_general_from_model_definition(case_dir, data_cfg)
