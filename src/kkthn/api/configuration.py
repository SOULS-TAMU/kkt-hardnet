"""Training and projection configuration normalization."""

from __future__ import annotations
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Mapping
import numpy as np
from purin.solvers import ADMMConfig
from purin.solvers import ChambollePockConfig
from purin.solvers import FischerBurmeisterConfig
from purin.solvers import ImplicitGradientConfig
from purin.solvers import LinearEqualityConfig
from purin.solvers import LinearSolverConfig
from purin.solvers import ProjectionConfig
from purin.solvers import ProjectionMethod
from purin.execution.runtime import resolve_jax_dtype

_dtype = resolve_jax_dtype


def _dataclass_keywords(cls, values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {name: value for name, value in values.items() if name in allowed}


def _normalize_mapping(config) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("Configuration must be a mapping or dataclass instance.")


def _normalize_bool_option(name: str, value: Any) -> bool:
    """Normalize a strict Boolean training option.

    Strings are accepted for configuration-file compatibility, but ambiguous
    truthy values such as nonempty arbitrary strings are rejected.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise TypeError(f"{name} must be a Boolean value; received {value!r}.")


def _normalize_train_config(config) -> dict[str, Any]:
    values = {} if config is None else dict(config)
    values.setdefault("epochs", 1200)
    values.setdefault("batch_size", 32)
    values.setdefault("learning_rate", 1e-3)
    values.setdefault("train_frac", 0.8)
    values.setdefault("seed", 42)
    values.setdefault("dtype", "float64")
    values.setdefault("print_every", 1)
    values.setdefault("standardize_inputs", True)
    values.setdefault("drop_last_batch", False)
    values.setdefault("pretrain", True)
    values.setdefault("output_dir", "runs")
    values.setdefault("epoch_mlp", max(1, min(100, int(values["epochs"]) // 4)))
    values.setdefault("eta", None)
    values.setdefault("minimum_projection_convergence", 0.95)
    values.setdefault("readiness_extra_epochs", 500)
    values.setdefault("readiness_check_every", 10)
    values.setdefault("dual_pretrain_gamma", 1e-3)
    gamma = float(values["dual_pretrain_gamma"])
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("dual_pretrain_gamma must be finite and strictly positive.")
    values["dual_pretrain_gamma"] = gamma
    # Discard the removed zero-then-predicted Fischer--Burmeister controls
    # when loading older metadata or receiving stale dictionaries.
    for legacy_name in (
        "fb_dual_warm_start",
        "fb_dual_warm_start_min_epochs",
        "fb_dual_warm_start_check_every",
        "fb_dual_warm_start_convergence",
    ):
        values.pop(legacy_name, None)
    values["drop_last_batch"] = _normalize_bool_option(
        "drop_last_batch", values["drop_last_batch"]
    )
    values["pretrain"] = _normalize_bool_option("pretrain", values["pretrain"])
    values.setdefault("backend", "cpu")
    return values


def _projection_definition(self, method: str, config):
    """Translate KKTHN projection configuration into the selected PURIN definition."""
    selected = ProjectionMethod.normalize(method)
    if config is not None and not isinstance(config, Mapping):
        return config
    values = {} if config is None else dict(config)
    values.pop("method", None)
    values.pop("projection_method", None)
    linear_values = values.pop("linear_solver", {})
    implicit_values = values.pop("implicit_gradient", {})
    linear_solver = (
        linear_values
        if isinstance(linear_values, LinearSolverConfig)
        else LinearSolverConfig(
            **_dataclass_keywords(LinearSolverConfig, dict(linear_values))
        )
    )
    implicit_gradient = (
        implicit_values
        if isinstance(implicit_values, ImplicitGradientConfig)
        else ImplicitGradientConfig(
            **_dataclass_keywords(ImplicitGradientConfig, dict(implicit_values))
        )
    )
    if selected is ProjectionMethod.ADMM:
        values["linear_solver"] = linear_solver
        values["implicit_gradient"] = implicit_gradient
        return ADMMConfig(**_dataclass_keywords(ADMMConfig, values))
    if selected is ProjectionMethod.CHAMBOLLE_POCK:
        values["linear_solver"] = linear_solver
        values["implicit_gradient"] = implicit_gradient
        return ChambollePockConfig(**_dataclass_keywords(ChambollePockConfig, values))
    if selected in {
        ProjectionMethod.FISCHER_BURMEISTER_NEWTON,
        ProjectionMethod.FISCHER_BURMEISTER_REDUCED,
    }:
        aliases = {
            "fb_eps": "smoothing",
            "gn_max_iters": "max_iterations",
            "max_newton_iter": "max_iterations",
            "gn_tol": "tolerance",
            "newton_tol": "tolerance",
            "gn_reg": "initial_regularization",
            "newton_reg_factor": "initial_regularization",
            "newton_step_length": "initial_step_size",
            "armijo_alpha": "armijo_constant",
            "armijo_beta": "backtracking_factor",
            "max_backtrack_iter": "maximum_backtracking_steps",
            "armijo_max_steps": "maximum_backtracking_steps",
        }
        for old_name, new_name in aliases.items():
            if old_name in values and new_name not in values:
                values[new_name] = values.pop(old_name)
        values["linear_solver"] = linear_solver
        values["implicit_gradient"] = implicit_gradient
        return FischerBurmeisterConfig(
            **_dataclass_keywords(FischerBurmeisterConfig, values)
        )
    if selected is ProjectionMethod.PRIMAL_DUAL_INTERIOR_POINT:
        values["linear_solver"] = linear_solver
        values["implicit_gradient"] = implicit_gradient
        return ProjectionConfig(**_dataclass_keywords(ProjectionConfig, values))
    return LinearEqualityConfig(**_dataclass_keywords(LinearEqualityConfig, values))
