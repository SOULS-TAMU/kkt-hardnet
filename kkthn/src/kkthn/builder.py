from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import jax.numpy as jnp
import numpy as np

from jaxmodel import HighLevelNLPBuilder

from .backbone import make_batched_mlp_apply
from .projection import ProjectionSettings, make_projection_layer
from .string_problem import build_model_from_string_problem
from .training import KKTTrainConfig, load_model_weights, train_kkt_hardnet


def _coerce_names(names: str | Iterable[str], *, allow_empty: bool = False) -> list[str]:
    if isinstance(names, str):
        out = [names]
    else:
        out = [str(name) for name in names]
    if not out and not allow_empty:
        raise ValueError("At least one name is required.")
    if len(set(out)) != len(out):
        raise ValueError(f"Names must be unique, got {out}.")
    return out


def _dtype(name: str):
    normalized = str(name).lower()
    if normalized in {"float32", "fp32", "32"}:
        return jnp.float32
    if normalized in {"float64", "fp64", "64"}:
        return jnp.float64
    raise ValueError(f"Unsupported dtype '{name}'.")


def _json_safe(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _as_expr(value) -> "Expression":
    if isinstance(value, Expression):
        return value
    return Expression(lambda _ctx, v=float(value): jnp.asarray(v), repr(float(value)))


@dataclass(frozen=True)
class Constraint:
    residual: "Expression"
    kind: str
    text: str


@dataclass(frozen=True)
class DatasetSpec:
    parameters_path: str
    variables_path: str | None = None


@dataclass(frozen=True)
class _EvalContext:
    y: jnp.ndarray
    x: jnp.ndarray
    variable_index: dict[str, int]
    parameter_index: dict[str, int]
    inverse_index: dict[str, int]
    inverse_values: jnp.ndarray
    train_inverse: bool
    forward_dim: int

    def value(self, kind: str, name: str):
        if kind == "variable":
            return self.y[self.variable_index[name]]
        if kind == "parameter":
            return self.x[self.parameter_index[name]]
        if kind == "inverse_parameter":
            idx = self.inverse_index[name]
            if self.train_inverse:
                return self.x[self.forward_dim + idx]
            return self.inverse_values[idx]
        raise KeyError(f"Unknown symbol kind '{kind}'.")


@dataclass
class _InferenceState:
    task: str
    model: Any
    params: Any
    projection: Any
    data_n_x: int
    output_dir: str | None


class Expression:
    def __init__(self, fn: Callable, text: str) -> None:
        self._fn = fn
        self.text = str(text)

    def eval(self, ctx):
        return self._fn(ctx)

    def _binary(self, other, op, symbol: str) -> "Expression":
        rhs = _as_expr(other)
        return Expression(lambda ctx, lhs=self, rhs=rhs: op(lhs.eval(ctx), rhs.eval(ctx)), f"({self.text} {symbol} {rhs.text})")

    def _rbinary(self, other, op, symbol: str) -> "Expression":
        lhs = _as_expr(other)
        return Expression(lambda ctx, lhs=lhs, rhs=self: op(lhs.eval(ctx), rhs.eval(ctx)), f"({lhs.text} {symbol} {self.text})")

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b, "+")

    def __radd__(self, other):
        return self._rbinary(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b, "-")

    def __rsub__(self, other):
        return self._rbinary(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b, "*")

    def __rmul__(self, other):
        return self._rbinary(other, lambda a, b: a * b, "*")

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b, "/")

    def __rtruediv__(self, other):
        return self._rbinary(other, lambda a, b: a / b, "/")

    def __pow__(self, other):
        return self._binary(other, lambda a, b: a**b, "**")

    def __rpow__(self, other):
        return self._rbinary(other, lambda a, b: a**b, "**")

    def __neg__(self):
        return Expression(lambda ctx, expr=self: -expr.eval(ctx), f"(-{self.text})")

    def __le__(self, other):
        rhs = _as_expr(other)
        return Constraint(self - rhs, "ineq", f"{self.text} <= {rhs.text}")

    def __ge__(self, other):
        rhs = _as_expr(other)
        return Constraint(rhs - self, "ineq", f"{self.text} >= {rhs.text}")

    def __eq__(self, other):  # type: ignore[override]
        rhs = _as_expr(other)
        return Constraint(self - rhs, "eq", f"{self.text} == {rhs.text}")


class _SymbolNamespace:
    def __init__(self, builder: "KKTHardNet", kind: str) -> None:
        self._builder = builder
        self._kind = kind

    def __getattr__(self, name: str) -> Expression:
        return self[name]

    def __getitem__(self, name: str) -> Expression:
        self._builder._check_symbol(self._kind, name)
        return Expression(lambda ctx, kind=self._kind, n=name: ctx.value(kind, n), name)


class ConstraintList:
    def __init__(self) -> None:
        self.items: list[Constraint] = []

    def add(self, *constraints: Constraint) -> "ConstraintList":
        for constraint in constraints:
            if not isinstance(constraint, Constraint):
                raise TypeError("constraints.add(...) expects expressions created with ==, <=, or >=.")
            self.items.append(constraint)
        return self

    def equality(self, expr, *, name: str | None = None) -> "ConstraintList":
        residual = _as_expr(expr)
        self.items.append(Constraint(residual, "eq", name or f"{residual.text} == 0"))
        return self

    def inequality(self, expr, *, name: str | None = None) -> "ConstraintList":
        residual = _as_expr(expr)
        self.items.append(Constraint(residual, "ineq", name or f"{residual.text} <= 0"))
        return self


class KKTHardNet:
    def __init__(
        self,
        name: str = "kkthardnet",
        *,
        train: dict[str, Any] | KKTTrainConfig | None = None,
        projection: dict[str, Any] | ProjectionSettings | None = None,
    ) -> None:
        self.name = str(name)
        self.parameter_names: list[str] = []
        self.variable_names: list[str] = []
        self.inverse_parameter_names: list[str] = []
        self._inverse_initial_values: list[float] = []
        self.parameter = _SymbolNamespace(self, "parameter")
        self.variable = _SymbolNamespace(self, "variable")
        self.inverse_parameter = _SymbolNamespace(self, "inverse_parameter")
        self.parameters = self.parameter
        self.variables = self.variable
        self.inverse_parameters = self.inverse_parameter
        self.objective: Expression | None = None
        self.constraints = ConstraintList()
        self.dataset_spec: DatasetSpec | None = None
        self._train_config = self._normalize_train_config(train)
        self._projection_config = self._normalize_projection_config(projection)
        self._inference_state: _InferenceState | None = None
        self._metadata_path: str | None = None

    def add_parameter(self, names: str | Iterable[str]):
        for name in _coerce_names(names):
            if name in self.parameter_names or name in self.variable_names or name in self.inverse_parameter_names:
                raise ValueError(f"Duplicate symbol name '{name}'.")
            self.parameter_names.append(name)
        return self.parameter

    def add_variable(self, names: str | Iterable[str]):
        for name in _coerce_names(names):
            if name in self.parameter_names or name in self.variable_names or name in self.inverse_parameter_names:
                raise ValueError(f"Duplicate symbol name '{name}'.")
            self.variable_names.append(name)
        return self.variable

    def add_inverse_parameter(
        self,
        names: str | Iterable[str],
        *,
        init_value: float | Iterable[float] | None = None,
    ):
        new_names = _coerce_names(names, allow_empty=True)
        if init_value is None:
            init_values = [0.0] * len(new_names)
        else:
            init_arr = np.asarray(init_value, dtype=np.float64).reshape(-1)
            if init_arr.size == 1 and len(new_names) > 1:
                init_arr = np.full((len(new_names),), float(init_arr[0]), dtype=np.float64)
            if init_arr.size != len(new_names):
                raise ValueError("init_value must be scalar or match the number of inverse parameters.")
            init_values = init_arr.astype(np.float64).tolist()
        for name, value in zip(new_names, init_values):
            if name in self.parameter_names or name in self.variable_names or name in self.inverse_parameter_names:
                raise ValueError(f"Duplicate symbol name '{name}'.")
            self.inverse_parameter_names.append(name)
            self._inverse_initial_values.append(float(value))
        return self.inverse_parameter

    def dataset(self, *, parameters: str | Path, variables: str | Path | None = None) -> "KKTHardNet":
        self.dataset_spec = DatasetSpec(parameters_path=str(parameters), variables_path=None if variables is None else str(variables))
        return self

    def set_dataset(self, *, parameters: str | Path, variables: str | Path | None = None) -> "KKTHardNet":
        return self.dataset(parameters=parameters, variables=variables)

    def use_dataset(self, *, parameters: str | Path, variables: str | Path | None = None) -> "KKTHardNet":
        return self.dataset(parameters=parameters, variables=variables)

    def set_train_config(self, config: dict[str, Any] | KKTTrainConfig) -> "KKTHardNet":
        self._train_config = self._normalize_train_config(config)
        return self

    def set_projection_config(self, config: dict[str, Any] | ProjectionSettings) -> "KKTHardNet":
        self._projection_config = self._normalize_projection_config(config)
        return self

    def model(self, *, train: dict[str, Any] | KKTTrainConfig | None = None, projection: dict[str, Any] | ProjectionSettings | None = None) -> dict[str, Any]:
        return self._run_training("surrogate", train=train, projection=projection)

    def optimize(self, *, train: dict[str, Any] | KKTTrainConfig | None = None, projection: dict[str, Any] | ProjectionSettings | None = None) -> dict[str, Any]:
        return self._run_training("optimize", train=train, projection=projection)

    def estimate(self, *, train: dict[str, Any] | KKTTrainConfig | None = None, projection: dict[str, Any] | ProjectionSettings | None = None) -> dict[str, Any]:
        return self._run_training("estimate", train=train, projection=projection)

    def load(self, metadata_path: str | Path) -> "KKTHardNet":
        metadata_file = _resolve_path(metadata_path)
        with open(metadata_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != "kkt-hardnet-metadata-v1":
            raise ValueError(f"Unsupported metadata format in {metadata_file}.")

        self.name = str(payload["model_name"])
        self.parameter_names = list(payload["problem"]["parameters"])
        self.variable_names = list(payload["problem"]["variables"])
        self.inverse_parameter_names = list(payload["problem"].get("inverse_parameters", []))
        self._inverse_initial_values = list(payload["problem"].get("inverse_initial_values", []))
        self.dataset_spec = DatasetSpec(
            parameters_path=str((metadata_file.parent / payload["artifacts"]["parameters"]).resolve()),
            variables_path=None
            if payload["artifacts"].get("variables") is None
            else str((metadata_file.parent / payload["artifacts"]["variables"]).resolve()),
        )
        self._train_config = self._normalize_train_config(payload.get("training", {}))
        self._projection_config = self._normalize_projection_config(payload.get("projection", {}))

        task = str(payload["task"])
        inverse_mode = bool(payload.get("inverse_mode", False))
        fixed_inverse_values = np.asarray(payload.get("fixed_inverse_values", []), dtype=np.float64).reshape(-1)
        problem_model, _ = build_model_from_string_problem(
            payload["problem"],
            dtype=_dtype(self._train_config["dtype"]),
            train_inverse=inverse_mode,
            inverse_values=fixed_inverse_values,
        )
        weights_path = (metadata_file.parent / payload["artifacts"]["weights"]).resolve()
        params = load_model_weights(weights_path, payload["weights_manifest"])
        projection_layer = make_projection_layer(problem_model, settings=self._projection_settings(self._projection_config))
        self._inference_state = _InferenceState(
            task=task,
            model=problem_model,
            params=params,
            projection=projection_layer,
            data_n_x=len(self.parameter_names),
            output_dir=str(metadata_file.parent),
        )
        self._metadata_path = str(metadata_file)
        return self

    def predict(self, values) -> np.ndarray:
        if self._inference_state is None:
            raise RuntimeError("Please train or load the model before calling predict().")
        data = np.asarray(values, dtype=np.float64)
        squeeze = data.ndim == 1
        if squeeze:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            raise ValueError("predict expects a 1D or 2D array-like input.")
        if data.shape[1] != self._inference_state.data_n_x:
            raise ValueError(
                f"predict expected {self._inference_state.data_n_x} parameter values, got {data.shape[1]}."
            )

        batched_mlp_apply = make_batched_mlp_apply()
        x_batch = jnp.asarray(data, dtype=_dtype(self._train_config["dtype"]))
        params = self._inference_state.params
        network = params["network"] if isinstance(params, dict) and "network" in params else params
        y_hat = batched_mlp_apply(network, x_batch)
        if isinstance(params, dict) and "inverse" in params:
            theta = params["inverse"]
            theta_batch = jnp.broadcast_to(theta, (x_batch.shape[0], theta.shape[0]))
            x_aug = jnp.concatenate([x_batch, theta_batch], axis=1)
        else:
            x_aug = x_batch
        y_proj = self._inference_state.projection.project(x_aug, y_hat)
        out = np.asarray(y_proj, dtype=np.float64)
        return out[0] if squeeze else out

    def sin(self, expr) -> Expression:
        expr = _as_expr(expr)
        return Expression(lambda ctx, e=expr: jnp.sin(e.eval(ctx)), f"sin({expr.text})")

    def cos(self, expr) -> Expression:
        expr = _as_expr(expr)
        return Expression(lambda ctx, e=expr: jnp.cos(e.eval(ctx)), f"cos({expr.text})")

    def exp(self, expr) -> Expression:
        expr = _as_expr(expr)
        return Expression(lambda ctx, e=expr: jnp.exp(e.eval(ctx)), f"exp({expr.text})")

    def log(self, expr) -> Expression:
        expr = _as_expr(expr)
        return Expression(lambda ctx, e=expr: jnp.log(e.eval(ctx)), f"log({expr.text})")

    def sqrt(self, expr) -> Expression:
        expr = _as_expr(expr)
        return Expression(lambda ctx, e=expr: jnp.sqrt(e.eval(ctx)), f"sqrt({expr.text})")

    def abs(self, expr) -> Expression:
        expr = _as_expr(expr)
        return Expression(lambda ctx, e=expr: jnp.abs(e.eval(ctx)), f"abs({expr.text})")

    def _check_symbol(self, kind: str, name: str) -> None:
        pools = {
            "parameter": self.parameter_names,
            "variable": self.variable_names,
            "inverse_parameter": self.inverse_parameter_names,
        }
        if name not in pools[kind]:
            raise AttributeError(f"Unknown {kind} '{name}'.")

    def _normalize_train_config(self, config: dict[str, Any] | KKTTrainConfig | None) -> dict[str, Any]:
        if config is None:
            return {
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 1e-3,
                "train_frac": 0.8,
                "hidden_size": 64,
                "hidden_layers": 2,
                "seed": 42,
                "dtype": "float64",
                "print_every": 1,
                "drop_last": False,
            }
        if isinstance(config, KKTTrainConfig):
            out = asdict(config)
            out.pop("projection", None)
            return out
        out = dict(config)
        out.setdefault("epochs", 2)
        out.setdefault("batch_size", 8)
        out.setdefault("learning_rate", 1e-3)
        out.setdefault("train_frac", 0.8)
        out.setdefault("hidden_size", 64)
        out.setdefault("hidden_layers", 2)
        out.setdefault("seed", 42)
        out.setdefault("dtype", "float64")
        out.setdefault("print_every", 1)
        out.setdefault("drop_last", False)
        return out

    def _normalize_projection_config(self, config: dict[str, Any] | ProjectionSettings | None) -> dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, ProjectionSettings):
            return asdict(config)
        return dict(config)

    def _projection_config_from_train(self, config: dict[str, Any]) -> dict[str, Any]:
        projection_keys = {
            "fb_eps",
            "gn_max_iters",
            "gn_tol",
            "gn_reg",
            "newton_step_length",
            "armijo_alpha",
            "armijo_beta",
            "max_backtrack_iter",
            "armijo_max_steps",
            "backward_reg",
            "max_newton_iter",
            "newton_tol",
            "newton_reg_factor",
        }
        return {key: config[key] for key in projection_keys if key in config}

    def _projection_settings(self, config: dict[str, Any]) -> ProjectionSettings:
        defaults = ProjectionSettings()
        backtrack_iters = int(config.get("max_backtrack_iter", config.get("armijo_max_steps", defaults.max_backtrack_iter)))
        return ProjectionSettings(
            fb_eps=float(config.get("fb_eps", defaults.fb_eps)),
            gn_max_iters=int(config.get("gn_max_iters", config.get("max_newton_iter", defaults.gn_max_iters))),
            gn_tol=float(config.get("gn_tol", config.get("newton_tol", defaults.gn_tol))),
            gn_reg=float(config.get("gn_reg", config.get("newton_reg_factor", defaults.gn_reg))),
            newton_step_length=float(config.get("newton_step_length", defaults.newton_step_length)),
            armijo_alpha=float(config.get("armijo_alpha", defaults.armijo_alpha)),
            armijo_beta=float(config.get("armijo_beta", defaults.armijo_beta)),
            max_backtrack_iter=backtrack_iters,
            armijo_max_steps=backtrack_iters,
            backward_reg=float(config.get("backward_reg", defaults.backward_reg)),
        )

    def _config_dataclass(
        self,
        train: dict[str, Any] | KKTTrainConfig | None,
        projection: dict[str, Any] | ProjectionSettings | None,
    ) -> KKTTrainConfig:
        train_cfg = self._normalize_train_config(self._train_config if train is None else train)
        proj_cfg = {
            **self._projection_config_from_train(train_cfg),
            **self._normalize_projection_config(self._projection_config if projection is None else projection),
        }
        return KKTTrainConfig(
            epochs=int(train_cfg["epochs"]),
            batch_size=int(train_cfg["batch_size"]),
            learning_rate=float(train_cfg["learning_rate"]),
            train_frac=float(train_cfg["train_frac"]),
            hidden_size=int(train_cfg["hidden_size"]),
            hidden_layers=int(train_cfg["hidden_layers"]),
            seed=int(train_cfg["seed"]),
            dtype=str(train_cfg["dtype"]),
            print_every=max(1, int(train_cfg["print_every"])),
            drop_last=bool(train_cfg.get("drop_last", False)),
            projection=self._projection_settings(proj_cfg),
        )

    def _initial_inverse_values(self) -> np.ndarray:
        return np.asarray(self._inverse_initial_values, dtype=np.float64).reshape(-1)

    def _problem_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": list(self.parameter_names),
            "variables": list(self.variable_names),
            "inverse_parameters": list(self.inverse_parameter_names),
            "inverse_initial_values": self._initial_inverse_values().tolist(),
            "objective": None if self.objective is None else self.objective.text,
            "constraints": [constraint.text for constraint in self.constraints.items],
            "constraint_details": [
                {"kind": constraint.kind, "text": constraint.text, "residual": constraint.residual.text}
                for constraint in self.constraints.items
            ],
        }

    def _build_model(
        self,
        *,
        dtype,
        train_inverse: bool,
        inverse_values=None,
        allow_missing_objective: bool,
    ):
        if self.objective is None and not allow_missing_objective:
            raise ValueError("Set model.objective before training an optimization model.")
        if not self.parameter_names:
            raise ValueError("Add at least one parameter.")
        if not self.variable_names:
            raise ValueError("Add at least one variable.")

        inverse_values_j = jnp.asarray([] if inverse_values is None else inverse_values, dtype=dtype).reshape(-1)
        if not train_inverse and self.inverse_parameter_names and inverse_values_j.size != len(self.inverse_parameter_names):
            raise ValueError("Fixed inverse parameters must match the inverse parameter count.")
        if train_inverse and inverse_values_j.size != len(self.inverse_parameter_names):
            raise ValueError("estimate() requires an init_value for every inverse parameter.")

        n_x = len(self.parameter_names) + (len(self.inverse_parameter_names) if train_inverse else 0)
        n_y = len(self.variable_names)
        p_index = {name: idx for idx, name in enumerate(self.parameter_names)}
        y_index = {name: idx for idx, name in enumerate(self.variable_names)}
        inv_index = {name: idx for idx, name in enumerate(self.inverse_parameter_names)}

        def make_ctx(params, vars_dict):
            return _EvalContext(
                y=jnp.ravel(vars_dict["y"]),
                x=jnp.ravel(params["x"]),
                variable_index=y_index,
                parameter_index=p_index,
                inverse_index=inv_index,
                inverse_values=inverse_values_j,
                train_inverse=bool(train_inverse),
                forward_dim=len(self.parameter_names),
            )

        def objective(params, vars_dict):
            if self.objective is None:
                return jnp.asarray(0.0, dtype=dtype)
            return _as_expr(self.objective).eval(make_ctx(params, vars_dict))

        builder = (
            HighLevelNLPBuilder(dtype=dtype)
            .add_parameter("x", n_x)
            .add_variable("y", n_y)
            .set_objective(objective)
        )

        eq_constraints = [constraint for constraint in self.constraints.items if constraint.kind == "eq"]
        ineq_constraints = [constraint for constraint in self.constraints.items if constraint.kind == "ineq"]

        if eq_constraints:

            def eq_block(params, vars_dict):
                ctx = make_ctx(params, vars_dict)
                return jnp.stack([constraint.residual.eval(ctx) for constraint in eq_constraints], axis=0)

            builder = builder.add_nonlinear_equality(eq_block, name="symbolic_eq_block")

        if ineq_constraints:

            def ineq_block(params, vars_dict):
                ctx = make_ctx(params, vars_dict)
                return jnp.stack([constraint.residual.eval(ctx) for constraint in ineq_constraints], axis=0)

            builder = builder.add_nonlinear_inequality(ineq_block, name="symbolic_ineq_block")

        model = builder.build(example_params={"x": jnp.zeros((n_x,), dtype=dtype)}, jit_compile=True)
        return model

    def _read_dataset(self, *, variables_required: bool) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        if self.dataset_spec is None:
            raise ValueError("Call model.dataset(parameters=..., variables=...) before training.")

        parameters_path = _resolve_path(self.dataset_spec.parameters_path)
        if not parameters_path.exists():
            raise FileNotFoundError(f"Parameters CSV not found: {parameters_path}")
        X = self._read_csv_columns(parameters_path, self.parameter_names)

        Y = None
        variables_path: Path | None = None
        if self.dataset_spec.variables_path is not None:
            variables_path = _resolve_path(self.dataset_spec.variables_path)
            if not variables_path.exists():
                raise FileNotFoundError(f"Variables CSV not found: {variables_path}")
            Y = self._read_csv_columns(variables_path, self.variable_names)
            if Y.shape[0] != X.shape[0]:
                raise ValueError("parameters.csv and variables.csv must have the same number of rows.")
        elif variables_required:
            raise ValueError("variables.csv is required for surrogate modeling and parameter estimation.")

        metadata = {
            "dataset": {
                "parameters": str(parameters_path),
                "variables": None if variables_path is None else str(variables_path),
                "num_samples": int(X.shape[0]),
            }
        }
        return X, Y, metadata

    def _read_csv_columns(self, path: Path, expected_columns: list[str]) -> np.ndarray:
        if not expected_columns:
            raise ValueError("Expected columns must be defined before loading data.")
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            headers = list(reader.fieldnames or [])
            missing = [name for name in expected_columns if name not in headers]
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            rows = []
            for row_idx, row in enumerate(reader, start=2):
                try:
                    rows.append([float(row[name]) for name in expected_columns])
                except ValueError as exc:
                    raise ValueError(f"Non-numeric value in {path} at line {row_idx}.") from exc
        if not rows:
            raise ValueError(f"{path} has no data rows.")
        return np.asarray(rows, dtype=np.float64)

    def _copy_dataset_artifacts(self, run_dir: Path, X: np.ndarray, Y: np.ndarray | None) -> dict[str, str | None]:
        parameters_file = run_dir / "parameters.csv"
        variables_file = None if Y is None else run_dir / "variables.csv"
        self._write_matrix_csv(parameters_file, X, self.parameter_names)
        if variables_file is not None:
            self._write_matrix_csv(variables_file, Y, self.variable_names)
        return {
            "parameters": parameters_file.name,
            "variables": None if variables_file is None else variables_file.name,
        }

    def _write_matrix_csv(self, path: Path, data: np.ndarray, headers: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(list(headers))
            writer.writerows(np.asarray(data, dtype=np.float64).tolist())

    def _run_training(
        self,
        task: str,
        *,
        train: dict[str, Any] | KKTTrainConfig | None,
        projection: dict[str, Any] | ProjectionSettings | None,
    ) -> dict[str, Any]:
        train_cfg = self._config_dataclass(train, projection)
        dtype = _dtype(train_cfg.dtype)
        variables_required = task in {"surrogate", "estimate"}
        X, Y, dataset_meta = self._read_dataset(variables_required=variables_required)
        inverse_mode = task == "estimate"
        inverse_values = self._initial_inverse_values()
        fixed_inverse_values = inverse_values if (self.inverse_parameter_names and not inverse_mode) else np.zeros((0,), dtype=np.float64)
        model = self._build_model(
            dtype=dtype,
            train_inverse=inverse_mode,
            inverse_values=inverse_values if self.inverse_parameter_names else np.zeros((0,), dtype=np.float64),
            allow_missing_objective=task != "optimize",
        )

        run_dir = Path.cwd() / f"{self.name}_{_timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        dataset_artifacts = self._copy_dataset_artifacts(run_dir, X, Y)
        result = train_kkt_hardnet(
            model=model,
            X=X,
            Y=Y,
            cfg=train_cfg,
            output_dir=run_dir,
            metadata={
                "task": task,
                "problem": self._problem_spec(),
                **dataset_meta,
            },
            inverse_param_init=inverse_values if inverse_mode else None,
            inverse_param_labels=None,
            inverse_param_names=list(self.inverse_parameter_names) if inverse_mode else None,
            task=task,
        )

        metadata_payload = {
            "format": "kkt-hardnet-metadata-v1",
            "model_name": self.name,
            "task": task,
            "created_at": _timestamp(),
            "problem": self._problem_spec(),
            "training": self._normalize_train_config(train_cfg),
            "projection": self._normalize_projection_config(train_cfg.projection),
            "inverse_mode": inverse_mode,
            "fixed_inverse_values": fixed_inverse_values.tolist(),
            "weights_manifest": result.get("model_weights", result.get("model_weights_manifest")),
            "artifacts": {
                "parameters": dataset_artifacts["parameters"],
                "variables": dataset_artifacts["variables"],
                "weights": "model_weights.npz",
                "summary": "summary.json",
                "history": "history.csv",
                "predictions": "predictions.csv",
            },
        }
        if metadata_payload["weights_manifest"] is None:
            summary_file = run_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as fh:
                    summary_payload = json.load(fh)
                metadata_payload["weights_manifest"] = summary_payload.get("model_weights")
        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(metadata_payload), fh, indent=2, sort_keys=True)

        params = result["params"]
        projection_layer = make_projection_layer(model, settings=train_cfg.projection)
        self._inference_state = _InferenceState(
            task=task,
            model=model,
            params=params,
            projection=projection_layer,
            data_n_x=len(self.parameter_names),
            output_dir=str(run_dir),
        )
        self._metadata_path = str(metadata_file)
        result["metadata_path"] = str(metadata_file)
        return result


ProblemBuilder = KKTHardNet
