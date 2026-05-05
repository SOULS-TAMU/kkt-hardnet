from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from jaxmodel import HighLevelNLPBuilder

from .backbone import make_batched_mlp_apply
from .native_projection import load_native_projection, load_or_compile_native_projection
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


def _fmt_time_sec(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f} s"


def _fmt_time_msec(value) -> str:
    if value is None:
        return "N/A"
    return f"{1000.0 * float(value):.2f} ms"


def _fmt_summary_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4e}" if abs(value) < 1e-3 and value != 0.0 else f"{value:.4f}"
    return str(value)


def _as_expr(value) -> "Expression":
    if isinstance(value, Expression):
        return value
    if "Constant" in globals() and isinstance(value, Constant):
        arr = np.asarray(value.value)
        if arr.ndim != 0:
            raise TypeError(f"Constant '{value.name}' is not scalar.")
        scalar = float(arr)
        return Expression(lambda _ctx, v=scalar: jnp.asarray(v), repr(scalar))
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
    native_projection: Any | None
    data_n_x: int
    output_dir: str | None
    predict_jax: Any | None = None


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


class VectorExpression:
    def __init__(self, fn: Callable, text: str, *, size: int, components: list[Expression] | None = None) -> None:
        self._fn = fn
        self.text = str(text)
        self.size = int(size)
        self.components = list(components) if components is not None else None

    def eval(self, ctx):
        return self._fn(ctx)

    def _component(self, idx: int) -> Expression:
        if self.components is not None:
            return self.components[idx]
        return Expression(lambda ctx, vec=self, i=idx: vec.eval(ctx)[i], f"{self.text}[{idx}]")

    def _binary(self, other, op, symbol: str) -> "VectorExpression":
        rhs = _as_vector_expr(other, size=self.size)
        components = [op(self._component(i), rhs._component(i)) for i in range(self.size)]
        return VectorExpression(
            lambda ctx, lhs=self, rhs=rhs: op(lhs.eval(ctx), rhs.eval(ctx)),
            f"({self.text} {symbol} {rhs.text})",
            size=self.size,
            components=components,
        )

    def _rbinary(self, other, op, symbol: str) -> "VectorExpression":
        lhs = _as_vector_expr(other, size=self.size)
        components = [op(lhs._component(i), self._component(i)) for i in range(self.size)]
        return VectorExpression(
            lambda ctx, lhs=lhs, rhs=self: op(lhs.eval(ctx), rhs.eval(ctx)),
            f"({lhs.text} {symbol} {self.text})",
            size=self.size,
            components=components,
        )

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b, "+")

    def __radd__(self, other):
        return self._rbinary(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b, "-")

    def __rsub__(self, other):
        return self._rbinary(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        rhs = _as_expr(other)
        return VectorExpression(
            lambda ctx, lhs=self, rhs=rhs: lhs.eval(ctx) * rhs.eval(ctx),
            f"({self.text} * {rhs.text})",
            size=self.size,
            components=[self._component(i) * rhs for i in range(self.size)],
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        rhs = _as_expr(other)
        return VectorExpression(
            lambda ctx, lhs=self, rhs=rhs: lhs.eval(ctx) / rhs.eval(ctx),
            f"({self.text} / {rhs.text})",
            size=self.size,
            components=[self._component(i) / rhs for i in range(self.size)],
        )

    def __neg__(self):
        return VectorExpression(
            lambda ctx, expr=self: -expr.eval(ctx),
            f"(-{self.text})",
            size=self.size,
            components=[-self._component(i) for i in range(self.size)],
        )

    def __getitem__(self, idx: int) -> Expression:
        index = int(idx)
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(index)
        return self._component(index)

    def _compare(self, other, kind: str, symbol: str):
        rhs = _as_vector_expr(other, size=self.size)
        constraints = []
        for idx in range(self.size):
            lhs_i = self._component(idx)
            rhs_i = rhs._component(idx)
            if symbol == "<=":
                constraints.append(Constraint(lhs_i - rhs_i, kind, f"{lhs_i.text} <= {rhs_i.text}"))
            elif symbol == ">=":
                constraints.append(Constraint(rhs_i - lhs_i, kind, f"{lhs_i.text} >= {rhs_i.text}"))
            else:
                constraints.append(Constraint(lhs_i - rhs_i, kind, f"{lhs_i.text} == {rhs_i.text}"))
        return constraints

    def __le__(self, other):
        return self._compare(other, "ineq", "<=")

    def __ge__(self, other):
        return self._compare(other, "ineq", ">=")

    def __eq__(self, other):  # type: ignore[override]
        return self._compare(other, "eq", "==")


class Constant:
    def __init__(self, name: str, value: Any) -> None:
        self.name = str(name)
        self.value = np.asarray(value)

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def _expr(self):
        arr = np.asarray(self.value)
        if arr.ndim == 0:
            return _as_expr(self)
        if arr.ndim == 1:
            return _constant_to_vector_expr(self)
        raise TypeError(f"Constant '{self.name}' is not directly usable as a scalar/vector expression.")

    def __add__(self, other):
        return self._expr() + other

    def __radd__(self, other):
        return other + self._expr()

    def __sub__(self, other):
        return self._expr() - other

    def __rsub__(self, other):
        return other - self._expr()

    def __mul__(self, other):
        return self._expr() * other

    def __rmul__(self, other):
        return other * self._expr()

    def __truediv__(self, other):
        return self._expr() / other

    def __rtruediv__(self, other):
        return other / self._expr()

    def __neg__(self):
        return -self._expr()

    def __le__(self, other):
        return self._expr() <= other

    def __ge__(self, other):
        return self._expr() >= other

    def __eq__(self, other):  # type: ignore[override]
        return self._expr() == other

    def __repr__(self) -> str:
        return f"Constant(name={self.name!r}, shape={tuple(self.value.shape)!r})"


def _constant_to_vector_expr(const: Constant) -> VectorExpression:
    arr = np.asarray(const.value, dtype=np.float64).reshape(-1)
    components = [Expression(lambda _ctx, v=float(value): jnp.asarray(v), repr(float(value))) for value in arr]
    return VectorExpression(
        lambda _ctx, value=arr: jnp.asarray(value),
        const.name,
        size=arr.size,
        components=components,
    )


def _as_vector_expr(value, *, size: int | None = None) -> VectorExpression:
    if isinstance(value, VectorExpression):
        vector = value
    elif isinstance(value, _SymbolNamespace):
        vector = value.vector()
    elif isinstance(value, Constant):
        arr = np.asarray(value.value)
        if arr.ndim == 0:
            if size is None:
                raise TypeError("Cannot broadcast a scalar constant to a vector without a target size.")
            return _as_vector_expr(_as_expr(value), size=size)
        if arr.ndim != 1:
            raise TypeError(f"Constant '{value.name}' is not a vector.")
        vector = _constant_to_vector_expr(value)
    elif isinstance(value, Expression):
        if size is None:
            raise TypeError("Cannot broadcast a scalar expression to a vector without a target size.")
        vector = VectorExpression(
            lambda ctx, expr=value, count=int(size): jnp.broadcast_to(expr.eval(ctx), (count,)),
            value.text,
            size=int(size),
            components=[value for _ in range(int(size))],
        )
    else:
        arr = np.asarray(value)
        if arr.ndim == 0:
            if size is None:
                raise TypeError("Cannot broadcast a scalar constant to a vector without a target size.")
            return _as_vector_expr(_as_expr(float(arr)), size=size)
        if arr.ndim != 1:
            raise TypeError(f"Expected a vector expression, got {type(value).__name__}.")
        vector = _constant_to_vector_expr(Constant("_literal", arr))
    if size is not None:
        expected = int(size)
        if vector.size == 1 and expected != 1:
            return _as_vector_expr(vector._component(0), size=expected)
        if vector.size != expected:
            raise ValueError(f"Vector shape mismatch: expected size {expected}, got {vector.size}.")
    return vector


class _SymbolNamespace:
    def __init__(self, builder: "KKTHardNet", kind: str) -> None:
        self._builder = builder
        self._kind = kind

    def __getattr__(self, name: str) -> Expression:
        return self[name]

    def __getitem__(self, name: str) -> Expression:
        if isinstance(name, int):
            names = {
                "parameter": self._builder.parameter_names,
                "variable": self._builder.variable_names,
                "inverse_parameter": self._builder.inverse_parameter_names,
            }[self._kind]
            return self[names[int(name)]]
        self._builder._check_symbol(self._kind, name)
        return Expression(lambda ctx, kind=self._kind, n=name: ctx.value(kind, n), name)

    def vector(self) -> VectorExpression:
        names = {
            "parameter": self._builder.parameter_names,
            "variable": self._builder.variable_names,
            "inverse_parameter": self._builder.inverse_parameter_names,
        }[self._kind]
        token = {
            "parameter": "x",
            "variable": "y",
            "inverse_parameter": "theta",
        }[self._kind]
        components = [self[name] for name in names]
        return VectorExpression(
            lambda ctx, kind=self._kind, ordered=list(names): jnp.asarray(
                [ctx.value(kind, name) for name in ordered]
            ),
            token,
            size=len(names),
            components=components,
        )


class ConstraintList:
    def __init__(self) -> None:
        self.items: list[Constraint] = []

    def add(self, *constraints: Constraint) -> "ConstraintList":
        for constraint in constraints:
            if isinstance(constraint, (list, tuple)):
                self.add(*constraint)
                continue
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
        self._constants: dict[str, np.ndarray] = {}
        self._constant_counter = 0
        self._extracted_problem_path: str | None = None
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

    def matrix(self, values) -> Constant:
        """Register a constant matrix and return its wrapper."""
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError("matrix(...) expects a 2D array.")
        return self._register_constant(values)

    def vector(self, values) -> Constant:
        """Register a constant vector and return its wrapper."""
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError("vector(...) expects a 1D array.")
        return self._register_constant(values)

    def tensor(self, values) -> Constant:
        """Register a constant tensor and return its wrapper."""
        arr = np.asarray(values)
        if arr.ndim != 3:
            raise ValueError("tensor(...) expects a 3D array.")
        return self._register_constant(values)

    def extract(self, path: str | Path) -> dict[str, Constant]:
        """Load arrays from a ``.npz`` file and expose them as model attributes."""
        target = _resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"problem.npz not found: {target}")
        loaded = np.load(target, allow_pickle=False)
        extracted: dict[str, Constant] = {}
        for key in loaded.files:
            const = self._register_constant(loaded[key], name=key)
            extracted[key] = const
            setattr(self, key, const)
        self._extracted_problem_path = str(target)
        print(f"Loaded problem constants: {sorted(extracted.keys())}")
        return extracted

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

    def load(self, metadata_path: str | Path, *, verbose: bool | None = None) -> "KKTHardNet":
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
        native_projection, native_manifest = load_or_compile_native_projection(
            metadata_file.parent,
            problem=payload["problem"],
            settings=self._normalize_projection_config(self._projection_config),
        )
        self._inference_state = _InferenceState(
            task=task,
            model=problem_model,
            params=params,
            projection=projection_layer,
            native_projection=native_projection,
            data_n_x=len(self.parameter_names),
            output_dir=str(metadata_file.parent),
        )
        # Build and cache jitted predictor
        self._inference_state.predict_jax = self._make_jitted_predict()

        # Warm-up (compile once here instead of first predict)
        dummy = jnp.zeros(
            (1, self._inference_state.data_n_x),
            dtype=_dtype(self._train_config["dtype"]),
        )
        _ = self._inference_state.predict_jax(dummy).block_until_ready()
        self._metadata_path = str(metadata_file)
        if verbose is None:
            verbose = True
        if verbose:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("\n" + "=" * 120)
            print("Model Loaded Successfully")
            print("-" * 120)
            print(f"Model Name   : {self.name}")
            print(f"Location     : {metadata_file.parent}")
            print(f"Time         : {now}")
            print("=" * 120 + "\n")
        return self

    def predict(self, values, *, projection_backend: str = "jax") -> np.ndarray:
        if self._inference_state is None:
            raise RuntimeError("Please train or load the model before calling predict().")
        backend = str(projection_backend).strip().lower()
        if backend not in {"auto", "jax", "native"}:
            raise ValueError("projection_backend must be one of 'auto', 'jax', or 'native'.")
        use_native = backend == "native" or (backend == "auto" and self._inference_state.native_projection is not None)
        if backend == "native" and self._inference_state.native_projection is None:
            raise RuntimeError("This run does not have a loadable native projection artifact.")
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

        x_batch = jnp.asarray(data, dtype=_dtype(self._train_config["dtype"]))

        if use_native:
            batched_mlp_apply = make_batched_mlp_apply()
            params = self._inference_state.params
            network = params["network"] if isinstance(params, dict) and "network" in params else params
            y_hat = batched_mlp_apply(network, x_batch)

            if isinstance(params, dict) and "inverse" in params:
                theta = params["inverse"]
                theta_batch = jnp.broadcast_to(theta, (x_batch.shape[0], theta.shape[0]))
                x_aug = jnp.concatenate([x_batch, theta_batch], axis=1)
            else:
                x_aug = x_batch

            y_proj_np = self._inference_state.native_projection.project(
                np.asarray(x_aug, dtype=np.float64),
                np.asarray(y_hat, dtype=np.float64),
            )
            out = np.asarray(y_proj_np, dtype=np.float64)
            return out[0] if squeeze else out

        y_proj = self._inference_state.predict_jax(x_batch)
        out = np.asarray(y_proj, dtype=np.float64)
        return out[0] if squeeze else out

    def summary(self) -> dict[str, Any]:
        """Print a compact summary of the current completed or loaded run."""
        run_dir = self._run_dir()
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"summary.json not found in {run_dir}. Train or load a completed run first.")
        with open(summary_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload = self._ensure_summary_inference_times(payload, summary_path)

        dims = payload.get("dims", {}) if isinstance(payload.get("dims"), dict) else {}
        final = payload.get("final", {}) if isinstance(payload.get("final"), dict) else {}
        dataset = payload.get("metadata", {}).get("dataset", {}) if isinstance(payload.get("metadata"), dict) else {}
        cfg = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
        total_samples = dataset.get("num_samples")
        train_samples = payload.get("train_samples")
        val_samples = payload.get("val_samples")
        if total_samples is not None and (train_samples is None or val_samples is None):
            train_samples = int(float(total_samples) * float(cfg.get("train_frac", 0.8)))
            val_samples = int(total_samples) - int(train_samples)
        max_violation = payload.get("max_violation")
        if max_violation is None and final:
            max_violation = max(
                abs(float(final.get("tr_eq", final.get("train_eq_l2", 0.0)))),
                abs(float(final.get("tr_ineq", final.get("train_ineq_l2", 0.0)))),
                abs(float(final.get("val_eq", final.get("val_eq_l2", 0.0)))),
                abs(float(final.get("val_ineq", final.get("val_ineq_l2", 0.0)))),
            )
        rows = [
            ("Model Name", payload.get("model_name", self.name)),
            ("No. of Parameters", payload.get("num_parameters", dims.get("n_x", len(self.parameter_names)))),
            ("No. of Variables", payload.get("num_variables", dims.get("n_y", len(self.variable_names)))),
            ("No. of Equalities", payload.get("num_equalities", dims.get("n_eq"))),
            ("No. of Inequalities", payload.get("num_inequalities", dims.get("n_ineq"))),
            ("No. of Train Samples", train_samples),
            ("No. of Validation Samples", val_samples),
            ("Maximum Constraint Violation", max_violation),
            ("Training Time", _fmt_time_sec(payload.get("training_wall_time_sec"))),
            ("Est. JAX Single Inference Time", _fmt_time_msec(payload.get("estimated_jax_single_inference_time_sec"))),
            # ("Est. Native Single Inference Time", _fmt_time_msec(payload.get("estimated_native_single_inference_time_sec"))),
            ("Est. JAX Batch Inference Time", _fmt_time_msec(payload.get("estimated_jax_batch_inference_time_sec"))),
        ]
        print("📊 KKT-HardNet Summary")
        print("-" * 60)
        width = max(len(key) for key, _ in rows)
        for key, value in rows:
            print(f"{key:<{width}} : {_fmt_summary_value(value)}")
        print("-" * 60)
        print(
            "Note: Inference time estimations are based on\n"
            "microbenchmarking on the hardware used during\n"
            "training and may vary across different hardware\n"
            "and runtime conditions."
        )
        # return payload

    def _ensure_summary_inference_times(self, payload: dict[str, Any], summary_path: Path) -> dict[str, Any]:
        if self._inference_state is None:
            return payload
        needs_jax = payload.get("estimated_jax_single_inference_time_sec") is None
        needs_native = (
            payload.get("estimated_native_single_inference_time_sec") is None
            and self._inference_state.native_projection is not None
        )
        needs_batch = payload.get("estimated_jax_batch_inference_time_sec") is None
        if not (needs_jax or needs_native or needs_batch):
            return payload
        params_path = summary_path.parent / "parameters.csv"
        if not params_path.exists():
            return payload
        try:
            X = self._read_csv_columns(params_path, self.parameter_names)
        except Exception:
            return payload
        if X.size == 0:
            return payload
        samples = X[: min(50, X.shape[0])]

        def time_single(backend: str):
            self.predict(samples[0], projection_backend=backend)
            start = time.perf_counter()
            for row in samples:
                self.predict(row, projection_backend=backend)
            elapsed = time.perf_counter() - start
            payload[f"estimated_{backend}_single_inference_time_sec"] = float(elapsed / max(1, samples.shape[0]))
            payload[f"estimated_{backend}_single_total_time_sec"] = float(elapsed)
            payload[f"estimated_{backend}_single_error"] = None

        try:
            if needs_jax:
                time_single("jax")
        except Exception as exc:
            payload["estimated_jax_single_error"] = f"{type(exc).__name__}: {exc}"
        try:
            if needs_native:
                time_single("native")
        except Exception as exc:
            payload["estimated_native_single_error"] = f"{type(exc).__name__}: {exc}"
        try:
            if needs_batch:
                cfg = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
                batch_size = int(max(1, cfg.get("batch_size", 32)))
                if X.shape[0] < batch_size:
                    reps = int(np.ceil(batch_size / X.shape[0]))
                    batch = np.tile(X, (reps, 1))[:batch_size]
                else:
                    batch = X[:batch_size]
                self.predict(batch, projection_backend="jax")
                start = time.perf_counter()
                for _ in range(50):
                    self.predict(batch, projection_backend="jax")
                elapsed = time.perf_counter() - start
                payload["estimated_jax_batch_inference_time_sec"] = float(elapsed / 50.0)
                payload["estimated_jax_batch_total_time_sec"] = float(elapsed)
                payload["estimated_jax_batch_error"] = None
        except Exception as exc:
            payload["estimated_jax_batch_error"] = f"{type(exc).__name__}: {exc}"
        try:
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(_json_safe(payload), fh, indent=2, sort_keys=True)
        except Exception:
            pass
        return payload

    def plot_history(
        self,
        *,
        show: bool = True,
        save_dir: str | Path | None = None,
        bg: str = "grey",
    ):
        """Plot and save MSE/loss and constraint violation histories."""
        history = self._read_history()
        epochs = history["epoch"]
        tr_mse = self._history_column(history, "tr_mse")
        val_mse = self._history_column(history, "val_mse")
        tr_violation = np.maximum(self._history_column(history, "tr_eq"), self._history_column(history, "tr_ineq"))
        val_violation = np.maximum(self._history_column(history, "val_eq"), self._history_column(history, "val_ineq"))
        tr_violation = np.maximum(tr_violation, 1e-16)
        val_violation = np.maximum(val_violation, 1e-16)

        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        font_names = {font.name for font in font_manager.fontManager.ttflist}
        font_family = "Arial" if "Arial" in font_names else "DejaVu Sans"
        bg_colors = {
            "grey": "#E5E5E5",
            "white": "#FFFFFF",
        }
        fig_bg = bg_colors.get(str(bg).lower(), "#E5E5E5")
        style = "default" if str(bg).lower() == "white" else "ggplot"

        with plt.style.context(style):
            plt.rcParams.update(
                {
                    "font.family": font_family,
                    "font.size": 32,
                    "axes.titlesize": 32,
                    "axes.labelsize": 24,
                    "legend.fontsize": 24,
                    "xtick.labelsize": 20,
                    "ytick.labelsize": 20,
                }
            )
            fig, axes = plt.subplots(1, 2, figsize=(20, 6), facecolor=fig_bg)
            for ax in axes:
                ax.set_facecolor(fig_bg)
            axes[0].plot(epochs, tr_mse, linewidth=2, label="Train MSE")
            axes[0].plot(epochs, val_mse, linewidth=2, linestyle="--", label="Validation MSE")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("MSE")
            axes[0].set_title("Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(epochs, tr_violation, linewidth=2, label="Train Violation")
            axes[1].plot(epochs, val_violation, linewidth=2, linestyle="--", label="Validation Violation")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Max. Violation")
            axes[1].set_title("Violation")
            axes[1].set_yscale("log")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            target = self._run_dir() if save_dir is None else _resolve_path(save_dir)
            target.mkdir(parents=True, exist_ok=True)
            fig.savefig(target / "training_history.png", dpi=600, bbox_inches="tight", facecolor=fig_bg)
            if show:
                plt.show()
        # return fig, axes

    def _run_dir(self) -> Path:
        if self._inference_state is not None and self._inference_state.output_dir is not None:
            return _resolve_path(self._inference_state.output_dir)
        if self._metadata_path is not None:
            return _resolve_path(self._metadata_path).parent
        raise RuntimeError("No completed or loaded run is available.")

    def _read_history(self) -> dict[str, np.ndarray]:
        history_path = self._run_dir() / "history.csv"
        if not history_path.exists():
            raise RuntimeError(f"history.csv not found in {history_path.parent}.")
        data = np.genfromtxt(history_path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
        if data.dtype.names is None:
            raise RuntimeError(f"history.csv has no header: {history_path}")
        return {name: np.atleast_1d(data[name]).astype(np.float64) for name in data.dtype.names}

    @staticmethod
    def _history_column(history: dict[str, np.ndarray], name: str) -> np.ndarray:
        if name in history:
            return history[name]
        aliases = {
            "tr_mse": ("tr_task_loss", "train_mse_y", "train_mse", "tr_loss", "train_loss", "train_objective"),
            "val_mse": ("val_task_loss", "val_mse_y", "validation_mse", "val_loss", "validation_loss", "val_objective"),
            "tr_loss": ("train_loss", "train_objective"),
            "val_loss": ("validation_loss", "val_objective"),
            "tr_eq": ("train_eq_l2", "train_eq_violation"),
            "tr_ineq": ("train_ineq_l2", "train_ineq_violation"),
            "val_eq": ("val_eq_l2", "val_eq_violation"),
            "val_ineq": ("val_ineq_l2", "val_ineq_violation"),
        }
        for alias in aliases.get(name, ()):
            if alias in history:
                return history[alias]
        if "epoch" in history:
            return np.zeros_like(history["epoch"], dtype=np.float64)
        raise KeyError(name)

    def lin(self, matrix, expr):
        """Form ``A @ expr`` from an extracted or registered matrix/vector."""
        const = self._ensure_constant(matrix)
        vector = _as_vector_expr(expr)
        arr = np.asarray(const.value, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != vector.size:
                raise ValueError(f"lin(...) dimension mismatch: {arr.shape[0]} vs {vector.size}.")
            return self._linear_combination(arr, vector)
        if arr.ndim == 2:
            if arr.shape[1] != vector.size:
                raise ValueError(f"lin(...) dimension mismatch: {arr.shape[1]} vs {vector.size}.")
            components = [self._linear_combination(row, vector) for row in arr]
            return VectorExpression(
                lambda ctx, a=arr, vec=vector: jnp.asarray(a) @ jnp.asarray(vec.eval(ctx)),
                f"lin({const.name}, {vector.text})",
                size=arr.shape[0],
                components=components,
            )
        raise ValueError("lin(...) expects a 1D or 2D constant.")

    def batch_lin(self, matrix, expr):
        """Form a vector expression from a 2D matrix and vector expression."""
        return self.lin(matrix, expr)

    def quad(self, matrix, expr) -> Expression:
        """Form ``expr.T @ Q @ expr``."""
        const = self._ensure_constant(matrix)
        vector = _as_vector_expr(expr)
        arr = np.asarray(const.value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("quad(...) expects a 2D constant.")
        if arr.shape[0] != vector.size or arr.shape[1] != vector.size:
            raise ValueError(f"quad(...) expects shape ({vector.size}, {vector.size}), got {arr.shape}.")
        terms: list[Expression] = []
        for i in range(vector.size):
            for j in range(vector.size):
                coeff = float(arr[i, j])
                if coeff != 0.0:
                    terms.append(coeff * vector[i] * vector[j])
        return self._sum_expressions(terms, text_if_empty="0.0")

    def batch_quad(self, tensor, expr) -> VectorExpression:
        """Form batched quadratic expressions from a rank-3 tensor."""
        const = self._ensure_constant(tensor)
        vector = _as_vector_expr(expr)
        arr = np.asarray(const.value, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError("batch_quad(...) expects a 3D constant.")
        if arr.shape[1] != vector.size or arr.shape[2] != vector.size:
            raise ValueError(f"batch_quad(...) expects trailing shape ({vector.size}, {vector.size}), got {arr.shape}.")
        components = [self.quad(Constant(f"{const.name}_{idx}", arr[idx]), vector) for idx in range(arr.shape[0])]
        return VectorExpression(
            lambda ctx, a=arr, vec=vector: jnp.einsum("mij,i,j->m", jnp.asarray(a), jnp.ravel(vec.eval(ctx)), jnp.ravel(vec.eval(ctx))),
            f"batch_quad({const.name}, {vector.text})",
            size=arr.shape[0],
            components=components,
        )

    def batch_exp(self, expr) -> VectorExpression:
        return self.exp(expr)

    def sin(self, expr):
        return self._elementwise(expr, jnp.sin, "sin")

    def cos(self, expr):
        return self._elementwise(expr, jnp.cos, "cos")

    def exp(self, expr):
        return self._elementwise(expr, jnp.exp, "exp")

    def log(self, expr):
        return self._elementwise(expr, jnp.log, "log")

    def sqrt(self, expr):
        return self._elementwise(expr, jnp.sqrt, "sqrt")

    def abs(self, expr):
        return self._elementwise(expr, jnp.abs, "abs")

    def _elementwise(self, expr, fn, name: str):
        if isinstance(expr, (VectorExpression, _SymbolNamespace, Constant)) and not (
            isinstance(expr, Constant) and np.asarray(expr.value).ndim == 0
        ):
            vector = _as_vector_expr(expr)
            components = [self._elementwise(vector[idx], fn, name) for idx in range(vector.size)]
            return VectorExpression(
                lambda ctx, e=vector: fn(e.eval(ctx)),
                f"{name}({vector.text})",
                size=vector.size,
                components=components,
            )
        scalar = _as_expr(expr)
        return Expression(lambda ctx, e=scalar: fn(e.eval(ctx)), f"{name}({scalar.text})")

    def _register_constant(self, value, *, name: str | None = None) -> Constant:
        if name is None:
            name = f"_const_{self._constant_counter}"
            self._constant_counter += 1
        arr = np.asarray(value)
        const = Constant(str(name), arr)
        self._constants[str(name)] = arr
        return const

    def _ensure_constant(self, value) -> Constant:
        if isinstance(value, Constant):
            return value
        return self._register_constant(value)

    def _sum_expressions(self, terms: list[Expression], *, text_if_empty: str) -> Expression:
        if not terms:
            return Expression(lambda _ctx: jnp.asarray(0.0), text_if_empty)
        out = terms[0]
        for term in terms[1:]:
            out = out + term
        return out

    def _linear_combination(self, coeffs: np.ndarray, vector: VectorExpression) -> Expression:
        terms: list[Expression] = []
        for idx, coeff in enumerate(np.asarray(coeffs, dtype=np.float64).reshape(-1)):
            scalar = float(coeff)
            if scalar != 0.0:
                terms.append(scalar * vector[idx])
        return self._sum_expressions(terms, text_if_empty="0.0")

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
                "epochs": 1200,
                "batch_size": 32,
                "learning_rate": 1e-3,
                "train_frac": 0.8,
                "hidden_size": 64,
                "hidden_layers": 2,
                "seed": 42,
                "dtype": "float64",
                "print_every": 1,
                "drop_last": False,
                "eta": None,
                "epoch_mlp": None,
                "cons_alpha": 0.0,
            }
        if isinstance(config, KKTTrainConfig):
            out = asdict(config)
            out.pop("projection", None)
            return out
        out = dict(config)
        out.setdefault("epochs", 1200)
        out.setdefault("batch_size", 32)
        out.setdefault("learning_rate", 1e-3)
        out.setdefault("train_frac", 0.8)
        out.setdefault("hidden_size", 64)
        out.setdefault("hidden_layers", 2)
        out.setdefault("seed", 42)
        out.setdefault("dtype", "float64")
        out.setdefault("print_every", 1)
        out.setdefault("drop_last", False)
        out.setdefault("eta", None)
        out.setdefault("epoch_mlp", None)
        out.setdefault("cons_alpha", 0.0)
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
    
    def _make_jitted_predict(self):
        state = self._inference_state
        if state is None:
            return None

        params = state.params
        projection = state.projection
        dtype = _dtype(self._train_config["dtype"])
        batched_mlp_apply = make_batched_mlp_apply()

        inverse_mode = isinstance(params, dict) and "inverse" in params

        def network_tree(params_in):
            return params_in["network"] if inverse_mode else params_in

        def augmented_x(params_in, x_batch):
            if not inverse_mode:
                return x_batch
            theta = params_in["inverse"]
            theta_batch = jnp.broadcast_to(theta, (x_batch.shape[0], theta.shape[0]))
            return jnp.concatenate([x_batch, theta_batch], axis=1)

        @jax.jit
        def predict_jax(x_batch):
            x_batch = jnp.asarray(x_batch, dtype=dtype)
            y_hat = batched_mlp_apply(network_tree(params), x_batch)
            x_aug = augmented_x(params, x_batch)
            return projection.project(x_aug, y_hat)

        return predict_jax

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
            eta=None if train_cfg.get("eta") is None else float(train_cfg["eta"]),
            epoch_mlp=None if train_cfg.get("epoch_mlp") is None else int(train_cfg["epoch_mlp"]),
            cons_alpha=float(train_cfg.get("cons_alpha", 0.0)),
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
                "native_projection_manifest": "projection_native.json",
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
            native_projection=load_native_projection(run_dir),
            data_n_x=len(self.parameter_names),
            output_dir=str(run_dir),
        )
        self._metadata_path = str(metadata_file)
        result["metadata_path"] = str(metadata_file)
        return result


ProblemBuilder = KKTHardNet
