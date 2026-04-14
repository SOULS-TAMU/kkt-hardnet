from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import jax.numpy as jnp
import numpy as np

from jaxmodel import HighLevelNLPBuilder


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
    path: str
    parameter_columns: list[str]
    variable_columns: list[str]


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
    def __init__(self, builder: "ProblemBuilder", kind: str) -> None:
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


class BoundSpec:
    def __init__(self) -> None:
        self.lower = None
        self.upper = None

    def set(self, *, lower=None, upper=None) -> "BoundSpec":
        self.lower = lower
        self.upper = upper
        return self

    def set_all(self, *, lower=None, upper=None) -> "BoundSpec":
        return self.set(lower=lower, upper=upper)

    def __call__(self, *, lower=None, upper=None) -> "BoundSpec":
        return self.set(lower=lower, upper=upper)

    def arrays(self, n_y: int, *, dtype, default_radius: float):
        lower = -float(default_radius) if self.lower is None else self.lower
        upper = float(default_radius) if self.upper is None else self.upper
        return _bound_array(lower, n_y, dtype), _bound_array(upper, n_y, dtype)


def _bound_array(value, n_y: int, dtype):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        arr = np.full((int(n_y),), float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.size != int(n_y):
        raise ValueError(f"Bounds must be scalar or length {n_y}, got length {arr.size}.")
    return jnp.asarray(arr, dtype=dtype)


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


class ProblemBuilder:
    def __init__(self, *, y_bound: float = 10.0) -> None:
        self.parameter_names: list[str] = []
        self.variable_names: list[str] = []
        self.inverse_parameter_names: list[str] = []
        self.parameter = _SymbolNamespace(self, "parameter")
        self.variable = _SymbolNamespace(self, "variable")
        self.inverse_parameter = _SymbolNamespace(self, "inverse_parameter")
        self.parameters = self.parameter
        self.variables = self.variable
        self.inverse_parameters = self.inverse_parameter
        self.objective: Expression | None = None
        self.constraints = ConstraintList()
        self.bounds = BoundSpec()
        self.y_bound = float(y_bound)
        self.dataset_spec: DatasetSpec | None = None

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

    def add_inverse_parameter(self, names: str | Iterable[str]):
        for name in _coerce_names(names, allow_empty=True):
            if name in self.parameter_names or name in self.variable_names or name in self.inverse_parameter_names:
                raise ValueError(f"Duplicate symbol name '{name}'.")
            self.inverse_parameter_names.append(name)
        return self.inverse_parameter

    def set_dataset(
        self,
        path: str | Path,
        *,
        parameter_columns: Iterable[str],
        variable_columns: Iterable[str],
    ) -> "ProblemBuilder":
        """Use a CSV dataset instead of solving the symbolic problem for labels."""

        self.dataset_spec = DatasetSpec(
            path=str(path),
            parameter_columns=_coerce_names(parameter_columns),
            variable_columns=_coerce_names(variable_columns),
        )
        return self

    def use_dataset(
        self,
        path: str | Path,
        *,
        parameter_columns: Iterable[str],
        variable_columns: Iterable[str],
    ) -> "ProblemBuilder":
        return self.set_dataset(path, parameter_columns=parameter_columns, variable_columns=variable_columns)

    def run(
        self_or_args,
        args=None,
        *,
        root: Path,
        data: dict[str, Any] | None = None,
        train: dict[str, Any] | None = None,
        proj: dict[str, Any] | None = None,
        model=None,
        X=None,
        Y=None,
        metadata: dict[str, Any] | None = None,
        problem_meta: dict[str, Any] | None = None,
        parameter_names: list[str] | None = None,
        variable_names: list[str] | None = None,
        inverse_param_init=None,
        inverse_param_labels=None,
        inverse_param_names: list[str] | None = None,
    ) -> int:
        """Run either a standard configured case or this builder-defined case."""

        from .utils import _run_builder_case, _run_prepared_case, _run_standard_case

        if isinstance(self_or_args, ProblemBuilder):
            if args is None:
                raise ValueError("Builder runs require parsed CLI args.")
            if data is None or train is None:
                raise ValueError("Builder runs require data and train dictionaries.")
            return _run_builder_case(args, root=root, builder=self_or_args, data=data, train=train, proj=proj)

        if args is not None:
            raise ValueError("Use ProblemBuilder.run(args, root=...) or builder.run(args, root=..., data=..., train=...).")
        if model is not None or X is not None or Y is not None:
            if model is None or X is None or Y is None:
                raise ValueError("Prepared runs require model, X, and Y.")
            if data is None or train is None:
                raise ValueError("Prepared runs require data and train dictionaries.")
            return _run_prepared_case(
                self_or_args,
                root=root,
                model=model,
                X=X,
                Y=Y,
                data=data,
                train=train,
                proj=proj,
                metadata=metadata,
                problem_meta=problem_meta,
                parameter_names=parameter_names,
                variable_names=variable_names,
                inverse_param_init=inverse_param_init,
                inverse_param_labels=inverse_param_labels,
                inverse_param_names=inverse_param_names,
            )
        return _run_standard_case(self_or_args, root=root, data=data, train=train, proj=proj)

    def _check_symbol(self, kind: str, name: str) -> None:
        pools = {
            "parameter": self.parameter_names,
            "variable": self.variable_names,
            "inverse_parameter": self.inverse_parameter_names,
        }
        if name not in pools[kind]:
            raise AttributeError(f"Unknown {kind} '{name}'.")

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

    def build_model(
        self,
        *,
        dtype=jnp.float64,
        train_inverse: bool = False,
        inverse_values=None,
        allow_missing_objective: bool = False,
    ):
        if self.objective is None and not allow_missing_objective:
            raise ValueError("Set builder.objective before build_model().")
        if not self.parameter_names:
            raise ValueError("Add at least one parameter.")
        if not self.variable_names:
            raise ValueError("Add at least one variable.")

        inverse_values_j = jnp.asarray([] if inverse_values is None else inverse_values, dtype=dtype).reshape(-1)
        if not train_inverse and self.inverse_parameter_names and inverse_values_j.size != len(self.inverse_parameter_names):
            raise ValueError("Forward mode requires one fixed value for each inverse parameter.")
        if train_inverse and inverse_values_j.size not in {0, len(self.inverse_parameter_names)}:
            raise ValueError("inverse_values must be empty or match the inverse parameter count.")

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

        lower, upper = self.bounds.arrays(n_y, dtype=dtype, default_radius=self.y_bound)
        zeros = jnp.zeros((n_y, n_x), dtype=dtype)
        model = (
            builder
            .set_affine_lower_bound(var_name="y", param_name="x", M=zeros, c=lower)
            .set_affine_upper_bound(var_name="y", param_name="x", M=zeros, c=upper)
            .build(example_params={"x": jnp.zeros((n_x,), dtype=dtype)}, jit_compile=True)
        )
        return model, self.metadata(train_inverse=train_inverse, inverse_values=inverse_values_j)

    def metadata(self, *, train_inverse: bool, inverse_values) -> dict:
        return {
            "builder": "ProblemBuilder",
            "parameter_names": list(self.parameter_names),
            "variable_names": list(self.variable_names),
            "inverse_parameter_names": list(self.inverse_parameter_names),
            "train_inverse": bool(train_inverse),
            "fixed_inverse_values": np.asarray(inverse_values, dtype=np.float64).reshape(-1).tolist(),
            "objective": None if self.objective is None else self.objective.text,
            "constraints": [
                {"kind": constraint.kind, "text": constraint.text, "residual": constraint.residual.text}
                for constraint in self.constraints.items
            ],
            "bounds": {
                "lower": None if self.bounds.lower is None else np.asarray(self.bounds.lower).reshape(-1).tolist(),
                "upper": None if self.bounds.upper is None else np.asarray(self.bounds.upper).reshape(-1).tolist(),
                "default_y_bound": self.y_bound,
            },
            "dataset": None if self.dataset_spec is None else {
                "path": self.dataset_spec.path,
                "parameter_columns": list(self.dataset_spec.parameter_columns),
                "variable_columns": list(self.dataset_spec.variable_columns),
            },
        }
