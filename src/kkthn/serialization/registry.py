"""Small explicit registry for custom serializable factories."""

from __future__ import annotations
import importlib
from pathlib import Path
from typing import Callable
import jax.numpy as jnp
from purin.solvers import ImplicitGradientConfig
from purin.solvers import LinearSolverConfig
from purin.solvers import ProjectionConfig
from purin.solvers import prepare_projection
from ohagi.serialization.predictor import make_network_apply

_FACTORIES: dict[str, Callable] = {}


def register(name: str):
    """Register a named callable used when serializing a symbolic problem."""
    name = str(name)
    if not name:
        raise ValueError("registry name must be nonempty")

    def decorator(factory: Callable):
        if name in _FACTORIES and _FACTORIES[name] is not factory:
            raise ValueError(f"factory {name!r} is already registered")
        _FACTORIES[name] = factory
        return factory

    return decorator


def resolve(name: str) -> Callable:
    """Resolve a saved callable identifier through the serialization registry."""
    try:
        return _FACTORIES[str(name)]
    except KeyError as exc:
        raise KeyError(
            f"no KKTHN serialization factory is registered as {name!r}"
        ) from exc


# Built-in symbolic projected-MLP factory.


def _import_factory(path: str):
    from ohagi.compat.paths import canonical_module_name

    module_name, attribute = path.split(":", 1)
    value = getattr(
        importlib.import_module(canonical_module_name(module_name)), attribute
    )
    if not callable(value):
        raise TypeError(f"problem factory {path!r} is not callable")
    return value


def _projection_config(value):
    value = dict(value or {})
    linear = LinearSolverConfig(**value.pop("linear_solver", {}))
    implicit = ImplicitGradientConfig(**value.pop("implicit_gradient", {}))
    return ProjectionConfig(**value, linear_solver=linear, implicit_gradient=implicit)


def projected_mlp_factory(*, params, manifest, root: Path, backend, dtype):
    """Build a predictor from a registered/importable symbolic problem factory.

    The referenced problem factory receives ``factory_config['problem_config']``
    as keyword arguments and must return a projection problem or a problem with
    ``as_projection_problem()``.
    """
    config = dict(manifest.get("factory_config", {}))
    problem_factory = _import_factory(config["problem_factory"])
    problem = problem_factory(**dict(config.get("problem_config", {})))
    projection = prepare_projection(
        problem,
        method="primal-dual-interior-point",
        config=_projection_config(config.get("projection_config", {})),
        batch_size=int(manifest["batch_size"]),
        dtype=dtype,
        backend=backend,
    )
    network_apply = make_network_apply(manifest["network"])
    transform = dict(config.get("input_transform", {"type": "identity"}))
    if transform.get("type", "identity") == "standard":
        mean = jnp.asarray(transform["mean"], dtype=dtype)
        scale = jnp.asarray(transform["scale"], dtype=dtype)

        def transform_fn(x):
            return (x - mean) / scale
    else:

        def transform_fn(x):
            return x

    def predict(inputs):
        raw = network_apply(params, transform_fn(inputs))
        return projection(raw, inputs)

    return predict
