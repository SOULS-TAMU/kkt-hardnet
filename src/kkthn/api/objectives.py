"""Single and batched scalar-objective gradients through projection."""

from __future__ import annotations
from typing import Mapping
import jax
import jax.numpy as jnp
import numpy as np
from purin.solvers import ProjectionMethod
from kkthn.api.data import _fixed_batches
from kkthn.api.types import ObjectiveGradientResult


def _validate_objective_problem(self, objective_problem):
    """Validate a post-prediction objective against this model."""
    if objective_problem.objective is None:
        raise ValueError("The objective problem must define an objective.")

    objective_parameters = tuple(objective_problem.parameter_names)
    model_parameters = tuple(self.parameter_names)
    if objective_parameters != model_parameters:
        raise ValueError(
            "Objective parameter names and ordering must match the model. "
            f"Expected {model_parameters}; got {objective_parameters}."
        )

    objective_variables = tuple(objective_problem.variable_names)
    model_variables = tuple(self.variable_names)
    if objective_variables != model_variables:
        raise ValueError(
            "Objective variable names and ordering must match the model. "
            f"Expected {model_variables}; got {objective_variables}."
        )


def _compile_single_objective_gradient(self, state, objective_function):
    """Compile one scalar objective-gradient evaluation."""
    predict_one = self._single_prediction_core(state)

    def scalar_objective_with_aux(x):
        y_tilde = predict_one(x)
        value = objective_function(x, y_tilde)
        return value, y_tilde

    value_and_gradient = jax.value_and_grad(scalar_objective_with_aux, has_aux=True)

    def evaluate(x):
        (value, y_tilde), df_dx = value_and_gradient(x)
        return value, y_tilde, df_dx

    compiled = jax.jit(evaluate)

    dummy_x = jnp.asarray(state.input_mean, dtype=state.dtype)
    value_device, y_device, gradient_device = compiled(dummy_x)
    jax.block_until_ready(value_device)
    jax.block_until_ready(y_device)
    jax.block_until_ready(gradient_device)
    return compiled


def _compile_batch_objective_gradient(self, state, objective_function):
    """Compile a true fixed-batch scalar objective-gradient evaluation.

    The objective values are summed only to provide one scalar seed for
    reverse-mode differentiation. Because every projected row depends only
    on the corresponding input row, the returned gradient has shape
    ``(batch_size, n_x)`` and contains one independent ``df_i/dx_i`` per
    sample.
    """
    projector = self._gradient_projector(state)

    def batch_objective_with_aux(X):
        y_tilde = self._predict_batch_jax(state, X, projector=projector)
        objective_values = jax.vmap(objective_function)(X, y_tilde)

        total_objective = jnp.sum(objective_values)
        return total_objective, (objective_values, y_tilde)

    value_and_gradient = jax.value_and_grad(batch_objective_with_aux, has_aux=True)

    def evaluate_batch(X):
        (_total_objective, (objective_values, y_tilde)), df_dX = value_and_gradient(X)

        return objective_values, y_tilde, df_dX

    compiled = jax.jit(evaluate_batch)

    dummy_single = jnp.asarray(state.input_mean, dtype=state.dtype)
    dummy_batch = jnp.broadcast_to(
        dummy_single, (state.batch_size, len(self.parameter_names))
    )

    values_device, y_device, gradients_device = compiled(dummy_batch)
    jax.block_until_ready(values_device)
    jax.block_until_ready(y_device)
    jax.block_until_ready(gradients_device)
    return compiled


def prepare_objective_gradient(
    self,
    objective_problem,
    *,
    name=None,
    method=None,
    verbose=True,
    overwrite=False,
    compile_batch=False,
):
    """Compile and cache scalar and optional batch objective gradients.

    The objective Problem must use the model's parameter and variable names
    in the same order. Its scalar objective may depend on both ``x`` and
    the projected output ``y_tilde``.

    The single-instance function is always prepared. Set
    ``compile_batch=True`` to additionally prepare a true fixed-batch
    reverse pass using the model's saved projection batch size.
    """
    selected = (
        ProjectionMethod.normalize(method).value
        if method is not None
        else self._last_method
    )
    if selected is None or selected not in self._trained:
        raise RuntimeError(
            "Train or load a model before preparing an objective gradient."
        )

    state = self._trained[selected]
    self._validate_objective_problem(objective_problem)

    objective_name = str(name if name is not None else objective_problem.name).strip()
    if not objective_name:
        raise ValueError("Objective cache name must not be empty.")

    if state.objective_gradient_fns is None:
        state.objective_gradient_fns = {}

    existing = state.objective_gradient_fns.get(objective_name)
    if existing is not None and not isinstance(existing, Mapping):
        # Compatibility with the earlier single-function cache format.
        existing = {"single": existing, "batch": None, "batch_size": state.batch_size}

    if overwrite or existing is None:
        entry = {"single": None, "batch": None, "batch_size": state.batch_size}
    else:
        entry = dict(existing)

    needs_single = entry.get("single") is None
    needs_batch = bool(compile_batch and entry.get("batch") is None)

    if not needs_single and not needs_batch:
        state.objective_gradient_fns[objective_name] = entry
        return objective_name

    canonical = objective_problem.compile()
    objective_function = canonical.objective

    if needs_single:
        single_compiled = self._compile_single_objective_gradient(
            state, objective_function
        )
        entry["single"] = single_compiled

        if verbose:
            print(
                f"Compiled objective gradient '{objective_name}' "
                f"(single JIT cache size: "
                f"{single_compiled._cache_size()})."
            )

    if needs_batch:
        batch_compiled = self._compile_batch_objective_gradient(
            state, objective_function
        )
        entry["batch"] = batch_compiled
        entry["batch_size"] = state.batch_size

        if verbose:
            print(
                f"Compiled batch objective gradient "
                f"'{objective_name}' "
                f"(batch_size={state.batch_size}, "
                f"JIT cache size: "
                f"{batch_compiled._cache_size()})."
            )

    state.objective_gradient_fns[objective_name] = entry
    return objective_name


def objective_gradient(
    self,
    values,
    *,
    objective,
    name=None,
    method=None,
    projection_backend="jax",
    verbose=True,
    batch=False,
):
    """Evaluate a scalar objective and total ``df/dx``.

    ``objective`` may be a prepared cache-name string or a compatible
    Problem object.

    With ``batch=False`` (the default), each row uses the efficient cached
    single-instance scalar VJP. With ``batch=True`` and more than one row,
    inputs are evaluated with the compiled fixed-batch projector and one
    batched reverse pass per physical batch.
    """
    if str(projection_backend).strip().lower() not in {"auto", "jax"}:
        raise RuntimeError("Objective gradients require the JAX projection backend.")

    selected = (
        ProjectionMethod.normalize(method).value
        if method is not None
        else self._last_method
    )
    if selected is None or selected not in self._trained:
        raise RuntimeError("Train or load a model before calling objective_gradient().")
    state = self._trained[selected]

    if isinstance(objective, str):
        objective_name = objective
    else:
        objective_name = self.prepare_objective_gradient(
            objective, name=name, method=selected, verbose=verbose, compile_batch=batch
        )

    functions = state.objective_gradient_fns or {}
    if objective_name not in functions:
        raise KeyError(f"Objective gradient {objective_name!r} is not prepared.")

    entry = functions[objective_name]
    if not isinstance(entry, Mapping):
        entry = {"single": entry, "batch": None, "batch_size": state.batch_size}

    single_compiled = entry.get("single")
    batch_compiled = entry.get("batch")

    if single_compiled is None:
        raise RuntimeError(
            f"Single objective gradient {objective_name!r} is not compiled."
        )

    X = np.asarray(values, dtype=np.dtype(state.dtype))
    squeeze = X.ndim == 1
    if squeeze:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != len(self.parameter_names):
        raise ValueError(
            f"Expected input shape (n, {len(self.parameter_names)}); got {X.shape}."
        )

    # Keep the existing very efficient scalar pathway for one instance,
    # even when batch=True was supplied.
    use_batch = bool(batch and len(X) > 1)

    if use_batch:
        if batch_compiled is None:
            raise RuntimeError(
                f"Batch objective gradient {objective_name!r} "
                "is not compiled. Call "
                "prepare_objective_gradient(..., "
                "compile_batch=True) first."
            )

        objective_rows = []
        prediction_rows = []
        gradient_rows = []

        for xb, _yb, _mask, physical_count in _fixed_batches(
            X, None, state.batch_size, shuffle=False, seed=0
        ):
            (objective_device, prediction_device, gradient_device) = batch_compiled(
                jnp.asarray(xb, dtype=state.dtype)
            )

            (objective_host, prediction_host, gradient_host) = jax.device_get(
                (objective_device, prediction_device, gradient_device)
            )

            objective_rows.append(np.asarray(objective_host)[:physical_count])
            prediction_rows.append(np.asarray(prediction_host)[:physical_count])
            gradient_rows.append(np.asarray(gradient_host)[:physical_count])

        objective_values = np.concatenate(objective_rows, axis=0)
        predictions = np.concatenate(prediction_rows, axis=0)
        gradients = np.concatenate(gradient_rows, axis=0)
    else:
        values_device = []
        predictions_device = []
        gradients_device = []

        for row in X:
            (value_device, prediction_device, gradient_device) = single_compiled(
                jnp.asarray(row, dtype=state.dtype)
            )
            values_device.append(value_device)
            predictions_device.append(prediction_device)
            gradients_device.append(gradient_device)

        (values_host, predictions_host, gradients_host) = jax.device_get(
            (values_device, predictions_device, gradients_device)
        )

        objective_values = np.stack([np.asarray(value) for value in values_host])
        predictions = np.stack([np.asarray(value) for value in predictions_host])
        gradients = np.stack([np.asarray(value) for value in gradients_host])

    if squeeze:
        return ObjectiveGradientResult(
            objective_values[0], predictions[0], gradients[0]
        )
    return ObjectiveGradientResult(objective_values, predictions, gradients)
