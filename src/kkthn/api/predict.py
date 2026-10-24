"""Projected prediction and full input-Jacobian inference."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from ohagi.execution.inference import require_jax_backend
from purin.solvers import ProjectionMethod
from purin.solvers import prepare_projection
from kkthn.api.types import PredictionWithGradient
from .composition import projected_model


def _prepared_projector(self, state):
    """Lazily prepare the batched JAX projector."""
    if state.projector is None:
        definition = self._projection_definition(state.method, state.projection_config)
        state.projector = prepare_projection(
            self,
            method=state.method,
            config=definition,
            batch_size=state.batch_size,
            dtype=state.dtype,
            backend=state.backend,
        )
    return state.projector


def _predict_batch_jax(self, state, X, *, projector=None):
    projector = state.projector if projector is None else projector
    model = projected_model(
        self,
        state.backbone,
        state.input_mean,
        state.input_scale,
        state.dtype,
        projector,
    )
    outputs, _ = model.apply(state.parameters, {}, X)
    return outputs["projected"]


def _gradient_projector(self, state):
    if state.gradient_projector is not None:
        return state.gradient_projector
    selected = ProjectionMethod.normalize(state.method)
    if selected is ProjectionMethod.LINEAR_EQUALITY:
        state.gradient_projector = _prepared_projector(self, state)
        return state.gradient_projector
    values = dict(state.projection_config)
    implicit = dict(values.get("implicit_gradient", {}))
    implicit["differentiate_problem_parameters"] = True
    values["implicit_gradient"] = implicit
    definition = self._projection_definition(state.method, values)
    state.gradient_projector = prepare_projection(
        self,
        method=state.method,
        config=definition,
        batch_size=state.batch_size,
        dtype=state.dtype,
        backend=state.backend,
    )
    return state.gradient_projector


def _single_gradient_projector(self, state):
    """Prepare and cache a batch-size-one differentiable projector."""
    if state.single_gradient_projector is not None:
        return state.single_gradient_projector

    values = dict(state.projection_config)
    selected = ProjectionMethod.normalize(state.method)
    if selected is not ProjectionMethod.LINEAR_EQUALITY:
        implicit = dict(values.get("implicit_gradient", {}))
        implicit["differentiate_problem_parameters"] = True
        values["implicit_gradient"] = implicit

    definition = self._projection_definition(state.method, values)
    state.single_gradient_projector = prepare_projection(
        self,
        method=state.method,
        config=definition,
        batch_size=1,
        dtype=state.dtype,
        backend=state.backend,
    )
    return state.single_gradient_projector


def _single_prediction_core(self, state):
    """Return the shared un-jitted batch-size-one forward map."""
    if state.single_prediction_core is not None:
        return state.single_prediction_core

    projector = self._single_gradient_projector(state)

    def predict_one(x):
        return self._predict_batch_jax(state, x[None, :], projector=projector)[0]

    state.single_prediction_core = predict_one
    return state.single_prediction_core


def _single_prediction_function(self, state):
    """Return the compiled batch-size-one forward prediction map."""
    if state.single_prediction_fn is not None:
        return state.single_prediction_fn

    state.single_prediction_fn = jax.jit(self._single_prediction_core(state))
    return state.single_prediction_fn


def _single_value_jacobian_function(self, state):
    """Return the projected value and its total input Jacobian."""
    if state.single_value_jacobian_fn is not None:
        return state.single_value_jacobian_fn

    predict_one = self._single_prediction_core(state)

    def prediction_with_aux(x):
        prediction = predict_one(x)
        return prediction, prediction

    jacobian_with_value = jax.jacrev(prediction_with_aux, has_aux=True)

    def value_and_jacobian(x):
        jacobian, prediction = jacobian_with_value(x)
        return prediction, jacobian

    state.single_value_jacobian_fn = jax.jit(value_and_jacobian)
    return state.single_value_jacobian_fn


def predict(self, values, *, method=None, projection_backend=None, gradient=False):
    """Predict projected outputs, optionally with their total input Jacobian.

    Parameters
    ----------
    values : array_like
        Input vector of shape ``(n_parameters,)`` or a batch ``(n, n_parameters)``.
    method : str, optional
        Trained projection method; defaults to the most recently selected method.
    projection_backend : {"jax", "auto"}, optional
        JAX is the only inference engine. ``auto`` is a compatibility alias.
    gradient : bool, optional
        Return ``PredictionWithGradient`` using the existing JAX differentiation
        path.

    Returns
    -------
    numpy.ndarray or PredictionWithGradient
        Projected outputs, with the input Jacobian when requested. A vector
        input produces a vector output; a batch preserves its leading dimension.
    """
    require_jax_backend(projection_backend)
    selected = (
        ProjectionMethod.normalize(method).value
        if method is not None
        else self._last_method
    )
    if selected is None or selected not in self._trained:
        raise RuntimeError("Train or load a model before calling predict().")
    state = self._trained[selected]
    X = np.asarray(values, dtype=np.dtype(state.dtype))
    squeeze = X.ndim == 1
    if squeeze:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != len(self.parameter_names):
        raise ValueError(
            f"Expected input shape (n, {len(self.parameter_names)}); got {X.shape}."
        )

    if X.shape[0] == 0:
        prediction = np.empty(
            (0, len(self.variable_names)), dtype=np.dtype(state.dtype)
        )
        if gradient:
            jacobian = np.empty(
                (*prediction.shape, len(self.parameter_names)), dtype=prediction.dtype
            )
            return PredictionWithGradient(prediction, jacobian)
        return prediction
    if gradient:
        value_jacobian_fn = self._single_value_jacobian_function(state)
        values_device = []
        jacobians_device = []

        for row in X:
            value_device, jacobian_device = value_jacobian_fn(
                jnp.asarray(row, dtype=state.dtype)
            )
            values_device.append(value_device)
            jacobians_device.append(jacobian_device)

        prediction_host, jacobian_host = jax.device_get(
            (values_device, jacobians_device)
        )
        prediction = np.stack([np.asarray(value) for value in prediction_host])
        jacobian = np.stack([np.asarray(value) for value in jacobian_host])

        if squeeze:
            return PredictionWithGradient(prediction[0], jacobian[0])
        return PredictionWithGradient(prediction, jacobian)

    # Use the exact same single-instance forward map, but without
    # activating jacrev or the implicit backward pass.
    single_prediction_fn = self._single_prediction_function(state)
    values_device = [
        single_prediction_fn(jnp.asarray(row, dtype=state.dtype)) for row in X
    ]
    prediction_host = jax.device_get(values_device)
    prediction = np.stack([np.asarray(value) for value in prediction_host])

    return prediction[0] if squeeze else prediction
