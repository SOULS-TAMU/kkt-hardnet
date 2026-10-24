"""Capture the complete frontend prediction graph for shared JAX export."""

import jax
import numpy as np
from purin.execution.runtime import (
    resolve_device,
    resolve_jax_dtype as resolve_jax_dtype,
)


def kkthn_graph(model, *, device="cpu", state=None):
    """Capture trained KKTHN parameters, transforms and projection in a JAX graph."""
    from dataclasses import replace

    state = model._trained[model._last_method] if state is None else state
    selected = resolve_device(device)
    state = replace(
        state,
        backend=f"{selected.platform}:{selected.id}",
        parameters=jax.device_put(state.parameters, selected),
        gradient_projector=None,
        single_gradient_projector=None,
        single_prediction_core=None,
        single_prediction_fn=None,
        single_value_jacobian_fn=None,
    )
    with jax.default_device(selected):
        one = model._single_prediction_core(state)
    return dict(
        predict_fn=lambda x: one(x[0])[None, :],
        framework="kkthn",
        input_size=len(model.parameter_names),
        output_size=len(model.variable_names),
        dtype=np.dtype(state.dtype).name,
        input_names=list(model.parameter_names),
        output_names=list(model.variable_names),
        example_input=np.asarray(state.input_mean),
    )
