"""Private runtime state and compiled-function caches."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from ohagi.nn import MLP
from ohagi.nn import MLPBackbone


@dataclass
class _TrainedState:
    method: str
    backbone: MLPBackbone
    architecture: MLP
    parameters: Any
    projector: Any
    input_mean: np.ndarray
    input_scale: np.ndarray
    dtype: Any
    batch_size: int
    backend: str
    projection_config: dict[str, Any]
    result: dict[str, Any]
    inference_backend: str = "jax"
    metadata_path: str | None = None
    gradient_projector: Any | None = None
    single_gradient_projector: Any | None = None
    single_prediction_core: Any | None = None
    single_prediction_fn: Any | None = None
    single_value_jacobian_fn: Any | None = None
    objective_gradient_fns: dict[str, Any] | None = None


__all__ = ["_TrainedState"]
