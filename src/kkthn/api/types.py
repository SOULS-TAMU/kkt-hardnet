"""Public high-level API value types."""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset source and column-role specification for KKTHN training."""

    parameters_path: str
    variables_path: str | None = None


class PredictionWithGradient(NamedTuple):
    """Prediction and projected total Jacobian ``dy/dx``."""

    y: np.ndarray
    dy_dx: np.ndarray


class ObjectiveGradientResult(NamedTuple):
    """Scalar objective value, projected prediction, and total ``df/dx``."""

    value: np.ndarray
    y: np.ndarray
    df_dx: np.ndarray


__all__ = ["DatasetSpec", "PredictionWithGradient", "ObjectiveGradientResult"]
