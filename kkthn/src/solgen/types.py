from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class QuadraticProgramData:
    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    d: np.ndarray
    l: Optional[np.ndarray]
    u: Optional[np.ndarray]


@dataclass(frozen=True)
class SolveResult:
    status: str
    y: np.ndarray
    objective: float
    iterations: int
    mode: str
    primal_residual: float
    step_norm: float
    mu: Optional[np.ndarray] = None
    solve_time_sec: Optional[float] = None
