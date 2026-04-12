from __future__ import annotations

from typing import Any

import numpy as np


def default_ineq_multipliers(n_ineq: int, *, fill_value: float = np.nan) -> np.ndarray:
    n_ineq = int(n_ineq)
    if n_ineq <= 0:
        return np.zeros((0,), dtype=np.float64)
    return np.full((n_ineq,), fill_value, dtype=np.float64)


def coerce_ineq_multipliers(
    raw: Any,
    n_ineq: int,
    *,
    fill_value: float = np.nan,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    n_ineq = int(n_ineq)
    if n_ineq <= 0:
        return np.zeros((0,), dtype=np.float64)
    if raw is None:
        return default_ineq_multipliers(n_ineq, fill_value=fill_value)

    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return default_ineq_multipliers(n_ineq, fill_value=fill_value)
    if arr.size != n_ineq:
        raise ValueError(f"Expected {n_ineq} inequality multipliers, received {arr.size}.")

    out = arr.astype(np.float64, copy=True)
    if clip_nonnegative:
        finite = np.isfinite(out)
        out[finite] = np.maximum(out[finite], 0.0)
    return out
