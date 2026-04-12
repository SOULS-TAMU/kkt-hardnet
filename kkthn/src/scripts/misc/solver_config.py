from __future__ import annotations

from typing import Any, Mapping


def resolve_solver_name(data_cfg: Mapping[str, Any], default: str = "SCS") -> str:
    raw = data_cfg.get("solver", data_cfg.get("cvxpy_solver", default))
    solver = str(raw).strip().upper()
    if not solver:
        raise ValueError("Solver name cannot be empty.")
    return solver

