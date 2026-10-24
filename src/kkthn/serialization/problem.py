"""Serialization for symbolic and structured optimization problems.

The high-level artifact format stores symbolic residual expressions together
with the exact row-support sparsity patterns used by the JAX projection
backends. Loading reconstructs the frontend problem and reinstates the saved
patterns before preparing the selected sparse solver.
"""

from __future__ import annotations
import ast
from pathlib import Path
from typing import Any, Mapping
import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from purin.problems import ParametricQP
from purin.problems import Problem
from ohagi.serialization.problem import (
    _save_matrix as _save_matrix,
    _load_matrix as _load_matrix,
)
from ohagi.serialization.problem import save_problem as _save_structured_problem
from ohagi.serialization.problem import load_problem as _load_structured_problem


def _pattern_to_csr(pattern) -> tuple[np.ndarray, np.ndarray]:
    indptr = [0]
    indices: list[int] = []
    for support in pattern or ():
        indices.extend(int(value) for value in support)
        indptr.append(len(indices))
    return np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int64)


def _pattern_from_csr(indptr, indices) -> tuple[tuple[int, ...], ...]:
    indptr = np.asarray(indptr, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indptr.size == 0 or int(indptr[0]) != 0 or int(indptr[-1]) != indices.size:
        raise ValueError("invalid saved sparse-pattern CSR arrays")
    return tuple(
        tuple(int(value) for value in indices[indptr[row] : indptr[row + 1]])
        for row in range(indptr.size - 1)
    )


def save_sparse_structure(path: str | Path, canonical) -> dict[str, Any]:
    """Save exact canonical row-support patterns for sparse prediction."""
    eq_indptr, eq_indices = _pattern_to_csr(canonical.equality_sparsity)
    ineq_indptr, ineq_indices = _pattern_to_csr(canonical.inequality_sparsity)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        equality_indptr=eq_indptr,
        equality_indices=eq_indices,
        inequality_indptr=ineq_indptr,
        inequality_indices=ineq_indices,
    )
    return {
        "format": "kkthn_sparse_structure_v1",
        "file": path.name,
        "variable_size": int(canonical.variable_size),
        "equality_size": int(canonical.equality_size),
        "inequality_size": int(canonical.inequality_size),
        "equality_nonzeros": int(eq_indices.size),
        "inequality_nonzeros": int(ineq_indices.size),
    }


def load_sparse_structure(path: str | Path, manifest: Mapping[str, Any]):
    """Load row-support patterns saved by :func:`save_sparse_structure`."""
    if manifest.get("format") != "kkthn_sparse_structure_v1":
        raise ValueError("unsupported sparse structure format")
    with np.load(Path(path), allow_pickle=False) as arrays:
        equality = _pattern_from_csr(
            arrays["equality_indptr"], arrays["equality_indices"]
        )
        inequality = _pattern_from_csr(
            arrays["inequality_indptr"], arrays["inequality_indices"]
        )
    if len(equality) != int(manifest["equality_size"]):
        raise ValueError("saved equality sparse-pattern row count mismatch")
    if len(inequality) != int(manifest["inequality_size"]):
        raise ValueError("saved inequality sparse-pattern row count mismatch")
    return equality, inequality


def symbolic_problem_spec(problem) -> dict[str, Any]:
    """Return a JSON-safe symbolic frontend specification."""
    return {
        "format": "kkthn_symbolic_problem_v1",
        "name": str(problem.name),
        "parameters": list(problem.parameter_names),
        "variables": list(problem.variable_names),
        "objective": None if problem.objective is None else problem.objective.text,
        "constraints": [
            {
                "kind": constraint.kind,
                "text": constraint.text,
                "residual": constraint.residual.text,
            }
            for constraint in problem.constraints
        ],
    }


class _ExpressionDecoder:
    """Restricted arithmetic decoder for saved symbolic residual strings."""

    def __init__(self, problem: Problem):
        self.problem = problem
        self.symbols = {
            name: getattr(problem.parameter, name) for name in problem.parameter_names
        }
        self.symbols.update(
            {name: getattr(problem.variable, name) for name in problem.variable_names}
        )
        self.functions = {
            "sin": problem.sin,
            "cos": problem.cos,
            "exp": problem.exp,
            "log": problem.log,
            "sqrt": problem.sqrt,
            "abs": problem.abs,
        }

    def decode(self, text: str):
        """Reconstruct a symbolic expression from its serialized definition."""
        try:
            node = ast.parse(str(text), mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"invalid saved symbolic expression: {text!r}") from error
        return self._node(node)

    def _node(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("only numeric constants are allowed in saved expressions")
        if isinstance(node, ast.Name):
            if node.id not in self.symbols:
                raise ValueError(f"unknown saved symbolic name {node.id!r}")
            return self.symbols[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = self._node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError("unsupported unary operator in saved expression")
        if isinstance(node, ast.BinOp):
            left = self._node(node.left)
            right = self._node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("unsupported binary operator in saved expression")
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in self.functions
            ):
                raise ValueError("unsupported function in saved expression")
            if len(node.args) != 1 or node.keywords:
                raise ValueError(
                    "saved symbolic functions require one positional argument"
                )
            return self.functions[node.func.id](self._node(node.args[0]))
        raise ValueError(
            f"unsupported syntax {type(node).__name__} in saved symbolic expression"
        )


def load_symbolic_problem(spec: Mapping[str, Any]) -> Problem:
    """Reconstruct a public symbolic problem without user constraint code."""
    if spec.get("format") != "kkthn_symbolic_problem_v1":
        raise ValueError("unsupported symbolic problem format")
    problem = Problem(str(spec.get("name", "problem")))
    problem.add_parameter(list(spec.get("parameters", [])))
    problem.add_variable(list(spec.get("variables", [])))
    decoder = _ExpressionDecoder(problem)
    objective = spec.get("objective")
    if objective is not None:
        problem.objective = decoder.decode(str(objective))
    for item in spec.get("constraints", []):
        residual = decoder.decode(str(item["residual"]))
        kind = str(item["kind"])
        text = str(item.get("text", "")) or None
        if kind == "eq":
            problem.constraints.equality(residual, name=text)
        elif kind == "ineq":
            problem.constraints.inequality(residual, name=text)
        else:
            raise ValueError(f"unsupported saved constraint kind {kind!r}")
    return problem


def save_problem(path: str | Path, problem) -> dict[str, Any]:
    """Save KKTHN's symbolic problem definition and registered resources."""
    if not isinstance(problem, ParametricQP):
        raise TypeError(
            "portable problem serialization currently supports ParametricQP; "
            "symbolic high-level runs use symbolic_problem_spec in metadata"
        )
    return _save_structured_problem(path, problem)


def load_problem(path: str | Path, manifest: dict[str, Any]):
    """Restore a KKTHN problem from its saved symbolic definition."""
    return _load_structured_problem(path, manifest)


__all__ = [
    "load_problem",
    "load_sparse_structure",
    "load_symbolic_problem",
    "save_problem",
    "save_sparse_structure",
    "symbolic_problem_spec",
]
