"""Restricted symbolic decoding, sparse resources and registry validation."""

from copy import deepcopy
import numpy as np
import pytest
from purin.problems import Problem
from kkthn.serialization.problem import (
    _ExpressionDecoder,
    _pattern_from_csr,
    load_sparse_structure,
    load_symbolic_problem,
    symbolic_problem_spec,
    save_problem,
    save_sparse_structure,
)
from kkthn.serialization.registry import (
    register,
    resolve,
    _import_factory,
    _projection_config,
)


@pytest.mark.parametrize(
    "text", ["x", '"a"', "x[0]", "~u", "u%2", "foo(u)", "sin(u,u)", "sin(x=u)", "u +"]
)
def test_invalid_symbolic_expressions(text):
    p = Problem()
    p.add_variable(["u"])
    with pytest.raises(ValueError):
        _ExpressionDecoder(p).decode(text)


def test_expression_roundtrip_and_sparse_patterns(tmp_path):
    p = Problem("arithmetic")
    t = p.add_parameter(["p"])
    v = p.add_variable(["u", "v"])
    p.objective = (
        p.sin(v.u)
        + p.cos(v.v)
        + p.exp(t.p)
        + p.log(v.u)
        + p.sqrt(v.v)
        + p.abs(v.u)
        + (v.u - v.v) * 2 / (t.p + 1)
        + v.u**2
    )
    p.constraints.add(v.u + v.v == t.p, v.u >= 0)
    spec = symbolic_problem_spec(p)
    restored = load_symbolic_problem(spec)
    a = p.compile()
    b = restored.compile()
    np.testing.assert_allclose(
        a.objective(np.array([1.0]), np.array([0.4, 0.6])),
        b.objective(np.array([1.0]), np.array([0.4, 0.6])),
    )
    path = tmp_path / "sparse.npz"
    manifest = save_sparse_structure(path, a)
    assert load_sparse_structure(path, manifest) == (
        a.equality_sparsity,
        a.inequality_sparsity,
    )
    for field in ["equality_size", "inequality_size"]:
        wrong = {**manifest, field: 100}
        with pytest.raises(ValueError):
            load_sparse_structure(path, wrong)
    with pytest.raises(ValueError):
        load_sparse_structure(path, {"format": "bad"})
    with pytest.raises(ValueError):
        _pattern_from_csr([], [])
    with pytest.raises(ValueError):
        load_symbolic_problem({"format": "bad"})
    wrong = deepcopy(spec)
    wrong["constraints"][0]["kind"] = "bad"
    with pytest.raises(ValueError):
        load_symbolic_problem(wrong)
    with pytest.raises(TypeError):
        save_problem(tmp_path / "problem.npz", p)


def test_registry_contracts():
    def factory():
        return 1

    assert register("coverage-test")(factory) is factory
    assert register("coverage-test")(factory) is factory
    assert resolve("coverage-test")() == 1
    with pytest.raises(ValueError):
        register("coverage-test")(lambda: 2)
    with pytest.raises(ValueError):
        register("")
    with pytest.raises(KeyError):
        resolve("unknown-factory")
    assert callable(_import_factory("numpy:sin"))
    with pytest.raises(TypeError):
        _import_factory("numpy:pi")
    assert (
        _projection_config(
            {"linear_solver": {"max_iterations": 7}}
        ).linear_solver.max_iterations
        == 7
    )
