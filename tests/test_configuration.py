"""Configuration, symbolic projection and differentiable facade contracts."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from kkthn import KKTHardNet, Projection
from kkthn.api.configuration import (
    _normalize_bool_option,
    _normalize_train_config,
    _normalize_mapping,
    _projection_definition,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        (" YES ", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("off", False),
        ("0", False),
    ],
)
def test_boolean_options(value, expected):
    assert _normalize_bool_option("pretrain", value) is expected


@pytest.mark.parametrize("value", [None, 1, "maybe", [], {}])
def test_ambiguous_boolean_rejected(value):
    with pytest.raises(TypeError):
        _normalize_bool_option("pretrain", value)


@pytest.mark.parametrize("gamma", [0, -1, np.inf, np.nan])
def test_invalid_dual_gamma(gamma):
    with pytest.raises(ValueError):
        _normalize_train_config({"dual_pretrain_gamma": gamma})


def test_train_defaults_and_removed_controls():
    c = _normalize_train_config(
        {"epochs": 8, "pretrain": "off", "fb_dual_warm_start": True}
    )
    assert c["epoch_mlp"] == 2 and c["dtype"] == "float64" and not c["pretrain"]
    assert "fb_dual_warm_start" not in c
    assert _normalize_mapping(None) == {}
    with pytest.raises(TypeError):
        _normalize_mapping(3)


@pytest.mark.parametrize(
    "method",
    [
        "linear-equality",
        "chambolle-pock",
        "admm",
        "fischer-burmeister-newton",
        "fischer-burmeister-reduced",
        "primal-dual-interior-point",
    ],
)
def test_shared_projection_configurations(method):
    c = _projection_definition(
        None, method, {"max_iterations": 37, "tolerance": 1e-8, "unknown": 4}
    )
    if method == "linear-equality":
        assert c.regularization == 0.0
    else:
        assert c.max_iterations == 37 and c.tolerance == 1e-8


def test_symbolic_projection_and_context_jacobian():
    projection = Projection(method="linear-equality")
    p = projection.add_parameter(["p"])
    y = projection.add_variable(["u", "v"])
    projection.constraints.add(y.u + y.v == p.p)
    model = KKTHardNet(projection=projection)
    problem = model.as_projection_problem()
    from purin.solvers.api import prepare_projection

    layer = prepare_projection(problem, method="linear-equality", batch_size=1)
    target = jnp.array([[0.8, 0.7]])

    def apply(context):
        return layer(target, jnp.concatenate((target, context), axis=1))

    np.testing.assert_allclose(apply(jnp.ones((1, 1))), [[0.55, 0.45]], atol=1e-8)
    np.testing.assert_allclose(
        jax.jacrev(apply)(jnp.ones((1, 1))).reshape(-1), [0.5, 0.5], atol=1e-8
    )
