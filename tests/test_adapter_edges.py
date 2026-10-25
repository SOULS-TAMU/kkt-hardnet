"""Stable logging, sparse-bound inference and public objective validation edges."""

from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import pytest
from kkthn import KKTHardNet, Projection
from kkthn.api import logging
from kkthn.api.network import _network_primal_bounds
from kkthn.api.execution import run_epoch
from kkthn.api.objectives import _validate_objective_problem
from kkthn.serialization.problem import _ExpressionDecoder
from purin.problems import Problem


def test_logging_and_epoch_adapter(capsys):
    logging.print_postprocessing_finish(include_citation=False)
    logging.print_postprocessing_finish(include_citation=True)
    assert "@article" in capsys.readouterr().out

    def step(p, o, b):
        return p + 1, o, {"loss": jnp.array(1.0)}

    batches = [(np.ones((2, 1)), None, np.ones(2, dtype=bool), 2)]
    result = run_epoch(0, 0, batches, step, jnp.float64)
    assert result.state[0] == 1


def test_nonlinear_rows_do_not_become_network_box_bounds():
    projection = Projection()
    t = projection.add_parameter(["p"])
    v = projection.add_variable(["u", "v"])
    projection.constraints.add(v.u >= t.p, v.v <= 2, v.u + v.v <= 3, v.u**2 <= 4)
    model = KKTHardNet(projection=projection)
    low, high = _network_primal_bounds(model, np.array([0.5]), jnp.float64)
    assert low[0] == 0.5 and high[1] == 2
    low, high = _network_primal_bounds(model, np.array([[0.5], [0.7]]), jnp.float64)
    np.testing.assert_array_equal(low[:, 0], [0.5, 0.7])
    with pytest.raises(ValueError):
        _network_primal_bounds(model, np.zeros((1, 1, 1)), jnp.float64)


def test_objective_schema_names_and_decoder_positive_sign():
    m = KKTHardNet()
    m.add_parameter(["p"])
    m.add_variable(["y"])
    p = Problem()
    p.add_parameter(["x"])
    v = p.add_variable(["y"])
    p.objective = v.y
    with pytest.raises(ValueError):
        _validate_objective_problem(m, p)
    p = Problem()
    p.add_parameter(["p"])
    v = p.add_variable(["other"])
    p.objective = v.other
    with pytest.raises(ValueError):
        _validate_objective_problem(m, p)
    assert _ExpressionDecoder(p).decode("+other").text == "other"
    with pytest.raises(ValueError):
        KKTHardNet("optimization")._validate_composition(allow_incomplete=False)


def test_readiness_phase_stops_at_convergence():
    from kkthn.api.fit import _ensure_projection_readiness, _scheduled_rate

    def evaluate(parameters, *args):
        return None, None, {"converged_fraction": float(parameters >= 1)}

    def step(parameters, optimizer_state, batch):
        return parameters + 1, optimizer_state + 1, {"loss": jnp.array(0.0)}

    frontend = SimpleNamespace(
        _evaluate_projector=evaluate, _raw_steps=lambda *args: (step, None)
    )
    prepared = {
        "Xtr": np.ones((2, 1)),
        "Ytr": None,
        "dtype": jnp.float64,
        "backbone": None,
        "mean": None,
        "scale": None,
    }
    config = {
        "pretrain": True,
        "minimum_projection_convergence": 1.0,
        "batch_size": 2,
        "seed": 7,
        "drop_last_batch": False,
        "dual_pretrain_gamma": 0.001,
        "readiness_check_every": 1,
        "readiness_extra_epochs": 5,
    }
    parameters, state, report = _ensure_projection_readiness(
        frontend, prepared, config, None, 0, 0, None, None, "chambolle-pock"
    )
    assert parameters == state == 1 and report["converged_fraction"] == 1
    config["readiness_extra_epochs"] = 0
    with pytest.warns(RuntimeWarning, match="converged for only"):
        parameters, state, report = _ensure_projection_readiness(
            frontend, prepared, config, None, 0, 0, None, None, "chambolle-pock"
        )
    assert parameters == state == 0
    config["pretrain"] = False
    assert (
        _ensure_projection_readiness(
            frontend, prepared, config, None, 0, 0, None, None, "chambolle-pock"
        )[2]
        is None
    )
    assert _scheduled_rate(0.01, [(1.0, 0.005), (0.5, 0.001)], 0.2) == 0.001
    assert _scheduled_rate(0.01, [(1.0, 0.005)], np.nan) == 0.01
