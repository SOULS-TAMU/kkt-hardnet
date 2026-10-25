"""Numerical gradients, historical reporting and every task's real training path."""

import numpy as np
import pytest
from kkthn import KKTHardNet, Projection, MLP, Loss, Problem

pytestmark = pytest.mark.coverage


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("extended-kkthn")
    x = np.linspace(0.4, 1.2, 9)
    data = root / "data.csv"
    np.savetxt(
        data,
        np.column_stack((x, 0.6 * x, 0.4 * x)),
        delimiter=",",
        header="p,u,v",
        comments="",
    )
    p = Projection(method="chambolle-pock")
    t = p.add_parameter(["p"])
    v = p.add_variable(["u", "v"])
    p.constraints.add(v.u + v.v == t.p, v.u >= 0, v.v >= 0, v.u + v.v <= 2)
    model = KKTHardNet(
        "surrogate",
        MLP(hidden_sizes=(4,)),
        Loss.mse(),
        p,
        name="extended",
        train={
            "epochs": 3,
            "epoch_mlp": 1,
            "batch_size": 2,
            "seed": 7,
            "print_every": 0,
            "output_dir": str(root),
            "readiness_extra_epochs": 0,
            "minimum_projection_convergence": 0,
            "objective_learning_rate_schedule": [(1e10, 1e-4)],
        },
    )
    model.dataset(parameters=data, variables=data)
    result = model.model(projection={"max_iterations": 400, "tolerance": 1e-9})
    return model, result, root


def objective(model):
    p = Problem("sum")
    t = p.add_parameter(["p"])
    v = p.add_variable(["u", "v"])
    p.objective = t.p * (v.u + v.v)
    return p


def test_value_and_total_objective_gradients(trained):
    model, _, _ = trained
    p = objective(model)
    name = model.prepare_objective_gradient(p, compile_batch=True, verbose=True)
    assert (
        model.prepare_objective_gradient(p, compile_batch=True, verbose=False) == name
    )
    for batch in [False, True]:
        result = model.objective_gradient(
            [[0.6], [0.7], [0.8]], objective=name, batch=batch
        )
        np.testing.assert_allclose(result.value, [0.36, 0.49, 0.64], atol=2e-6)
        np.testing.assert_allclose(result.df_dx[:, 0], [1.2, 1.4, 1.6], atol=2e-5)
    result = model.objective_gradient([0.7], objective=p, verbose=False)
    np.testing.assert_allclose(result.df_dx, [1.4], atol=2e-5)
    model.prepare_objective_gradient(p, overwrite=True, verbose=False)
    with pytest.raises(RuntimeError, match="Batch objective"):
        model.objective_gradient([[0.5], [0.6]], objective=name, batch=True)
    with pytest.raises(KeyError):
        model.objective_gradient([0.7], objective="missing")
    with pytest.raises(ValueError):
        model.objective_gradient([[0.7, 0.8]], objective=name)
    with pytest.raises(RuntimeError):
        model.objective_gradient([0.7], objective=name, projection_backend="native")
    with pytest.raises(ValueError):
        model.prepare_objective_gradient(p, name=" ")
    p = Problem("empty")
    with pytest.raises(ValueError):
        model.prepare_objective_gradient(p)


def test_prediction_shapes_jacobians_and_exports(trained, tmp_path):
    model, result, _ = trained
    for x in [np.array([0.7]), np.array([[0.7], [0.8], [0.9]])]:
        pred = model.predict(x, gradient=True)
        np.testing.assert_allclose(
            pred.y.sum(axis=-1), np.asarray(x).reshape(-1), atol=1e-6
        )
        np.testing.assert_allclose(pred.dy_dx.sum(axis=-2), 1, atol=2e-5)
    assert model.predict(np.empty((0, 1))).shape == (0, 2)
    assert model.predict(np.empty((0, 1)), gradient=True).dy_dx.shape == (0, 2, 1)
    with pytest.raises(ValueError):
        model.predict(np.ones((1, 2)))
    info = model.export_predictor(directory=tmp_path / "export")
    np.testing.assert_allclose(
        info["predictor"].predict([0.7]), model.predict([0.7]), atol=1e-8
    )
    from kkthn.api.predict import _prepared_projector, _gradient_projector

    state = model._trained[model._last_method]
    old = state.projector
    state.projector = None
    assert _prepared_projector(model, state) is not None
    state.projector = old
    assert _gradient_projector(model, state) is not None
    assert model.load_inference(info["metadata_path"]).predict([0.7]).shape == (2,)
    loaded = KKTHardNet().load(result["metadata_path"])
    np.testing.assert_allclose(loaded.predict([0.7]), model.predict([0.7]), atol=1e-8)


def test_history_and_figures(trained, tmp_path, monkeypatch):
    model, result, _ = trained
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    original = Figure.savefig

    def small(self, *args, **kwargs):
        kwargs["dpi"] = 25
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", small)
    fig, _ = model.plot_history(show=False, save_dir=tmp_path / "loss", bg="white")
    plt.close(fig)
    assert (tmp_path / "loss/training_history.png").stat().st_size > 0
    for row in result["history"]:
        row["inverse_k"] = 1 + row["epoch"] * 0.1
    model.inverse_actual_values = {"k": 1.5}
    fig, _ = model.plot_history(show=True, save_dir=tmp_path / "inverse")
    plt.close(fig)
    history = model._read_history()
    assert "inverse_k" in history
    assert model.summary()["inference_backend"] == "jax"
    rows = result["history"]
    result["history"] = []
    assert model._read_history()["epoch"].size == len(rows)
    result["history"] = rows


@pytest.mark.parametrize("kind", ["optimization", "parameter_estimation"])
def test_additional_task_training(kind, tmp_path):
    x = np.linspace(0.5, 1.2, 8)
    path = tmp_path / "data.csv"
    np.savetxt(
        path,
        np.column_stack((x, 0.6 * x, 0.4 * x)),
        delimiter=",",
        header="p,u,v",
        comments="",
    )
    p = Projection(method="linear-equality")
    t = p.add_parameter(["p"])
    v = p.add_variable(["u", "v"])
    p.constraints.add(v.u + v.v == t.p)
    if kind == "optimization":
        p.objective = (v.u - 0.6 * t.p) ** 2 + (v.v - 0.4 * t.p) ** 2
    loss = (
        Loss.objective()
        if kind == "optimization"
        else Loss.parameter_estimation(observed_variables=("u",))
    )
    model = KKTHardNet(
        kind,
        MLP(hidden_sizes=(3,), predict_duals=False),
        loss,
        p,
        name=kind,
        train={
            "epochs": 2,
            "epoch_mlp": 1,
            "batch_size": 3,
            "seed": 7,
            "print_every": 0,
            "output_dir": str(tmp_path),
            "readiness_extra_epochs": 0,
            "minimum_projection_convergence": 0,
            "drop_last_batch": True,
            "standardize_inputs": False,
            "eta": 1e10,
            "objective_learning_rate_schedule": [(1e10, 1e-4)],
        },
    )
    model.dataset(parameters=path, variables=None if kind == "optimization" else path)
    result = model.optimize() if kind == "optimization" else model.estimate()
    np.testing.assert_allclose(model.predict([0.7]).sum(), 0.7, atol=1e-7)
    assert np.isfinite(result["metrics"]["loss"])
