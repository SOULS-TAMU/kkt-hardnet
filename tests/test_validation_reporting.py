"""Invalid configurations, batching edge cases and persisted-history contracts."""

from dataclasses import dataclass
import numpy as np
import pytest
from kkthn import KKTHardNet, Projection, Loss
from kkthn.api.data import _read_named_csv, _fixed_batches
from kkthn.api.configuration import _normalize_mapping, _projection_definition
from kkthn.api import reporting, fit


def test_csv_validation(tmp_path):
    p = tmp_path / "data.csv"
    with pytest.raises(FileNotFoundError):
        _read_named_csv(p, ["a"])
    for text, match in [
        ("", "header"),
        ("b\n1\n", "columns"),
        ("a\n", "rows"),
        ("a\nnan\n", "NaN"),
    ]:
        p.write_text(text)
        with pytest.raises(ValueError, match=match):
            _read_named_csv(p, ["a"])
    p.write_text("a,b\n1,2\n3,4\n")
    np.testing.assert_array_equal(_read_named_csv(p, ["b", "a"]), [[2, 1], [4, 3]])


def test_batches_padding_drop_and_shuffle():
    x = np.arange(5)[:, None]
    batches = list(_fixed_batches(x, x, 3, shuffle=False, seed=0))
    np.testing.assert_array_equal(batches[-1][2], [True, True, False])
    assert batches[-1][3] == 2
    assert (
        len(list(_fixed_batches(x, None, 3, shuffle=True, seed=3, drop_last=True))) == 1
    )
    assert list(_fixed_batches(x[:0], None, 3, shuffle=False, seed=0)) == []
    with pytest.raises(ValueError):
        list(_fixed_batches(x, None, 0, shuffle=False, seed=0))


def test_composition_and_data_contracts(tmp_path):
    m = KKTHardNet("historical-name")
    assert m.name == "historical-name"
    with pytest.raises(ValueError):
        KKTHardNet("surrogate", loss=Loss.objective())
    with pytest.raises(ValueError):
        KKTHardNet("optimization", loss=Loss.mse())
    with pytest.raises(ValueError):
        KKTHardNet("parameter_estimation", loss=Loss.objective())
    with pytest.raises(ValueError):
        m.optimize()
    with pytest.raises(ValueError):
        m.estimate()
    with pytest.raises(ValueError):
        m._load_dataset()
    m.add_parameter(["p"])
    m.add_variable(["u"])
    p = tmp_path / "data.csv"
    p.write_text("p,u\n1,2\n2,3\n")
    m.dataset(parameters=p)
    with pytest.raises(ValueError):
        m._load_dataset()
    m.dataset(parameters=p, variables=p)
    assert m._load_dataset()[0].shape == (2, 1)
    q = tmp_path / "one.csv"
    q.write_text("u\n1\n")
    m.dataset(parameters=p, variables=q)
    with pytest.raises(ValueError):
        m._load_dataset()
    for fraction in [0, 1]:
        with pytest.raises(ValueError):
            m._split(np.ones((2, 1)), None, {"train_frac": fraction, "seed": 1})
    m.set_train_config({"epochs": 1})
    m.set_projection_config({"method": "admm", "max_iterations": 3})
    assert m.projection_method == "admm"
    with pytest.raises(FileNotFoundError):
        m.extract(tmp_path / "absent.npz")
    np.savez(tmp_path / "test.npz", extra=np.array([3]))
    assert m.extract(tmp_path / "test.npz")["extra"][0] == 3
    assert m.extra[0] == 3

    @dataclass
    class Config:
        epochs: int = 2

    assert _normalize_mapping(Config()) == {"epochs": 2}
    config = object()
    assert _projection_definition(None, "admm", config) is config
    assert (
        _projection_definition(
            None, "fischer-burmeister-newton", {"gn_tol": 1e-5}
        ).tolerance
        == 1e-5
    )
    with pytest.raises(RuntimeError):
        m.predict([1])
    with pytest.raises(RuntimeError):
        m.export_inference()
    with pytest.raises(RuntimeError):
        m.export_predictor()
    with pytest.raises(RuntimeError):
        m.objective_gradient([1], objective="x")
    with pytest.raises(RuntimeError):
        m.prepare_objective_gradient(None)
    with pytest.raises(RuntimeError):
        m._checkpoint_payload()
    with pytest.raises(ValueError):
        m.compare_projections(methods=[])


def test_observed_variables():
    p = Projection()
    p.add_parameter(["p"])
    p.add_variable(["u", "v"])
    m = KKTHardNet(
        "parameter_estimation",
        loss=Loss.parameter_estimation(observed_variables=("u",)),
        projection=p,
    )
    assert m._target_names() == ("u",) and m._observation_indices() == (0,)
    m.loss = Loss.parameter_estimation(observed_variables=("unknown",))
    with pytest.raises(ValueError):
        m._target_names()
    m.loss = Loss.parameter_estimation()
    assert m._target_names() == ("u", "v")


def test_history_decode_aliases_and_failures(tmp_path):
    h = reporting._decode_history_columns(
        [
            {"epoch": "1", "stage": "raw", "a": "", "b": "word"},
            {"epoch": "2", "b": None},
        ]
    )
    assert h["epoch"].dtype == np.int64 and np.isnan(h["a"]).all()
    assert list(h["b"]) == ["word", ""] and list(h["stage"]) == ["raw", ""]
    assert reporting._history_column({"train_loss": np.array([1])}, "tr_mse")[0] == 1
    with pytest.raises(RuntimeError):
        reporting._history_column({}, "tr_eq")
    with pytest.raises(RuntimeError):
        reporting._history_column({"tr_eq": np.zeros((2, 1))}, "tr_eq")
    assert (
        reporting._inverse_history_columns(
            {
                "inverse_x": np.array([1]),
                "inverse_actual_x": np.array([2]),
                "inverse_nan": np.array([np.nan]),
            }
        )[0][0]
        == "x"
    )
    m = KKTHardNet()
    assert m._inverse_actual_values_for_plot(["x"]) == [None]
    m.inverse_actual_values = [2]
    assert m._inverse_actual_values_for_plot(["x", "y"]) == [2, None]
    with pytest.raises(RuntimeError):
        m._read_history()
    with pytest.raises(RuntimeError):
        m._run_dir()
    m._metadata_path = str(tmp_path / "metadata.json")
    assert m._run_dir() == tmp_path
    for method in [
        "linear-equality",
        "admm",
        "chambolle-pock",
        "fischer-burmeister-newton",
        "fischer-burmeister-reduced",
        "primal-dual-interior-point",
    ]:
        assert "matrix_free" in reporting._sparse_runtime_description(method)
    assert m.summary()["projection_method"] is None


def test_scalar_metric_aggregation():
    assert fit._aggregate_epoch_metrics([], [], ("x",)) == {"x": 0}
    assert fit._aggregate_epoch_metrics([{}], [1], ("x",)) == {"x": 0}
    assert fit._aggregate_epoch_metrics([{"x": 1.0}, {"x": 3.0}], [1, 3], ("x",)) == {
        "x": 2.5
    }
    result = fit._lightweight_history_metrics(
        {"task_loss": 2, "total_loss": 3, "primal_consistency_loss": 1}
    )
    assert result["loss"] == 2 and result["total_loss"] == 3
