"""Orbax resume must reproduce uninterrupted training for both backbone adapters."""

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.optional


def same(a, b):
    aa, sa = jax.tree_util.tree_flatten(a)
    bb, sb = jax.tree_util.tree_flatten(b)
    assert sa == sb
    for one, two in zip(aa, bb):
        np.testing.assert_array_equal(np.asarray(one), np.asarray(two))


@pytest.mark.parametrize("backend", ["jax", "flax"])
def test_exact_checkpoint_resume(tmp_path, monkeypatch, backend):
    monkeypatch.chdir(tmp_path)
    x = np.linspace(0.3, 1.3, 8)[:, None]
    y = np.column_stack((0.6 * x[:, 0], 0.4 * x[:, 0]))
    from kkthn import KKTHardNet, Projection, MLP, Loss

    data = tmp_path / "data.csv"
    np.savetxt(
        data, np.column_stack((x, y)), delimiter=",", header="p,u,v", comments=""
    )

    def make(name, epochs):
        p = Projection(method="chambolle-pock")
        t = p.add_parameter(["p"])
        v = p.add_variable(["u", "v"])
        p.constraints.add(v.u + v.v == t.p)
        m = KKTHardNet(
            "surrogate",
            MLP(hidden_sizes=(4,), backend=backend),
            Loss.mse(),
            p,
            name=name,
            train={
                "epochs": epochs,
                "batch_size": 4,
                "seed": 7,
                "pretrain": False,
                "print_every": 0,
                "output_dir": str(tmp_path / "models"),
            },
        )
        m.dataset(parameters=data, variables=data)
        return m

    config = {"max_iterations": 40, "tolerance": 1e-7}
    full = make("full", 2)
    full.model(projection=config)
    first = make("first", 1)
    first.model(projection=config)
    checkpoint = first.save_checkpoint(tmp_path / "checkpoint")
    resumed = make("resumed", 2)
    resumed.resume_training(checkpoint, epochs=2)
    same(full._checkpoint_payload()[0], resumed._checkpoint_payload()[0])
    same(full.predict(x), resumed.predict(x))
