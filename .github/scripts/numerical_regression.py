import argparse


import json

import os

from pathlib import Path

import subprocess

import sys

import tempfile

import jax


import numpy as np


def reload_check(path, inputs, expected, framework, full_metadata=None):
    code = """
import importlib.abc, json, sys
import numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'kkthn','daehn','nlpoptnet','flax','orbax'}:
   raise AssertionError('Deployment imported '+fullname)
sys.meta_path.insert(0,Block())
from ohagi.inference import load
predictor=load(sys.argv[1])
x=json.loads(sys.argv[2]); expected=json.loads(sys.argv[3])
for _ in range(25):
 np.testing.assert_allclose(predictor.predict(x),expected,atol=1e-8,rtol=1e-7)
 np.testing.assert_allclose(predictor.predict_jax(x),expected,atol=1e-8,rtol=1e-7)
"""
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(path),
            json.dumps(np.asarray(inputs).tolist()),
            json.dumps(np.asarray(expected).tolist()),
        ],
        check=True,
        timeout=180,
    )
    if framework == "kkthn":
        from kkthn import KKTHardNet

        facade = KKTHardNet().load(full_metadata or path)
    elif framework == "daehn":
        from daehn import DAEHardNet

        facade = DAEHardNet().load(path)
    else:
        from nlpoptnet import NLPOptNet

        facade = NLPOptNet().load(full_metadata or path)
    np.testing.assert_allclose(
        np.asarray(facade.predict(np.asarray(inputs)[None])).reshape(-1),
        np.asarray(expected).reshape(-1),
        atol=1e-8,
        rtol=1e-7,
    )


def kkthn(root):
    from kkthn import KKTHardNet, Projection, MLP, Loss

    x = np.linspace(0.4, 1.2, 8)
    data = root / "data.csv"
    np.savetxt(
        data,
        np.column_stack((x, 0.6 * x, 0.4 * x)),
        delimiter=",",
        header="p,u,v",
        comments="",
    )
    problem = Projection(method="chambolle-pock")
    p = problem.add_parameter(["p"])
    y = problem.add_variable(["u", "v"])
    problem.constraints.add(y.u + y.v == p.p)
    model = KKTHardNet(
        "surrogate",
        MLP(hidden_sizes=(4,)),
        Loss.mse(),
        problem,
        name="ci",
        train={
            "epochs": 2,
            "batch_size": 2,
            "seed": 7,
            "pretrain": False,
            "print_every": 0,
            "output_dir": str(root / "models"),
        },
    )
    model.dataset(parameters=data, variables=data)
    result = model.model(projection={"max_iterations": 1000, "tolerance": 1e-9})
    assert len(result["history"]) == 2
    path = root / "metadata.json"
    model.export_inference(path)
    from ohagi.inference import load

    expected = load(path).predict([0.7])
    np.testing.assert_allclose(expected.sum(), 0.7, atol=1e-7)
    reload_check(path, [0.7], expected, "kkthn", result["metadata_path"])
    return {"epochs": 2, "prediction": expected.tolist()}


def run_case(directory):
    """Exercise the installed frontend through short training and fresh-process reload."""
    jax.config.update("jax_enable_x64", True)
    original = Path.cwd()
    os.chdir(directory)
    try:
        return kkthn(Path(directory))
    finally:
        os.chdir(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as directory:
        result = run_case(directory)
    baseline = root / ".github/validation/numerical.json"
    if args.write_baseline:
        baseline.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        expected = json.loads(baseline.read_text())

        def check(actual, reference):
            if isinstance(reference, dict):
                assert actual.keys() == reference.keys()
                for key in reference:
                    check(actual[key], reference[key])
            else:
                np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)

        check(result, expected)
    (root / "numerical-results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
