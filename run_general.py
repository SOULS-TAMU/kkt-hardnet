#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KKTHN_SRC = ROOT / "kkthn" / "src"

for candidate in (KKTHN_SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from kkthn.problems import generate_labels_with_slsqp, write_json  # noqa: E402
from kkthn.problems import apply_label_noise  # noqa: E402
from kkthn.problem_builder import ProblemBuilder  # noqa: E402
from kkthn.training import KKTTrainConfig, train_kkt_hardnet  # noqa: E402


DATA = {
    "num_samples": 100,
    "seed": 42,
    "noise_scale": 0.1,
    "x_L": [-1.0, -1.0],
    "x_U": [1.0, 1.0],
    "inv_param": ["a0", "a1"],
    "inv_param_label": [1, -1],
    "inv_param_init": [10, -10],
}

TRAIN = {
    "epochs": 10000,
    "batch_size": 20,
    "learning_rate": 1e-3,
    "train_frac": 0.8,
    "hidden_size": 32,
    "hidden_layers": 2,
    "seed": 42,
    "dtype": "float64",
    "print_every": 1000,
    "verbose": True,
}


def build_problem() -> ProblemBuilder:
    """Define the editable symbolic general problem.

    In forward mode, inverse parameters are fixed to DATA["inv_param_label"].
    In inverse mode, they become trainable Optax parameters.
    """

    builder = ProblemBuilder(y_bound=4.0)
    x = builder.add_parameter(["x1", "x2"])
    theta = builder.add_inverse_parameter(DATA["inv_param"])
    y = builder.add_variable(["y1", "y2", "y3"])

    builder.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)
    builder.constraints.add(
        theta.a0 * y.y1 + y.y2 - x.x1 == 0,
        y.y2 - theta.a1 * y.y3 - x.x2 == 0,
        y.y1**2 + y.y3**2 <= 2.0,
    )
    builder.bounds.set(lower=-4.0, upper=4.0)
    return builder


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a builder-defined general KKT-HardNet example.")
    parser.add_argument("--mode", default="forward", choices=("forward", "inverse"))
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args(argv)


def _inverse_arrays(data: dict, names: list[str], *, require_init: bool) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(data.get("inv_param_label", []), dtype=np.float64).reshape(-1)
    init_raw = data.get("inv_param_init", [])
    init = np.asarray(init_raw, dtype=np.float64).reshape(-1) if len(init_raw) else np.zeros_like(labels)
    if require_init and not names:
        raise ValueError("DATA['inv_param'] must contain at least one inverse parameter name in inverse mode.")
    if len(names) != labels.size:
        raise ValueError("DATA['inv_param_label'] must have the same length as builder.inverse_parameter_names.")
    if init.size != labels.size:
        raise ValueError("DATA['inv_param_init'] must be empty or have the same length as DATA['inv_param'].")
    return labels, init


def _print_configuration(mode: str, problem_meta: dict, data: dict, train: dict) -> None:
    if not bool(train.get("verbose", False)):
        return
    print("=" * 80)
    print(f"KKT-HardNet general builder runner | mode={mode}")
    print(f"Parameters: {problem_meta['parameter_names']}")
    print(f"Variables : {problem_meta['variable_names']}")
    print(f"Objective : {problem_meta['objective']}")
    print(f"Constraints ({len(problem_meta['constraints'])}):")
    for idx, constraint in enumerate(problem_meta["constraints"], start=1):
        print(f"  {idx}. {constraint['text']}")
    if problem_meta["inverse_parameter_names"]:
        print(f"Inverse parameters: {problem_meta['inverse_parameter_names']}")
        if mode == "inverse":
            print(f"Inverse labels    : {data.get('inv_param_label', [])}")
            print(f"Inverse init      : {data.get('inv_param_init', []) or '[zeros]'}")
        else:
            print(f"Fixed inverse values: {data.get('inv_param_label', [])}")
    print(f"Data config: {data}")
    print(f"Train config: {train}")
    print("=" * 80)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data = dict(DATA)
    train = dict(TRAIN)
    if args.samples is not None:
        data["num_samples"] = int(args.samples)
    if args.epochs is not None:
        train["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        data["seed"] = int(args.seed)
        train["seed"] = int(args.seed)
    if args.noise_scale is not None:
        data["noise_scale"] = float(args.noise_scale)

    problem_builder = build_problem()
    inv_names = list(problem_builder.inverse_parameter_names)
    inv_labels = np.zeros((0,), dtype=np.float64)
    inv_init = np.zeros((0,), dtype=np.float64)
    if inv_names or args.mode == "inverse":
        inv_labels, inv_init = _inverse_arrays(data, inv_names, require_init=args.mode == "inverse")

    model, problem_meta = problem_builder.build_model(
        train_inverse=args.mode == "inverse",
        inverse_values=inv_labels,
    )
    _print_configuration(args.mode, problem_meta, data, train)

    rng = np.random.default_rng(int(data["seed"]))
    X = rng.uniform(
        np.asarray(data["x_L"], dtype=np.float64),
        np.asarray(data["x_U"], dtype=np.float64),
        size=(int(data["num_samples"]), len(problem_builder.parameter_names)),
    )
    if args.mode == "inverse":
        theta = np.broadcast_to(inv_labels, (X.shape[0], inv_labels.size))
        label_X = np.concatenate([X, theta], axis=1)
    else:
        label_X = X
    Y, label_meta = generate_labels_with_slsqp(model, label_X)
    Y, label_meta = apply_label_noise(Y, data, label_meta)
    metadata = {
        "mode": args.mode,
        "problem": problem_meta,
        "inverse_parameter_names": inv_names,
        "inverse_parameter_labels": inv_labels.tolist(),
        "inverse_parameter_initial": inv_init.tolist(),
        **label_meta,
    }

    output_root = Path(args.output_dir) if args.output_dir else ROOT / "case" / "general" / "runs"
    run_dir = output_root / f"builder_general_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "problem.json", problem_meta)
    write_json(run_dir / "data.json", data)
    write_json(run_dir / "config.json", train)
    write_json(run_dir / "label_metadata.json", metadata)

    train_kkt_hardnet(
        model=model,
        X=X,
        Y=Y,
        cfg=KKTTrainConfig(
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            learning_rate=float(train["learning_rate"]),
            train_frac=float(train["train_frac"]),
            hidden_size=int(train["hidden_size"]),
            hidden_layers=int(train["hidden_layers"]),
            seed=int(train["seed"]),
            dtype=str(train["dtype"]),
            print_every=int(train["print_every"]),
        ),
        output_dir=run_dir,
        metadata=metadata,
        inverse_param_init=inv_init if args.mode == "inverse" else None,
        inverse_param_labels=inv_labels if args.mode == "inverse" else None,
        inverse_param_names=inv_names if args.mode == "inverse" else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
