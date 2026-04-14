#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KKTHN_SRC = ROOT / "kkthn" / "src"

for candidate in (KKTHN_SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from kkthn.builder import ProblemBuilder  # noqa: E402


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    builder = build_problem()
    return builder.run(args, root=ROOT, data=DATA, train=TRAIN)


if __name__ == "__main__":
    raise SystemExit(main())
