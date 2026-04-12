#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KKTHN_SRC = ROOT / "kkthn" / "src"


def _discover_nlpopt_root(start: Path) -> Path | None:
    for env_name in ("NLP_OPT_NET_ROOT", "NLPOPTNET_ROOT"):
        raw = __import__("os").environ.get(env_name)
        if raw:
            candidate = Path(raw).expanduser().resolve()
            if (candidate / "nlpopt" / "src").exists():
                return candidate
    for parent in (start, *start.parents):
        for candidate in (parent, parent / "NLPOptNet"):
            if (candidate / "nlpopt" / "src").exists():
                return candidate
    return None


NLP_ROOT = _discover_nlpopt_root(ROOT)
NLP_SRC = NLP_ROOT / "nlpopt" / "src" if NLP_ROOT is not None else None
for candidate in (KKTHN_SRC, ROOT, NLP_ROOT, NLP_SRC):
    if candidate is None:
        continue
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from kkthn.problems import generate_labels_with_slsqp, write_json  # noqa: E402
from kkthn.string_problem import build_model_from_string_problem  # noqa: E402
from kkthn.training import KKTTrainConfig, train_kkt_hardnet  # noqa: E402


# Edit this block for string-defined general problems.
PROBLEM = {
    "parameters": ["x1", "x2"],
    "variables": ["y1", "y2", "y3"],
    "objective": "0.5*(y1**2 + y2**2 + y3**2)",
    "constraints": [
        "y1 + y2 - x1 == 0",
        "y2 + y3 - x2 == 0",
        "y1**2 + y3**2 <= 2.0",
    ],
    "y_bound": 4.0,
}

DATA = {
    "num_samples": 12,
    "seed": 42,
    "x_L": [-1.0, -1.0],
    "x_U": [1.0, 1.0],
}

TRAIN = {
    "epochs": 2,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "train_frac": 0.8,
    "hidden_size": 32,
    "hidden_layers": 2,
    "seed": 42,
    "dtype": "float64",
    "print_every": 1,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a string-defined general KKT-HardNet example.")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args(argv)


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

    model, string_meta = build_model_from_string_problem(PROBLEM)
    rng = np.random.default_rng(int(data["seed"]))
    X = rng.uniform(
        np.asarray(data["x_L"], dtype=np.float64),
        np.asarray(data["x_U"], dtype=np.float64),
        size=(int(data["num_samples"]), len(PROBLEM["parameters"])),
    )
    Y, label_meta = generate_labels_with_slsqp(model, X)
    metadata = {"problem": PROBLEM, "string_problem": string_meta, **label_meta}

    output_root = Path(args.output_dir) if args.output_dir else ROOT / "case" / "general" / "runs"
    run_dir = output_root / f"string_general_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "string_problem.json", PROBLEM)
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
