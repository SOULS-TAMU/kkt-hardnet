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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KKT-HardNet runner.")
    parser.add_argument("--type", required=True, choices=("qp", "qcqp", "nlp", "nonconvex", "nonconvx", "general"))
    parser.add_argument("--action", default="run", choices=("run", "data"))
    parser.add_argument("--mode", default="forward", choices=("forward", "inverse"), help="main.py supports forward mode only.")
    parser.add_argument("--p", type=int, default=None, help="Parameter dimension override.")
    parser.add_argument("--n", type=int, default=None, help="Decision dimension override.")
    parser.add_argument("--me", type=int, default=None, help="Equality count override.")
    parser.add_argument("--mi", type=int, default=None, help="Inequality count override.")
    parser.add_argument("--samples", type=int, default=None, help="Number of labeled data points.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--hidden_layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=None, help="Gaussian label-noise scale applied after clean labels are generated.")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return ProblemBuilder.run(args, root=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
