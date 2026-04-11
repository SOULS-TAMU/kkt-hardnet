#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
NLP_SRC = PARENT / "nlpopt" / "src"
KKTHN_SRC = ROOT / "kkthn" / "src"
for candidate in (KKTHN_SRC, ROOT, PARENT, NLP_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from kkthn.problems import (  # noqa: E402
    apply_overrides,
    build_problem_bundle,
    case_folder_name,
    load_json,
    normalize_problem_type,
    write_json,
)
from kkthn.projection import ProjectionSettings  # noqa: E402
from kkthn.training import KKTTrainConfig, train_kkt_hardnet  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KKT-HardNet runner.")
    parser.add_argument("--type", required=True, choices=("qp", "qcqp", "nlp", "nonconvex", "nonconvx", "general"))
    parser.add_argument("--action", default="run", choices=("run", "data"))
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
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args(argv)


def _projection_cfg(proj_cfg: dict) -> ProjectionSettings:
    defaults = ProjectionSettings()
    return ProjectionSettings(
        fb_eps=float(proj_cfg.get("fb_eps", defaults.fb_eps)),
        gn_max_iters=int(proj_cfg.get("gn_max_iters", defaults.gn_max_iters)),
        gn_tol=float(proj_cfg.get("gn_tol", defaults.gn_tol)),
        gn_reg=float(proj_cfg.get("gn_reg", defaults.gn_reg)),
        armijo_alpha=float(proj_cfg.get("armijo_alpha", defaults.armijo_alpha)),
        armijo_beta=float(proj_cfg.get("armijo_beta", defaults.armijo_beta)),
        armijo_max_steps=int(proj_cfg.get("armijo_max_steps", defaults.armijo_max_steps)),
        backward_reg=float(proj_cfg.get("backward_reg", defaults.backward_reg)),
    )


def _train_cfg(cfg_dict: dict, proj_cfg: dict) -> KKTTrainConfig:
    return KKTTrainConfig(
        epochs=int(cfg_dict.get("epochs", 2)),
        batch_size=int(cfg_dict.get("batch_size", 8)),
        learning_rate=float(cfg_dict.get("learning_rate", 1e-3)),
        train_frac=float(cfg_dict.get("train_frac", 0.8)),
        hidden_size=int(cfg_dict.get("hidden_size", 64)),
        hidden_layers=int(cfg_dict.get("hidden_layers", cfg_dict.get("hidden_dim", 2))),
        seed=int(cfg_dict.get("seed", 42)),
        dtype=str(cfg_dict.get("dtype", "float64")),
        print_every=max(1, int(cfg_dict.get("print_every", 1))),
        projection=_projection_cfg(proj_cfg),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    problem_type = normalize_problem_type(args.type)
    case_dir = ROOT / "case" / case_folder_name(problem_type)
    data_cfg = load_json(case_dir / "data.json")
    cfg_dict = load_json(case_dir / "config.json")
    proj_cfg = load_json(case_dir / "proj.json")
    data_cfg, cfg_dict = apply_overrides(
        data_cfg,
        cfg_dict,
        p=args.p,
        n=args.n,
        me=args.me,
        mi=args.mi,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        solver=args.solver,
        train_frac=args.train_frac,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        seed=args.seed,
    )

    print("=" * 80)
    print(f"KKT-HardNet runner | type={args.type} action={args.action}")
    print(f"Case directory: {case_dir}")
    print(f"Data config: {data_cfg}")
    print(f"Train config: {cfg_dict}")
    print(f"Projection config: {proj_cfg}")
    print("=" * 80)

    bundle = build_problem_bundle(problem_type, case_dir, data_cfg)
    print(f"Labels: X={bundle.X.shape} Y={bundle.Y.shape} source={bundle.metadata.get('label_source')}")

    run_root = Path(args.output_dir) if args.output_dir else case_dir / "runs"
    run_dir = run_root / f"{problem_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "data.json", data_cfg)
    write_json(run_dir / "config.json", cfg_dict)
    write_json(run_dir / "proj.json", proj_cfg)
    write_json(run_dir / "label_metadata.json", bundle.metadata)

    if args.action == "data":
        print(f"Generated labels only. Metadata saved to: {run_dir}")
        return 0

    train_kkt_hardnet(
        model=bundle.model,
        X=bundle.X,
        Y=bundle.Y,
        cfg=_train_cfg(cfg_dict, proj_cfg),
        output_dir=run_dir,
        metadata=bundle.metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
