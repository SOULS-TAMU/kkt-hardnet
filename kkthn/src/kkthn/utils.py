from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .problems import (
    apply_label_noise,
    apply_overrides,
    build_problem_bundle,
    case_folder_name,
    generate_labels_with_slsqp,
    load_json,
    normalize_problem_type,
    write_json,
)
from .projection import ProjectionSettings
from .training import KKTTrainConfig, train_kkt_hardnet


def _projection_cfg(proj_cfg: dict[str, Any]) -> ProjectionSettings:
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


def _train_cfg(cfg_dict: dict[str, Any], proj_cfg: dict[str, Any] | None = None) -> KKTTrainConfig:
    projection = _projection_cfg(proj_cfg or {})
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
        projection=projection,
    )


def _resolve_dataset_path(path: str | Path, *, root: Path | None = None) -> Path:
    dataset_path = Path(path)
    if dataset_path.is_absolute():
        return dataset_path
    return Path.cwd() / dataset_path if root is None else Path(root) / dataset_path


def _read_builder_dataset(
    builder: Any,
    *,
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    spec = builder.dataset_spec
    if spec is None:
        return None

    dataset_path = _resolve_dataset_path(spec.path, root=root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")
    if len(spec.parameter_columns) != len(builder.parameter_names):
        raise ValueError("Dataset parameter_columns must match the number of builder parameters.")
    if len(spec.variable_columns) != len(builder.variable_names):
        raise ValueError("Dataset variable_columns must match the number of builder variables.")

    with open(dataset_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = set(reader.fieldnames or [])
        required = list(spec.parameter_columns) + list(spec.variable_columns)
        missing = [name for name in required if name not in headers]
        if missing:
            raise ValueError(f"Dataset CSV is missing columns: {missing}")
        x_rows = []
        y_rows = []
        for row_idx, row in enumerate(reader, start=2):
            try:
                x_rows.append([float(row[name]) for name in spec.parameter_columns])
                y_rows.append([float(row[name]) for name in spec.variable_columns])
            except ValueError as exc:
                raise ValueError(f"Non-numeric value in dataset CSV at line {row_idx}.") from exc

    if not x_rows:
        raise ValueError(f"Dataset CSV has no data rows: {dataset_path}")

    meta = {
        "label_source": "csv_dataset",
        "dataset_path": str(dataset_path),
        "dataset_parameter_columns": list(spec.parameter_columns),
        "dataset_variable_columns": list(spec.variable_columns),
        "num_samples": len(x_rows),
    }
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64), meta


def _write_matrix_csv(path: Path, data: np.ndarray, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(names))
        writer.writerows(np.asarray(data, dtype=np.float64).tolist())


def _inverse_arrays(data: dict[str, Any], names: list[str], *, require_init: bool) -> tuple[np.ndarray, np.ndarray]:
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


def _print_configuration(mode: str, problem_meta: dict[str, Any], data: dict[str, Any], train: dict[str, Any]) -> None:
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


def _write_run_config(
    run_dir: Path,
    *,
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    proj_cfg: dict[str, Any],
) -> None:
    write_json(
        run_dir / "config.json",
        {
            "data": data_cfg,
            "config": train_cfg,
            "proj": proj_cfg,
        },
    )


def run_prepared_problem(
    *,
    model,
    X: np.ndarray,
    Y: np.ndarray,
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    proj_cfg: dict[str, Any],
    run_dir: Path,
    metadata: dict[str, Any],
    action: str = "run",
    problem_meta: dict[str, Any] | None = None,
    parameter_names: list[str] | None = None,
    variable_names: list[str] | None = None,
    inverse_param_init: np.ndarray | None = None,
    inverse_param_labels: np.ndarray | None = None,
    inverse_param_names: list[str] | None = None,
    return_result: bool = False,
) -> int | dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if problem_meta is not None:
        write_json(run_dir / "problem.json", problem_meta)
    _write_run_config(run_dir, data_cfg=data_cfg, train_cfg=train_cfg, proj_cfg=proj_cfg)
    if parameter_names is not None:
        _write_matrix_csv(run_dir / "parameters.csv", X, parameter_names)
    if variable_names is not None:
        _write_matrix_csv(run_dir / "variables.csv", Y, variable_names)

    if action == "data":
        print(f"Generated labels only. Config saved to: {run_dir}")
        return 0

    result = train_kkt_hardnet(
        model=model,
        X=X,
        Y=Y,
        cfg=_train_cfg(train_cfg, proj_cfg),
        output_dir=run_dir,
        metadata=metadata,
        inverse_param_init=inverse_param_init,
        inverse_param_labels=inverse_param_labels,
        inverse_param_names=inverse_param_names,
    )
    return result if return_result else 0


def _apply_builder_overrides(args, data: dict[str, Any], train: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data_out = dict(data)
    train_out = dict(train)
    if args.samples is not None:
        data_out["num_samples"] = int(args.samples)
    if args.epochs is not None:
        train_out["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_out["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        data_out["seed"] = int(args.seed)
        train_out["seed"] = int(args.seed)
    if args.noise_scale is not None:
        data_out["noise_scale"] = float(args.noise_scale)
    return data_out, train_out


def _run_standard_case(
    args,
    *,
    root: Path,
    data: dict[str, Any] | None = None,
    train: dict[str, Any] | None = None,
    proj: dict[str, Any] | None = None,
) -> int:
    if args.mode != "forward":
        print("main.py only supports --mode forward. Use run_general.py --mode inverse for builder-defined inverse problems.", file=sys.stderr)
        return 2

    problem_type = normalize_problem_type(args.type)
    case_dir = Path(root) / "case" / case_folder_name(problem_type)
    data_cfg = dict(data) if data is not None else load_json(case_dir / "data.json")
    cfg_dict = dict(train) if train is not None else load_json(case_dir / "config.json")
    proj_cfg = dict(proj) if proj is not None else load_json(case_dir / "proj.json")
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
        noise_scale=args.noise_scale,
    )

    print("=" * 80)
    print(f"KKT-HardNet runner | type={args.type} action={args.action} mode={args.mode}")
    print(f"Case directory: {case_dir}")
    print(f"Data config: {data_cfg}")
    print(f"Train config: {cfg_dict}")
    print(f"Projection config: {proj_cfg}")
    print("=" * 80)

    bundle = build_problem_bundle(problem_type, case_dir, data_cfg)
    print(f"Labels: X={bundle.X.shape} Y={bundle.Y.shape} source={bundle.metadata.get('label_source')}")

    run_root = Path(args.output_dir) if args.output_dir else case_dir / "runs"
    run_dir = run_root / f"{problem_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    return run_prepared_problem(
        model=bundle.model,
        X=bundle.X,
        Y=bundle.Y,
        data_cfg=data_cfg,
        train_cfg=cfg_dict,
        proj_cfg=proj_cfg,
        run_dir=run_dir,
        metadata=bundle.metadata,
        action=args.action,
    )


def _run_builder_case(
    args,
    *,
    root: Path,
    builder: Any,
    data: dict[str, Any],
    train: dict[str, Any],
    proj: dict[str, Any] | None = None,
) -> int:
    data_cfg, train_cfg = _apply_builder_overrides(args, data, train)
    inv_names = list(builder.inverse_parameter_names)
    inv_labels = np.zeros((0,), dtype=np.float64)
    inv_init = np.zeros((0,), dtype=np.float64)
    if inv_names or args.mode == "inverse":
        inv_labels, inv_init = _inverse_arrays(data_cfg, inv_names, require_init=args.mode == "inverse")

    dataset = _read_builder_dataset(builder, root=root)
    model, problem_meta = builder.build_model(
        train_inverse=args.mode == "inverse",
        inverse_values=inv_labels,
        allow_missing_objective=dataset is not None,
    )
    _print_configuration(args.mode, problem_meta, data_cfg, train_cfg)

    if dataset is not None:
        X, Y, label_meta = dataset
        if args.samples is not None:
            X = X[: int(data_cfg["num_samples"])]
            Y = Y[: int(data_cfg["num_samples"])]
            label_meta["num_samples"] = int(X.shape[0])
    else:
        rng = np.random.default_rng(int(data_cfg["seed"]))
        X = rng.uniform(
            np.asarray(data_cfg["x_L"], dtype=np.float64),
            np.asarray(data_cfg["x_U"], dtype=np.float64),
            size=(int(data_cfg["num_samples"]), len(builder.parameter_names)),
        )
        if args.mode == "inverse":
            theta = np.broadcast_to(inv_labels, (X.shape[0], inv_labels.size))
            label_X = np.concatenate([X, theta], axis=1)
        else:
            label_X = X
        Y, label_meta = generate_labels_with_slsqp(model, label_X)
        Y, label_meta = apply_label_noise(Y, data_cfg, label_meta)

    metadata = {
        "mode": args.mode,
        "problem": problem_meta,
        "inverse_parameter_names": inv_names,
        "inverse_parameter_labels": inv_labels.tolist(),
        "inverse_parameter_initial": inv_init.tolist(),
        **label_meta,
    }

    output_root = Path(args.output_dir) if args.output_dir else Path(root) / "case" / "general" / "runs"
    run_dir = output_root / f"builder_general_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    return run_prepared_problem(
        model=model,
        X=X,
        Y=Y,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        proj_cfg=dict(proj or {}),
        run_dir=run_dir,
        metadata=metadata,
        problem_meta=problem_meta,
        parameter_names=builder.parameter_names,
        variable_names=builder.variable_names,
        inverse_param_init=inv_init if args.mode == "inverse" else None,
        inverse_param_labels=inv_labels if args.mode == "inverse" else None,
        inverse_param_names=inv_names if args.mode == "inverse" else None,
    )


def _run_prepared_case(
    args,
    *,
    root: Path,
    model,
    X: np.ndarray,
    Y: np.ndarray,
    data: dict[str, Any],
    train: dict[str, Any],
    proj: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    problem_meta: dict[str, Any] | None = None,
    parameter_names: list[str] | None = None,
    variable_names: list[str] | None = None,
    inverse_param_init: np.ndarray | None = None,
    inverse_param_labels: np.ndarray | None = None,
    inverse_param_names: list[str] | None = None,
) -> int:
    run_name = str(getattr(args, "type", "prepared"))
    mode = str(getattr(args, "mode", "forward"))
    output_root = Path(args.output_dir) if getattr(args, "output_dir", None) else Path(root) / "notebooks" / "_runs" / run_name
    run_dir = output_root / f"{run_name}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    return run_prepared_problem(
        model=model,
        X=X,
        Y=Y,
        data_cfg=dict(data),
        train_cfg=dict(train),
        proj_cfg=dict(proj or {}),
        run_dir=run_dir,
        metadata=dict(metadata or {}),
        action=str(getattr(args, "action", "run")),
        problem_meta=problem_meta,
        parameter_names=parameter_names,
        variable_names=variable_names,
        inverse_param_init=inverse_param_init,
        inverse_param_labels=inverse_param_labels,
        inverse_param_names=inverse_param_names,
    )
