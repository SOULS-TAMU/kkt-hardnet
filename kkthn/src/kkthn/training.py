from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .backbone import init_mlp_params, make_batched_mlp_apply
from .projection import ProjectionSettings, make_projection_layer

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class KKTTrainConfig:
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 1e-3
    train_frac: float = 0.8
    hidden_size: int = 64
    hidden_layers: int = 2
    seed: int = 42
    dtype: str = "float64"
    print_every: int = 1
    drop_last: bool = False
    projection: ProjectionSettings = ProjectionSettings()


def _dtype(name: str):
    normalized = str(name).lower()
    if normalized in {"float32", "fp32", "32"}:
        return jnp.float32
    if normalized in {"float64", "fp64", "64"}:
        return jnp.float64
    raise ValueError(f"Unsupported dtype '{name}'.")


def _sync(tree):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _split_dataset(X: np.ndarray, Y: np.ndarray, *, train_frac: float, seed: int):
    if not 0.0 < float(train_frac) < 1.0:
        raise ValueError("train_frac must satisfy 0 < train_frac < 1.")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(X.shape[0]))
    rng.shuffle(idx)
    n_train = max(1, min(int(float(train_frac) * len(idx)), len(idx) - 1))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], train_idx, val_idx


def _iterate_minibatches(X, Y, *, batch_size: int, seed: int, drop_last: bool):
    idx = np.arange(int(X.shape[0]))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    last = (len(idx) // int(batch_size)) * int(batch_size) if drop_last else len(idx)
    for start in range(0, last, int(batch_size)):
        end = start + int(batch_size)
        if end > len(idx) and drop_last:
            break
        sel = idx[start:end]
        if sel.size:
            yield X[sel], Y[sel]


def _measure_time(fn, *args, repeats: int = 5) -> float:
    durations = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        out = fn(*args)
        _sync(out)
        durations.append(time.perf_counter() - t0)
    return float(np.mean(durations))


def _json_safe(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def _fallback_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{idx}" for idx in range(int(count))]


def _named_columns(metadata: dict[str, Any] | None, *, data_n_x: int, n_y: int) -> tuple[list[str], list[str]]:
    problem_meta = {}
    if isinstance(metadata, dict):
        problem_meta = metadata.get("problem") if isinstance(metadata.get("problem"), dict) else metadata

    parameter_names = list(problem_meta.get("parameter_names", [])) if isinstance(problem_meta, dict) else []
    variable_names = list(problem_meta.get("variable_names", [])) if isinstance(problem_meta, dict) else []

    if len(parameter_names) != int(data_n_x):
        parameter_names = _fallback_names("x", int(data_n_x))
    if len(variable_names) != int(n_y):
        variable_names = _fallback_names("y", int(n_y))
    return [str(name) for name in parameter_names], [str(name) for name in variable_names]


def _write_predictions_csv(
    path: Path,
    *,
    X: np.ndarray,
    Y: np.ndarray | None,
    Y_hat: np.ndarray,
    Y_proj: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    parameter_names: list[str],
    variable_names: list[str],
) -> None:
    split_by_index = {int(idx): "train" for idx in np.asarray(train_idx, dtype=int).reshape(-1)}
    split_by_index.update({int(idx): "validation" for idx in np.asarray(val_idx, dtype=int).reshape(-1)})
    has_labels = Y is not None
    fieldnames = ["sample_index", "split"] + [f"param_{name}" for name in parameter_names]
    if has_labels:
        fieldnames += [f"true_{name}" for name in variable_names]
    fieldnames += [f"raw_pred_{name}" for name in variable_names] + [f"pred_{name}" for name in variable_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx in range(int(X.shape[0])):
            row: dict[str, Any] = {
                "sample_index": sample_idx,
                "split": split_by_index.get(sample_idx, ""),
            }
            for name, value in zip(parameter_names, X[sample_idx]):
                row[f"param_{name}"] = float(value)
            if has_labels:
                assert Y is not None
                for name, value in zip(variable_names, Y[sample_idx]):
                    row[f"true_{name}"] = float(value)
            for name, value in zip(variable_names, Y_hat[sample_idx]):
                row[f"raw_pred_{name}"] = float(value)
            for name, value in zip(variable_names, Y_proj[sample_idx]):
                row[f"pred_{name}"] = float(value)
            writer.writerow(row)


def _save_model_weights(path: Path, params: Any, *, inverse_mode: bool) -> dict[str, Any]:
    params_host = jax.device_get(params)
    network = params_host["network"] if inverse_mode else params_host
    arrays: dict[str, np.ndarray] = {}
    layers = []
    for idx, layer in enumerate(network):
        w_key = f"layer_{idx}_W"
        b_key = f"layer_{idx}_b"
        arrays[w_key] = np.asarray(layer["W"])
        arrays[b_key] = np.asarray(layer["b"])
        layers.append(
            {
                "index": idx,
                "weight_key": w_key,
                "bias_key": b_key,
                "weight_shape": list(arrays[w_key].shape),
                "bias_shape": list(arrays[b_key].shape),
            }
        )
    manifest: dict[str, Any] = {
        "format": "kkthn_mlp_weights_v1",
        "file": path.name,
        "layers": layers,
        "inverse_mode": bool(inverse_mode),
    }
    if inverse_mode:
        arrays["inverse_parameters"] = np.asarray(params_host["inverse"])
        manifest["inverse_parameter_key"] = "inverse_parameters"
        manifest["inverse_parameter_shape"] = list(arrays["inverse_parameters"].shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return manifest


def load_model_weights(path: Path, manifest: dict[str, Any]) -> Any:
    with np.load(path, allow_pickle=False) as arrays:
        network = []
        for layer in manifest["layers"]:
            network.append(
                {
                    "W": jnp.asarray(arrays[layer["weight_key"]]),
                    "b": jnp.asarray(arrays[layer["bias_key"]]),
                }
            )
        if bool(manifest.get("inverse_mode", False)):
            inverse_key = str(manifest["inverse_parameter_key"])
            return {
                "network": network,
                "inverse": jnp.asarray(arrays[inverse_key]),
            }
        return network


def train_kkt_hardnet(
    *,
    model,
    X: np.ndarray,
    Y: np.ndarray | None,
    cfg: KKTTrainConfig,
    param_name: str = "x",
    output_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
    inverse_param_init: np.ndarray | None = None,
    inverse_param_labels: np.ndarray | None = None,
    inverse_param_names: list[str] | None = None,
    task: str = "surrogate",
) -> dict[str, Any]:
    train_dtype = _dtype(cfg.dtype)
    X_np = np.asarray(X, dtype=np.float64)
    task_name = str(task).strip().lower()
    if task_name not in {"surrogate", "estimate", "optimize"}:
        raise ValueError("task must be one of surrogate, estimate, or optimize.")
    supervised = task_name in {"surrogate", "estimate"}
    if X_np.ndim != 2:
        raise ValueError("X must be a 2D array.")

    inverse_mode = inverse_param_init is not None
    inverse_init_np = (
        np.asarray(inverse_param_init, dtype=np.float64).reshape(-1)
        if inverse_mode
        else np.zeros((0,), dtype=np.float64)
    )
    inverse_labels_np = (
        None
        if inverse_param_labels is None
        else np.asarray(inverse_param_labels, dtype=np.float64).reshape(-1)
    )
    inverse_names = list(inverse_param_names or [f"theta_{idx}" for idx in range(inverse_init_np.size)])
    if inverse_mode and len(inverse_names) != inverse_init_np.size:
        raise ValueError("inverse_param_names must have the same length as inverse_param_init.")
    if inverse_labels_np is not None and inverse_labels_np.size != inverse_init_np.size:
        raise ValueError("inverse_param_labels must have the same length as inverse_param_init.")

    projection = make_projection_layer(model, param_name=param_name, settings=cfg.projection)
    dims = projection.dims
    Y_true_np = None if Y is None else np.asarray(Y, dtype=np.float64)
    if supervised:
        if Y_true_np is None:
            raise ValueError(f"{task_name} training requires Y labels.")
        if Y_true_np.ndim != 2:
            raise ValueError("Y must be a 2D array.")
        if X_np.shape[0] != Y_true_np.shape[0]:
            raise ValueError("X and Y must have the same number of rows.")
        if Y_true_np.shape[1] != dims["n_y"]:
            raise ValueError(f"Y has {Y_true_np.shape[1]} columns, expected {dims['n_y']}.")
        Y_np = Y_true_np
    else:
        Y_np = np.zeros((int(X_np.shape[0]), int(dims["n_y"])), dtype=np.float64)

    data_n_x = int(X_np.shape[1])
    if inverse_mode:
        expected_total = data_n_x + int(inverse_init_np.size)
        if expected_total != dims["n_x"]:
            raise ValueError(
                f"X columns plus inverse parameters equal {expected_total}, "
                f"but the string model expects {dims['n_x']} parameters."
            )
    elif data_n_x != dims["n_x"]:
        raise ValueError(f"X has {data_n_x} columns, expected {dims['n_x']}.")
    Xtr, Ytr, Xva, Yva, train_idx, val_idx = _split_dataset(X_np, Y_np, train_frac=cfg.train_frac, seed=cfg.seed)
    Xall_j = jnp.asarray(X_np, dtype=train_dtype)
    Xtr_j = jnp.asarray(Xtr, dtype=train_dtype)
    Ytr_j = jnp.asarray(Ytr, dtype=train_dtype)
    Xva_j = jnp.asarray(Xva, dtype=train_dtype)
    Yva_j = jnp.asarray(Yva, dtype=train_dtype)

    layer_sizes = [data_n_x] + [int(cfg.hidden_size)] * int(cfg.hidden_layers) + [dims["n_y"]]
    key = jax.random.PRNGKey(int(cfg.seed))
    network_params = init_mlp_params(key, layer_sizes, dtype=train_dtype)
    if inverse_mode:
        params = {
            "network": network_params,
            "inverse": jnp.asarray(inverse_init_np, dtype=train_dtype),
        }
    else:
        params = network_params
    batched_mlp_apply = make_batched_mlp_apply()
    optimizer = optax.adam(float(cfg.learning_rate))
    opt_state = optimizer.init(params)

    def network_tree(params_in):
        return params_in["network"] if inverse_mode else params_in

    def augmented_x(params_in, x_batch):
        if not inverse_mode:
            return x_batch
        theta = params_in["inverse"]
        theta_batch = jnp.broadcast_to(theta, (x_batch.shape[0], theta.shape[0]))
        return jnp.concatenate([x_batch, theta_batch], axis=1)

    @jax.jit
    def model_forward(params_in, x_batch):
        y_hat = batched_mlp_apply(network_tree(params_in), x_batch)
        y_proj = projection.project(augmented_x(params_in, x_batch), y_hat)
        return y_hat, y_proj

    @jax.jit
    def objective_batch(params_in, x_batch, y_batch):
        x_aug = augmented_x(params_in, x_batch)
        return jax.vmap(lambda xx, yy: model.objective_value({param_name: xx}, yy))(x_aug, y_batch)

    @jax.jit
    def constraint_violation_batch(params_in, x_batch, y_batch):
        x_aug = augmented_x(params_in, x_batch)
        ce = jax.vmap(projection.eq_constraints, in_axes=(0, 0))(y_batch, x_aug)
        gi = jax.vmap(projection.ineq_constraints, in_axes=(0, 0))(y_batch, x_aug)
        eq_l2 = jnp.mean(jnp.linalg.norm(ce, axis=1)) if dims["n_eq"] > 0 else jnp.asarray(0.0, dtype=train_dtype)
        ineq_l2 = jnp.mean(jnp.linalg.norm(jnp.maximum(gi, 0.0), axis=1)) if dims["n_ineq"] > 0 else jnp.asarray(0.0, dtype=train_dtype)
        return eq_l2, ineq_l2

    @jax.jit
    def loss_fn(params_in, x_batch, y_batch):
        y_hat, y_proj = model_forward(params_in, x_batch)
        if supervised:
            return jnp.mean(jnp.sum((y_proj - y_batch) ** 2, axis=1))
        del y_hat, y_batch
        return jnp.mean(objective_batch(params_in, x_batch, y_proj))

    @jax.jit
    def train_step(params_in, opt_state_in, x_batch, y_batch):
        loss_val, grads = jax.value_and_grad(loss_fn)(params_in, x_batch, y_batch)
        updates, opt_state_out = optimizer.update(grads, opt_state_in, params_in)
        params_out = optax.apply_updates(params_in, updates)
        return params_out, opt_state_out, loss_val

    @jax.jit
    def eval_metrics(params_in, x_batch, y_batch):
        y_hat, y_proj = model_forward(params_in, x_batch)
        if supervised:
            loss = jnp.mean(jnp.sum((y_proj - y_batch) ** 2, axis=1))
            raw_metric = jnp.mean(jnp.sum((y_hat - y_batch) ** 2, axis=1))
        else:
            del y_batch
            loss = jnp.mean(objective_batch(params_in, x_batch, y_proj))
            raw_metric = jnp.mean(objective_batch(params_in, x_batch, y_hat))
        eq_l2, ineq_l2 = constraint_violation_batch(params_in, x_batch, y_proj)
        return loss, raw_metric, eq_l2, ineq_l2, y_hat, y_proj

    @jax.jit
    def backbone_forward_fn(params_in, x_batch):
        return batched_mlp_apply(network_tree(params_in), x_batch)

    @jax.jit
    def projection_only_fn(params_in, x_batch, y_hat):
        return projection.project(augmented_x(params_in, x_batch), y_hat)

    @jax.jit
    def forward_loss_fn(params_in, x_batch, y_batch):
        y_hat = backbone_forward_fn(params_in, x_batch)
        y_proj = projection_only_fn(params_in, x_batch, y_hat)
        if supervised:
            return jnp.mean(jnp.sum((y_proj - y_batch) ** 2, axis=1))
        del y_batch
        return jnp.mean(objective_batch(params_in, x_batch, y_proj))

    grad_only_fn = jax.jit(jax.grad(forward_loss_fn))

    @jax.jit
    def optimizer_update_fn(params_in, opt_state_in, grads):
        updates, opt_state_out = optimizer.update(grads, opt_state_in, params_in)
        return optax.apply_updates(params_in, updates), opt_state_out

    print("KKT-HardNet")
    print(f"  dims: n_x={dims['n_x']} n_y={dims['n_y']} n_eq={dims['n_eq']} n_ineq={dims['n_ineq']}")
    print(f"  task: {task_name}")
    if inverse_mode:
        print(f"  mode: inverse data_n_x={data_n_x} inverse_n={inverse_init_np.size}")
        print(f"  inverse init: {dict(zip(inverse_names, inverse_init_np.tolist()))}")
    else:
        print("  mode: forward")
    print(f"  samples: train={Xtr.shape[0]} val={Xva.shape[0]} batch_size={cfg.batch_size}")
    print(f"  network: {layer_sizes}")

    warm_train = next(_iterate_minibatches(Xtr, Ytr, batch_size=cfg.batch_size, seed=cfg.seed, drop_last=cfg.drop_last))
    warm_val = next(_iterate_minibatches(Xva, Yva, batch_size=cfg.batch_size, seed=cfg.seed, drop_last=False))
    warm_x = jnp.asarray(warm_train[0], dtype=train_dtype)
    warm_y = jnp.asarray(warm_train[1], dtype=train_dtype)
    warm_vx = jnp.asarray(warm_val[0], dtype=train_dtype)
    warm_vy = jnp.asarray(warm_val[1], dtype=train_dtype)
    warm_params, warm_opt_state, warm_loss = train_step(params, opt_state, warm_x, warm_y)
    warm_eval = eval_metrics(params, warm_vx, warm_vy)
    warm_hat = backbone_forward_fn(params, warm_x)
    warm_proj = projection_only_fn(params, warm_x, warm_hat)
    warm_grads = grad_only_fn(params, warm_x, warm_y)
    warm_update = optimizer_update_fn(params, opt_state, warm_grads)
    _sync((warm_params, warm_opt_state, warm_loss, warm_eval, warm_hat, warm_proj, warm_grads, warm_update))

    history: list[dict[str, float | int]] = []
    train_step_time_total = 0.0
    validation_time_total = 0.0
    train_eval_time_total = 0.0
    train_batch_count_total = 0
    validation_batch_count_total = 0
    t0 = time.perf_counter()
    for epoch in range(1, int(cfg.epochs) + 1):
        batch_losses = []
        epoch_train_batches = 0
        epoch_train_step_time = 0.0
        train_epoch_t0 = time.perf_counter()
        for xb, yb in _iterate_minibatches(Xtr, Ytr, batch_size=cfg.batch_size, seed=cfg.seed + epoch, drop_last=cfg.drop_last):
            xb_j = jnp.asarray(xb, dtype=train_dtype)
            yb_j = jnp.asarray(yb, dtype=train_dtype)
            batch_t0 = time.perf_counter()
            params, opt_state, batch_loss = train_step(params, opt_state, xb_j, yb_j)
            _sync(batch_loss)
            batch_elapsed = time.perf_counter() - batch_t0
            train_step_time_total += batch_elapsed
            epoch_train_step_time += batch_elapsed
            batch_losses.append(float(batch_loss))
            epoch_train_batches += 1
            train_batch_count_total += 1
        train_epoch_time = time.perf_counter() - train_epoch_t0

        train_eval_t0 = time.perf_counter()
        tr_loss, tr_raw, tr_eq, tr_ineq, _tr_hat, _tr_proj = eval_metrics(params, Xtr_j, Ytr_j)
        _sync((tr_loss, tr_raw, tr_eq, tr_ineq))
        train_eval_time = time.perf_counter() - train_eval_t0
        train_eval_time_total += train_eval_time

        val_epoch_t0 = time.perf_counter()
        val_weight = 0
        val_acc = np.zeros((4,), dtype=np.float64)
        epoch_val_batches = 0
        for xb, yb in _iterate_minibatches(Xva, Yva, batch_size=cfg.batch_size, seed=cfg.seed, drop_last=False):
            xb_j = jnp.asarray(xb, dtype=train_dtype)
            yb_j = jnp.asarray(yb, dtype=train_dtype)
            va_batch = eval_metrics(params, xb_j, yb_j)
            _sync(va_batch)
            weight = int(xb.shape[0])
            val_acc += weight * np.asarray([float(va_batch[0]), float(va_batch[1]), float(va_batch[2]), float(va_batch[3])])
            val_weight += weight
            epoch_val_batches += 1
            validation_batch_count_total += 1
        validation_epoch_time = time.perf_counter() - val_epoch_t0
        validation_time_total += validation_epoch_time
        va_loss, va_raw, va_eq, va_ineq = (val_acc / max(1, val_weight)).tolist()
        row = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "train_raw_metric": float(tr_raw),
            "train_eq_l2": float(tr_eq),
            "train_ineq_l2": float(tr_ineq),
            "val_loss": float(va_loss),
            "val_raw_metric": float(va_raw),
            "val_eq_l2": float(va_eq),
            "val_ineq_l2": float(va_ineq),
            "mean_batch_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            "train_epoch_time_sec": float(train_epoch_time),
            "train_step_time_sec": float(epoch_train_step_time),
            "train_eval_time_sec": float(train_eval_time),
            "validation_epoch_time_sec": float(validation_epoch_time),
            "train_batches": int(epoch_train_batches),
            "validation_batches": int(epoch_val_batches),
            "train_time_per_batch_sec": float(train_epoch_time / max(1, epoch_train_batches)),
            "train_step_time_per_batch_sec": float(epoch_train_step_time / max(1, epoch_train_batches)),
            "validation_time_per_batch_sec": float(validation_epoch_time / max(1, epoch_val_batches)),
        }
        history.append(row)
        if epoch == 1 or epoch == int(cfg.epochs) or (epoch % max(1, int(cfg.print_every))) == 0:
            print(
                f"epoch {epoch:04d} | "
                f"train={row['train_loss']:.6e} val={row['val_loss']:.6e} "
                f"raw_val={row['val_raw_metric']:.6e} "
                f"eq={row['val_eq_l2']:.3e} ineq={row['val_ineq_l2']:.3e} | "
                f"train_batch={row['train_time_per_batch_sec']:.4f}s "
                f"val_batch={row['validation_time_per_batch_sec']:.4f}s"
            )

    train_time = time.perf_counter() - t0
    va_loss_j, va_raw_j, va_eq_j, va_ineq_j, va_hat, va_proj = eval_metrics(params, Xva_j, Yva_j)
    _sync((va_loss_j, va_raw_j, va_eq_j, va_ineq_j, va_hat, va_proj))
    all_hat, all_proj = model_forward(params, Xall_j)
    _sync((all_hat, all_proj))

    sample_train_x = jnp.asarray(warm_train[0], dtype=train_dtype)
    sample_train_y = jnp.asarray(warm_train[1], dtype=train_dtype)
    sample_val_x = jnp.asarray(warm_val[0], dtype=train_dtype)
    sample_train_hat = backbone_forward_fn(params, sample_train_x)
    sample_val_hat = backbone_forward_fn(params, sample_val_x)
    sample_grads = grad_only_fn(params, sample_train_x, sample_train_y)
    _sync((sample_train_hat, sample_val_hat, sample_grads))

    backbone_train_t = _measure_time(backbone_forward_fn, params, sample_train_x)
    backbone_val_t = _measure_time(backbone_forward_fn, params, sample_val_x)
    projection_train_t = _measure_time(projection_only_fn, params, sample_train_x, sample_train_hat)
    projection_val_t = _measure_time(projection_only_fn, params, sample_val_x, sample_val_hat)
    forward_total_t = _measure_time(forward_loss_fn, params, sample_train_x, sample_train_y)
    grad_total_t = _measure_time(grad_only_fn, params, sample_train_x, sample_train_y)
    optimizer_t = _measure_time(optimizer_update_fn, params, opt_state, sample_grads)
    backward_t = max(0.0, grad_total_t - forward_total_t)

    backbone_total = backbone_train_t * train_batch_count_total + backbone_val_t * validation_batch_count_total
    projection_total = projection_train_t * train_batch_count_total + projection_val_t * validation_batch_count_total
    backward_total = backward_t * train_batch_count_total
    optimizer_total = optimizer_t * train_batch_count_total
    component_total = backbone_total + projection_total + backward_total + optimizer_total
    component_percent = {
        "backbone": 100.0 * backbone_total / component_total if component_total > 0.0 else 0.0,
        "projection": 100.0 * projection_total / component_total if component_total > 0.0 else 0.0,
        "backprop": 100.0 * backward_total / component_total if component_total > 0.0 else 0.0,
        "optimizer": 100.0 * optimizer_total / component_total if component_total > 0.0 else 0.0,
    }
    timing_profile = {
        "training_wall_time_sec": float(train_time),
        "train_step_time_total_sec": float(train_step_time_total),
        "train_eval_time_total_sec": float(train_eval_time_total),
        "validation_time_total_sec": float(validation_time_total),
        "avg_train_time_per_epoch_sec": float(train_step_time_total / max(1, int(cfg.epochs))),
        "avg_validation_time_per_epoch_sec": float(validation_time_total / max(1, int(cfg.epochs))),
        "avg_train_time_per_batch_sec": float(train_step_time_total / max(1, train_batch_count_total)),
        "avg_validation_time_per_batch_sec": float(validation_time_total / max(1, validation_batch_count_total)),
        "train_batches_total": int(train_batch_count_total),
        "validation_batches_total": int(validation_batch_count_total),
        "profiled_batch_times_sec": {
            "backbone_train": float(backbone_train_t),
            "backbone_validation": float(backbone_val_t),
            "projection_train": float(projection_train_t),
            "projection_validation": float(projection_val_t),
            "forward_total_train": float(forward_total_t),
            "grad_total_train": float(grad_total_t),
            "backprop_estimated_train": float(backward_t),
            "optimizer_train": float(optimizer_t),
        },
        "component_time_total_estimated_sec": {
            "backbone": float(backbone_total),
            "projection": float(projection_total),
            "backprop": float(backward_total),
            "optimizer": float(optimizer_total),
        },
        "component_time_percent": component_percent,
    }

    inverse_summary = None
    if inverse_mode:
        estimated = np.asarray(jax.device_get(params["inverse"]), dtype=np.float64).reshape(-1)
        actual = inverse_labels_np
        rows = []
        for idx, name in enumerate(inverse_names):
            actual_value = None if actual is None else float(actual[idx])
            estimated_value = float(estimated[idx])
            rows.append(
                {
                    "name": str(name),
                    "actual": actual_value,
                    "estimated": estimated_value,
                    "error": None if actual is None else float(estimated_value - actual_value),
                    "abs_error": None if actual is None else float(abs(estimated_value - actual_value)),
                }
            )
        inverse_summary = {
            "names": inverse_names,
            "initial": inverse_init_np,
            "actual": actual,
            "estimated": estimated,
            "comparison": rows,
        }

    parameter_names, variable_names = _named_columns(metadata, data_n_x=data_n_x, n_y=dims["n_y"])
    final = history[-1]
    out = {
        "task": task_name,
        "dims": dims,
        "config": asdict(cfg),
        "metadata": metadata or {},
        "history": history,
        "final": final,
        "final_metrics": final,
        "training_wall_time_sec": train_time,
        "timing_profile": timing_profile,
        "inverse_parameters": inverse_summary,
        "params": params,
        "column_names": {
            "parameters": parameter_names,
            "variables": variable_names,
        },
        "predictions": {
            "X": X_np,
            "Y": Y_true_np,
            "Y_hat": np.asarray(all_hat),
            "Y_proj": np.asarray(all_proj),
            "train_indices": train_idx,
            "validation_indices": val_idx,
        },
        "val_predictions": {
            "X": np.asarray(Xva_j),
            "Y": None if Y_true_np is None else np.asarray(Yva_j),
            "Y_hat": np.asarray(va_hat),
            "Y_proj": np.asarray(va_proj),
            "sample_indices": val_idx,
        },
    }

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        history_file = output / "history.csv"
        predictions_file = output / "predictions.csv"
        weights_file = output / "model_weights.npz"
        inverse_file = output / "inverse_comparison.json"
        _write_history_csv(history_file, history)
        _write_predictions_csv(
            predictions_file,
            X=out["predictions"]["X"],
            Y=out["predictions"]["Y"],
            Y_hat=out["predictions"]["Y_hat"],
            Y_proj=out["predictions"]["Y_proj"],
            train_idx=train_idx,
            val_idx=val_idx,
            parameter_names=parameter_names,
            variable_names=variable_names,
        )
        weights_manifest = _save_model_weights(weights_file, params, inverse_mode=inverse_mode)
        artifacts = {
            "history": history_file.name,
            "summary": "summary.json",
            "model_weights": weights_file.name,
            "predictions": predictions_file.name,
        }
        if (output / "config.json").exists():
            artifacts["config"] = "config.json"
        if inverse_summary is not None:
            artifacts["inverse_comparison"] = inverse_file.name
        summary = {
            k: v
            for k, v in out.items()
            if k not in {"params", "val_predictions", "history", "predictions"}
        }
        summary["model_weights"] = weights_manifest
        summary["artifacts"] = artifacts
        summary["metrics_at_end"] = final
        summary["epoch_and_batch_timing"] = {
            "final_train_epoch_time_sec": final["train_epoch_time_sec"],
            "final_validation_epoch_time_sec": final["validation_epoch_time_sec"],
            "final_train_time_per_batch_sec": final["train_time_per_batch_sec"],
            "final_train_step_time_per_batch_sec": final["train_step_time_per_batch_sec"],
            "final_validation_time_per_batch_sec": final["validation_time_per_batch_sec"],
            "avg_train_time_per_epoch_sec": timing_profile["avg_train_time_per_epoch_sec"],
            "avg_validation_time_per_epoch_sec": timing_profile["avg_validation_time_per_epoch_sec"],
            "avg_train_time_per_batch_sec": timing_profile["avg_train_time_per_batch_sec"],
            "avg_validation_time_per_batch_sec": timing_profile["avg_validation_time_per_batch_sec"],
        }
        with open(output / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(_json_safe(summary), fh, indent=2, sort_keys=True)
        if inverse_summary is not None:
            with open(inverse_file, "w", encoding="utf-8") as fh:
                json.dump(_json_safe(inverse_summary), fh, indent=2, sort_keys=True)
        out["output_dir"] = str(output)
        out["model_weights_manifest"] = weights_manifest
        print(f"Saved run artifacts: {output}")

    print(f"Training time: {train_time:.3f}s")
    print("=== Timing summary ===")
    print(f"Train step time total      : {train_step_time_total:.3f}s")
    print(f"Validation time total      : {validation_time_total:.3f}s")
    print(f"Avg train time / epoch     : {timing_profile['avg_train_time_per_epoch_sec']:.4f}s")
    print(f"Avg validation time / epoch: {timing_profile['avg_validation_time_per_epoch_sec']:.4f}s")
    print(f"Avg train time / batch     : {timing_profile['avg_train_time_per_batch_sec']:.4f}s")
    print(f"Avg validation time / batch: {timing_profile['avg_validation_time_per_batch_sec']:.4f}s")
    print("=== Profiled component time distribution ===")
    print(f"Backbone forward : {component_percent['backbone']:6.2f}%")
    print(f"Projection       : {component_percent['projection']:6.2f}%")
    print(f"Backprop         : {component_percent['backprop']:6.2f}%")
    print(f"Optimizer update : {component_percent['optimizer']:6.2f}%")
    if inverse_summary is not None:
        print("=== Inverse parameter comparison ===")
        for row in inverse_summary["comparison"]:
            actual_txt = "n/a" if row["actual"] is None else f"{row['actual']:.8g}"
            err_txt = "n/a" if row["error"] is None else f"{row['error']:+.3e}"
            print(f"{row['name']}: actual={actual_txt} estimated={row['estimated']:.8g} error={err_txt}")
    return out
