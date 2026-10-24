"""Console reporting, history loading, plotting, and summaries."""

from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from purin.solvers import ProjectionMethod


def _format_metric(value, width=12):
    return f"{float(value):>{int(width)}.4e}"


def _print_training_header(self, *, method, config, X_train, X_validation):
    """Print the problem, backbone and solver summary before KKTHN training."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    width = 12
    n_eq = sum(item.kind == "eq" for item in self.constraints)
    n_ineq = sum(item.kind == "ineq" for item in self.constraints)
    output_size = self.nn.output_size(
        n_y=len(self.variable_names), n_eq=n_eq, n_ineq=n_ineq
    )
    print(f"Model training started! [{now}]\n")
    print("KKT-HardNet")
    print(
        f"  dims: n_x={len(self.parameter_names)} n_y={len(self.variable_names)} "
        f"n_eq={n_eq} n_ineq={n_ineq}"
    )
    print(f"  type: {self.model_type.value}")
    print(f"  projection: {ProjectionMethod.normalize(method).value}")
    print(
        f"  samples: train={len(X_train)} val={len(X_validation)} batch_size={config['batch_size']}"
    )
    print(
        f"  network: {[len(self.parameter_names), *self.nn.hidden_sizes, output_size]} "
        f"(visible y={len(self.variable_names)}, latent duals={output_size - len(self.variable_names)})"
    )
    if bool(config.get("pretrain", True)):
        print(f"  pretraining: enabled, epoch_mlp={config.get('epoch_mlp')}")
        print(
            "  dual pretrain box: "
            "min(-min(0,-z), gamma), "
            f"gamma={float(config['dual_pretrain_gamma']):.4e}"
        )
        print("  projection dual initialization: predicted from first projected epoch")
    else:
        print("  pretraining: disabled")
        print("  projection dual initialization: predicted from epoch 1")

    drop_last = bool(config.get("drop_last_batch", False))
    remainder = len(X_train) % int(config["batch_size"])
    if drop_last and remainder:
        print(
            f"  drop_last_batch: enabled ({remainder} shuffled training samples omitted per epoch)"
        )
    else:
        print(f"  drop_last_batch: {'enabled' if drop_last else 'disabled'}")
    table_width = 120
    group_width = width * 4 + 3
    print("=" * table_width)
    print(
        f"{'Epoch':>{width}} | {'Training':^{group_width}} | {'Validation':^{group_width}}"
    )
    print("-" * table_width)
    task_label = "Objective" if self.model_type.value == "optimization" else "Data"
    print(
        f"{'':>{width}} | "
        f"{'Loss':>{width}} {task_label:>{width}} "
        f"{'Eq':>{width}} {'Ineq':>{width}} | "
        f"{'Loss':>{width}} {task_label:>{width}} "
        f"{'Eq':>{width}} {'Ineq':>{width}}"
    )
    print("=" * table_width)


def _print_training_row(self, epoch, total, train, validation):
    width = 12
    print(
        f"{f'{epoch:03d}/{total:03d}':>{width}} | "
        f"{self._format_metric(train['loss'], width)} "
        f"{self._format_metric(train['data_loss'], width)} "
        f"{self._format_metric(train['eq'], width)} "
        f"{self._format_metric(train['ineq'], width)} | "
        f"{self._format_metric(validation['loss'], width)} "
        f"{self._format_metric(validation['data_loss'], width)} "
        f"{self._format_metric(validation['eq'], width)} "
        f"{self._format_metric(validation['ineq'], width)}"
    )


def _sparse_runtime_description(method):
    selected = ProjectionMethod.normalize(method)
    if selected is ProjectionMethod.ADMM:
        return {
            "matrix_free": True,
            "jacobians": "local-support sparse",
            "linear_solver": "warm-started PCG forward; GMRES adjoint",
        }
    if selected is ProjectionMethod.CHAMBOLLE_POCK:
        return {
            "matrix_free": True,
            "jacobians": "local-support sparse",
            "linear_solver": "none in forward; GMRES adjoint",
        }
    if selected is ProjectionMethod.FISCHER_BURMEISTER_NEWTON:
        return {
            "matrix_free": True,
            "jacobians": "fixed-pattern BCOO",
            "linear_solver": "BiCGSTAB",
        }
    if selected is ProjectionMethod.FISCHER_BURMEISTER_REDUCED:
        return {
            "matrix_free": True,
            "jacobians": "local-support sparse",
            "linear_solver": "auto Schur-CG/MINRES with GMRES fallback",
        }
    if selected is ProjectionMethod.PRIMAL_DUAL_INTERIOR_POINT:
        return {
            "matrix_free": True,
            "jacobians": "local-support sparse",
            "linear_solver": "GMRES",
        }
    return {
        "matrix_free": False,
        "jacobians": "affine equality structure",
        "linear_solver": "factorized solve",
    }


def _run_dir(self, method: str | None = None) -> Path:
    """Return the artifact directory for a trained or loaded model."""
    selected = method or self._last_method
    if selected is not None and selected in self._trained:
        result = self._trained[selected].result
        output_dir = result.get("output_dir")
        if output_dir:
            return Path(output_dir).expanduser().resolve()
        metadata_path = result.get("metadata_path")
        if metadata_path:
            return Path(metadata_path).expanduser().resolve().parent
    if self._metadata_path:
        return Path(self._metadata_path).expanduser().resolve().parent
    raise RuntimeError(
        "No training artifact directory is available. Train or load a model first."
    )


def _read_history(self, method: str | None = None) -> dict[str, np.ndarray]:
    """Read epoch history from memory or the saved ``history.csv`` artifact."""
    selected = method or self._last_method
    if selected is None or selected not in self._trained:
        raise RuntimeError("Train or load a model before calling plot_history().")

    rows = list(self._trained[selected].result.get("history", []) or [])
    if not rows:
        history_path = self._run_dir(selected) / "history.csv"
        if not history_path.exists():
            raise RuntimeError(f"Training history is unavailable: {history_path}.")
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("No training history is available to plot.")

    history = _decode_history_columns(rows)
    return history


def _history_column(history: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Return a history column while supporting old and new field names."""
    aliases = {
        "tr_mse": ("tr_mse", "training_loss", "train_loss", "raw_mse"),
        "val_mse": ("val_mse", "validation_loss", "val_loss"),
        "tr_eq": ("tr_eq", "training_eq", "train_eq"),
        "tr_ineq": ("tr_ineq", "training_ineq", "train_ineq"),
        "val_eq": ("val_eq", "validation_eq", "val_eq"),
        "val_ineq": ("val_ineq", "validation_ineq", "val_ineq"),
    }
    for candidate in aliases.get(name, (name,)):
        if candidate in history:
            values = np.asarray(history[candidate], dtype=np.float64)
            if values.ndim != 1:
                raise RuntimeError(
                    f"History column {candidate!r} must be one-dimensional."
                )
            return values
    raise RuntimeError(
        f"Required history column {name!r} is unavailable; available columns are {sorted(history)}."
    )


def _inverse_history_columns(
    history: Mapping[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Return optional inverse-parameter series saved with ``inverse_*`` names."""
    result = []
    for key in sorted(history):
        if not key.startswith("inverse_") or key.startswith("inverse_actual_"):
            continue
        values = np.asarray(history[key], dtype=np.float64)
        if values.ndim == 1 and np.any(np.isfinite(values)):
            result.append((key.removeprefix("inverse_"), values))
    return result


def _inverse_actual_values_for_plot(
    self, inverse_names: Sequence[str]
) -> list[float | None]:
    """Return optional actual inverse-parameter values when present."""
    values = getattr(self, "inverse_actual_values", None)
    if values is None:
        return [None for _ in inverse_names]
    if isinstance(values, Mapping):
        return [values.get(name) for name in inverse_names]
    sequence = list(values)
    return [
        sequence[index] if index < len(sequence) else None
        for index, _name in enumerate(inverse_names)
    ]


def plot_history(
    self, *, show: bool = True, save_dir: str | Path | None = None, bg: str = "grey"
):
    """Plot and save loss plus mean-violation or inverse-parameter histories."""
    history = self._read_history()
    epochs = np.asarray(history["epoch"])
    tr_mse = self._history_column(history, "tr_mse")
    val_mse = self._history_column(history, "val_mse")
    inverse_history = self._inverse_history_columns(history)
    if not inverse_history:
        tr_violation = np.maximum(
            self._history_column(history, "tr_eq"),
            self._history_column(history, "tr_ineq"),
        )
        val_violation = np.maximum(
            self._history_column(history, "val_eq"),
            self._history_column(history, "val_ineq"),
        )
        tr_violation = np.maximum(tr_violation, 1e-16)
        val_violation = np.maximum(val_violation, 1e-16)

    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as error:
        raise ImportError("matplotlib is required for plot_history().") from error

    font_names = {font.name for font in font_manager.fontManager.ttflist}
    font_family = "Arial" if "Arial" in font_names else "DejaVu Sans"
    bg_colors = {"grey": "#E5E5E5", "gray": "#E5E5E5", "white": "#FFFFFF"}
    fig_bg = bg_colors.get(str(bg).lower(), "#E5E5E5")
    style = "default" if str(bg).lower() == "white" else "ggplot"

    with plt.style.context(style):
        plt.rcParams.update(
            {
                "font.family": font_family,
                "font.size": 32,
                "axes.titlesize": 32,
                "axes.labelsize": 24,
                "legend.fontsize": 24,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
            }
        )
        fig, axes = plt.subplots(1, 2, figsize=(20, 6), facecolor=fig_bg)
        for axis in axes:
            axis.set_facecolor(fig_bg)

        axes[0].plot(epochs, tr_mse, linewidth=2, label="Train MSE")
        axes[0].plot(
            epochs, val_mse, linewidth=2, linestyle="--", label="Validation MSE"
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if inverse_history:
            inverse_names = []
            for name, values in inverse_history:
                inverse_names.append(str(name))
                axes[1].plot(epochs, values, linewidth=2, label=str(name))
            actual_values = self._inverse_actual_values_for_plot(inverse_names)
            for actual_value in actual_values:
                if actual_value is not None and np.isfinite(actual_value):
                    axes[1].axhline(actual_value, linewidth=2, linestyle=":")
            axes[1].set_ylabel("Value")
            axes[1].set_title("Inverse Parameters")
        else:
            axes[1].plot(epochs, tr_violation, linewidth=2, label="Train Violation")
            axes[1].plot(
                epochs,
                val_violation,
                linewidth=2,
                linestyle="--",
                label="Validation Violation",
            )
            axes[1].set_ylabel("Mean Violation")
            axes[1].set_title("Violation")
            axes[1].set_yscale("log")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        target = (
            self._run_dir()
            if save_dir is None
            else Path(save_dir).expanduser().resolve()
        )
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            target / "training_history.png",
            dpi=600,
            bbox_inches="tight",
            facecolor=fig_bg,
        )
        if show:
            plt.show()
    return fig, axes


def summary(self):
    """Print and return the model configuration and available metrics.

    Returns
    -------
    dict
        Model details, constraint violations, artifact location, and the actual
        JAX inference backend.
    """
    state = None if self._last_method is None else self._trained.get(self._last_method)
    metrics = {} if state is None else state.result.get("metrics", {})
    summary = {
        "name": self.name,
        "type": self.model_type.value,
        "nn": self.nn.to_dict(),
        "loss": self.loss.to_dict(),
        "projection_method": self._last_method,
        "parameters": tuple(self.parameter_names),
        "variables": tuple(self.variable_names),
        "mean_equality_violation": float(metrics.get("eq", 0.0) or 0.0),
        "mean_inequality_violation": float(metrics.get("ineq", 0.0) or 0.0),
        "maximum_equality_violation": float(metrics.get("eq_max", 0.0) or 0.0),
        "maximum_inequality_violation": float(metrics.get("ineq_max", 0.0) or 0.0),
        "metadata_path": self._metadata_path,
        "inference_backend": "jax" if state is None else state.inference_backend,
    }
    print("KKT-HardNet Summary")
    print("-" * 60)
    for key in (
        "name",
        "type",
        "projection_method",
        "mean_equality_violation",
        "mean_inequality_violation",
        "maximum_equality_violation",
        "maximum_inequality_violation",
        "metadata_path",
        "inference_backend",
    ):
        print(f"{key:<34}: {summary[key]}")
    print("-" * 60)
    return summary


def _decode_history_columns(rows):
    """Decode epoch columns using the existing numeric and string rules."""
    keys = sorted({key for row in rows for key in row})
    history: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        if key in {"stage"}:
            history[key] = np.asarray(
                ["" if value is None else str(value) for value in values], dtype=object
            )
            continue
        converted = []
        numeric = True
        for value in values:
            if value is None or value == "":
                converted.append(np.nan)
                continue
            try:
                converted.append(float(value))
            except (TypeError, ValueError):
                numeric = False
                break
        if numeric:
            array = np.asarray(converted, dtype=np.float64)
            if key == "epoch" and np.all(np.isfinite(array)):
                array = array.astype(np.int64)
            history[key] = array
        else:
            history[key] = np.asarray(
                ["" if value is None else str(value) for value in values], dtype=object
            )
    return history
