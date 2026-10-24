"""Complete-model persistence and restoration."""

from __future__ import annotations
import csv
import json
import warnings
from datetime import datetime
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from ohagi.export.artifacts import artifact_paths
from kkthn.serialization.problem import load_sparse_structure
from kkthn.serialization.problem import load_symbolic_problem
from kkthn.serialization.problem import save_sparse_structure
from kkthn.serialization.problem import symbolic_problem_spec
from ohagi.losses import Loss
from ohagi.model_types import ModelType
from ohagi.nn import MLP
from purin.solvers import prepare_projection
from purin.problems.problem import Problem as CoreProblem
from purin.execution.runtime import resolve_device
from kkthn.api._utils import _json_safe
from kkthn.api.configuration import _dtype
from ohagi.execution.inference import require_jax_backend
from kkthn.api.state import _TrainedState


def _save_training_result(self, result, config, *, state=None):
    """Write training reports and the shared model bundle for a completed KKTHN run."""
    root = Path(config["output_dir"]) / self.name / result["method"]
    paths = artifact_paths(root, create=True)
    canonical = self._canonical_for_metrics()
    sparse_manifest = save_sparse_structure(paths.projection_arrays, canonical)
    np.savez(
        paths.normalization,
        input_mean=np.asarray(result["input_mean"], dtype=np.float64),
        input_scale=np.asarray(result["input_scale"], dtype=np.float64),
    )
    rows = result["history"]
    with paths.history.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row}) if rows else []
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                [{key: row.get(key) for key in fieldnames} for row in rows]
            )
    with paths.metrics.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result["metrics"]), handle, indent=2, sort_keys=True)

    projection_payload = {
        "method": result["method"],
        "config": _json_safe(result["projection_config"]),
        "backend": result["backend"],
        "batch_size": result["batch_size"],
        "dtype": result["dtype_name"],
        "sparse_runtime": self._sparse_runtime_description(result["method"]),
        "sparse_structure": sparse_manifest,
        "dual_warm_start_schedule": result.get("dual_warm_start_schedule", "predicted"),
        "dual_warm_start_switched_epoch": result.get("dual_warm_start_switched_epoch"),
        "dual_pretrain_gamma": result.get("dual_pretrain_gamma", 1e-3),
    }
    manifest = {
        "model_name": self.name,
        "type": self.model_type.value,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "nn": self.nn.to_dict(),
        "loss": self.loss.to_dict(),
        "projection": projection_payload,
        "problem": symbolic_problem_spec(self),
        "training": _json_safe(config),
        "network": {
            "hidden_sizes": list(self.nn.hidden_sizes),
            "activation": self.nn.activation,
            "input_size": len(self.parameter_names),
            "visible_output_size": len(self.variable_names),
            "total_output_size": self.nn.output_size(
                n_y=len(self.variable_names),
                n_eq=sum(item.kind == "eq" for item in self.constraints),
                n_ineq=sum(item.kind == "ineq" for item in self.constraints),
            ),
            "latent_duals": bool(self.nn.predict_duals),
        },
        "metrics": _json_safe(result["metrics"]),
        "artifacts": {
            "normalization": paths.normalization.name,
            "sparse_structure": paths.projection_arrays.name,
            "history": paths.history.name,
            "metrics": paths.metrics.name,
        },
    }
    if state is None:
        raise RuntimeError(
            "A trained model state is required to save a JAX model bundle."
        )
    from ohagi.serialization import save_model
    from kkthn.api.artifacts import kkthn_graph

    save_model(
        paths.metadata,
        framework="kkthn",
        parameters=result["parameters"],
        definition=manifest,
        inference_spec=kkthn_graph(self, state=state),
    )
    result["output_dir"] = str(root)
    result["metadata_path"] = str(paths.metadata)
    self._metadata_path = str(paths.metadata)
    print("=" * 120)
    print(
        f"\nModel training time (projection stage) = {result['training_seconds']:.4f} seconds"
    )
    print("Model training finished!")
    print(
        "Model saved with task, NN, loss, symbolic constraints, latent-dual architecture, and sparse projection data!"
    )
    print(f"metadata_path = {paths.metadata}")
    print(f"run_dir = {root}")


def load(
    self,
    metadata_path,
    *,
    verbose=True,
    compile_on_load=True,
    backend="jax",
    cache_dir=None,
):
    """Restore a model for JAX prediction.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Shared OHAGI ``metadata.json``, with its model artifacts.
    verbose : bool, optional
        Print loading and compilation information.
    compile_on_load : bool, optional
        Warm up JAX prediction and gradients.
    backend : {"jax", "auto"}, optional
        JAX is the only inference engine. ``auto`` is a compatibility alias.
    cache_dir : str or pathlib.Path, optional
        Compatibility argument; ignored by JAX loading.

    Returns
    -------
    KKTHardNet
        This restored model. ``summary()`` reports the actual inference engine.
    """
    _inference_backend = require_jax_backend(backend)
    metadata_file = Path(metadata_path).expanduser().resolve()
    from ohagi.serialization import read_model, load_parameters

    payload = read_model(metadata_file, framework="kkthn")
    restored = load_symbolic_problem(payload["problem"])
    sparse_spec = payload["projection"]["sparse_structure"]
    eq_sp, ineq_sp = load_sparse_structure(
        metadata_file.parent / payload["artifacts"]["sparse_structure"], sparse_spec
    )
    restored.set_sparsity_override(eq_sp, ineq_sp)

    CoreProblem.__init__(self, name=restored.name)
    self.add_parameter(restored.parameter_names)
    self.add_variable(restored.variable_names)
    self.objective = restored.objective
    self.constraints.add(*list(restored.constraints))
    self.set_sparsity_override(eq_sp, ineq_sp)
    self.model_type = ModelType.normalize(
        payload.get("type", payload.get("task", "surrogate"))
    )
    self.nn = MLP.from_dict(
        payload.get(
            "nn",
            {
                "hidden_sizes": payload["network"]["hidden_sizes"],
                "activation": payload["network"].get("activation", "tanh"),
                "predict_duals": False,
            },
        )
    )
    self.loss = Loss.from_dict(payload.get("loss", {"kind": "mse"}))
    self._user_objective_defined = self.model_type is ModelType.OPTIMIZATION
    self.dataset_spec = None
    self._train_config = self._normalize_train_config(payload.get("training", {}))
    self.projection_method = payload["projection"]["method"]
    self.projection_config = dict(payload["projection"].get("config", {}))
    self._trained, self._last_method, self._metric_canonical = {}, None, None

    dtype = _dtype(payload["projection"].get("dtype", self._train_config["dtype"]))
    backbone = self.nn.build()
    parameters = load_parameters(metadata_file, payload)
    with np.load(
        metadata_file.parent / payload["artifacts"]["normalization"], allow_pickle=False
    ) as arrays:
        mean = np.asarray(arrays["input_mean"], dtype=np.float64)
        scale = np.asarray(arrays["input_scale"], dtype=np.float64)
    definition = self._projection_definition(
        self.projection_method, self.projection_config
    )
    batch_size = int(payload["projection"]["batch_size"])
    saved_device = str(payload["projection"].get("backend", "cpu"))
    requested_device = saved_device
    try:
        device = resolve_device(requested_device)
    except ValueError:
        device = resolve_device("cpu")
        warnings.warn(
            f"Saved device {requested_device!r} is unavailable; using CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
    backend = f"{device.platform}:{device.id}"
    parameters = jax.device_put(parameters, device)
    projector = prepare_projection(
        self,
        method=self.projection_method,
        config=definition,
        batch_size=batch_size,
        dtype=dtype,
        backend=backend,
    )
    result = {
        "model_name": self.name,
        "method": self.projection_method,
        "metrics": payload.get("metrics", {}),
        "history": [],
        "output_dir": str(metadata_file.parent),
        "metadata_path": str(metadata_file),
        "sparse_runtime": payload["projection"].get("sparse_runtime", {}),
    }
    state = _TrainedState(
        method=self.projection_method,
        backbone=backbone,
        architecture=self.nn,
        parameters=parameters,
        projector=projector,
        input_mean=mean,
        input_scale=scale,
        dtype=dtype,
        batch_size=batch_size,
        backend=backend,
        projection_config=self.projection_config,
        result=result,
        metadata_path=str(metadata_file),
    )
    self._trained[self.projection_method] = state
    self._last_method = self.projection_method
    self._metadata_path = str(metadata_file)
    state.result["inference_backend"] = "jax"
    if compile_on_load and state.inference_backend == "jax":
        # Compile the same batch-size-one forward map used by both
        # gradient=False and gradient=True.
        dummy_single = jnp.asarray(state.input_mean, dtype=state.dtype)

        if verbose:
            print("Compiling single-instance prediction during load...")

        single_prediction_fn = self._single_prediction_function(state)
        prediction_device = single_prediction_fn(dummy_single)
        jax.block_until_ready(prediction_device)

        if verbose:
            print("Compiling single-instance gradient prediction during load...")

        value_jacobian_fn = self._single_value_jacobian_function(state)
        value_device, jacobian_device = value_jacobian_fn(dummy_single)
        jax.block_until_ready(value_device)
        jax.block_until_ready(jacobian_device)

    if verbose:
        print("\n" + "=" * 120)
        print("Model Loaded Successfully")
        print("-" * 120)
        print(f"Model Name        : {self.name}")
        print(f"Type              : {self.model_type.value}")
        print(f"Projection Method : {self.projection_method}")
        print(f"Latent Duals      : {self.nn.predict_duals}")
        print(f"Inference Backend: {state.inference_backend}")
        print(f"Location          : {metadata_file.parent}")
        print("=" * 120 + "\n")
    return self
