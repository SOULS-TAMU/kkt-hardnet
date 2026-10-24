"""Stable public ``KKTHardNet`` facade over the modular API services."""

from __future__ import annotations
from purin.problems.problem import Problem as CoreProblem
from kkthn.api import composition
from kkthn.api import configuration
from kkthn.api import data
from kkthn.api import deployment
from kkthn.api import fit
from kkthn.api import network
from kkthn.api import objectives
from kkthn.api import persistence
from kkthn.api import predict
from kkthn.api import reporting
from kkthn.api import resume


class KKTHardNet(CoreProblem):
    """Compose, train, persist, load, and evaluate a KKT-HardNet model.

    The class intentionally remains the stable user-facing facade. Its method
    implementations live in focused modules under :mod:`kkthn.api`.
    """

    __init__ = composition.__init__
    _attach_projection = composition._attach_projection
    _validate_composition = composition._validate_composition
    as_projection_problem = composition.as_projection_problem
    dataset = composition.dataset
    set_dataset = dataset
    use_dataset = dataset
    set_train_config = composition.set_train_config
    set_projection_config = composition.set_projection_config
    extract = composition.extract

    _normalize_mapping = staticmethod(configuration._normalize_mapping)
    _normalize_train_config = staticmethod(configuration._normalize_train_config)
    _projection_definition = configuration._projection_definition

    _target_names = data._target_names
    _observation_indices = data._observation_indices
    _load_dataset = data._load_dataset
    _split = data._split

    _network_setup = network._network_setup
    _network_outputs = network._network_outputs
    _network_pretraining_outputs = network._network_pretraining_outputs
    _network_primal_output = network._network_primal_output

    _canonical_for_metrics = fit._canonical_for_metrics
    _task_value = fit._task_value
    _output_metrics = fit._output_metrics
    _raw_loss_builder = fit._raw_loss_builder
    _raw_steps = fit._raw_steps
    _pretrain = fit._pretrain
    _projection_steps = fit._projection_steps
    _evaluate_projector = fit._evaluate_projector
    _prepare_training = fit._prepare_training
    _train_one_method = fit._train_one_method
    model = fit.model
    resume_training = resume.resume_training
    compare_projections = fit.compare_projections
    model_all = compare_projections

    _save_training_result = persistence._save_training_result
    load = persistence.load

    def save_checkpoint(self, path):
        """Save parameters and optimizer state using optional Orbax."""
        from ohagi.io.orbax import save_model_checkpoint

        return save_model_checkpoint(self, path)

    def restore_checkpoint(self, path):
        """Restore into this equivalent trained/built model."""
        from ohagi.io.orbax import restore_model_checkpoint

        return restore_model_checkpoint(self, path)

    _predict_batch_jax = predict._predict_batch_jax
    _gradient_projector = predict._gradient_projector
    _single_gradient_projector = predict._single_gradient_projector
    _single_prediction_core = predict._single_prediction_core
    _single_prediction_function = predict._single_prediction_function
    _single_value_jacobian_function = predict._single_value_jacobian_function
    predict = predict.predict

    _validate_objective_problem = objectives._validate_objective_problem
    _compile_single_objective_gradient = objectives._compile_single_objective_gradient
    _compile_batch_objective_gradient = objectives._compile_batch_objective_gradient
    prepare_objective_gradient = objectives.prepare_objective_gradient
    objective_gradient = objectives.objective_gradient

    export_predictor = deployment.export_predictor

    def export_inference(self, metadata_path=None):
        """Export the complete batch-one graph for shared metadata loading."""
        from ohagi.inference import export
        from kkthn.api.artifacts import kkthn_graph

        if self._last_method not in self._trained:
            raise RuntimeError("Train or load the model before exporting inference.")
        path = metadata_path or self._metadata_path
        if path is None:
            raise ValueError("Provide a destination metadata.json path.")
        return export(path, **kkthn_graph(self))

    @staticmethod
    def load_inference(metadata_path, *, device="cpu"):
        """Prepare the shared JAX predictor from a metadata.json path."""
        from ohagi.inference import load

        return load(metadata_path, device=device)

    _format_metric = staticmethod(reporting._format_metric)
    _print_training_header = reporting._print_training_header
    _print_training_row = reporting._print_training_row
    _sparse_runtime_description = staticmethod(reporting._sparse_runtime_description)
    _run_dir = reporting._run_dir
    _read_history = reporting._read_history
    _history_column = staticmethod(reporting._history_column)
    _inverse_history_columns = staticmethod(reporting._inverse_history_columns)
    _inverse_actual_values_for_plot = reporting._inverse_actual_values_for_plot
    plot_history = reporting.plot_history
    summary = reporting.summary

    optimize = composition.optimize
    estimate = composition.estimate

    def _checkpoint_payload(self):
        state = self._trained.get(self._last_method)
        if state is None:
            raise RuntimeError("Train or load the KKTHN model first.")
        tree = {
            "parameters": state.parameters,
            "optimizer_state": state.result.get("optimizer_state"),
            "input_mean": state.input_mean,
            "input_scale": state.input_scale,
        }
        metadata = {"framework": "kkthn"}
        if state.result.get("resume_state") is not None:
            from ._utils import _json_safe

            metadata["resume"] = _json_safe(state.result["resume_state"])
        return tree, metadata

    def _restore_checkpoint_payload(self, tree, metadata):
        state = self._trained[self._last_method]
        state.parameters = tree["parameters"]
        state.result.update(
            parameters=state.parameters, optimizer_state=tree["optimizer_state"]
        )
        if "resume" in metadata:
            state.result["resume_state"] = metadata["resume"]
        state.input_mean, state.input_scale = tree["input_mean"], tree["input_scale"]
        state.result.update(input_mean=state.input_mean, input_scale=state.input_scale)
        for name in (
            "single_prediction_core",
            "single_prediction_fn",
            "single_value_jacobian_fn",
            "objective_gradient_fns",
        ):
            setattr(state, name, None)


ProblemBuilder = KKTHardNet

__all__ = ["KKTHardNet", "ProblemBuilder"]
