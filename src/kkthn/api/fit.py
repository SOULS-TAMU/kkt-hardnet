"""High-level pretraining and projected-training workflow."""

from __future__ import annotations
import time
import warnings
from dataclasses import asdict, is_dataclass
from typing import Iterable
import jax
import jax.numpy as jnp
import numpy as np
from ohagi.losses import configured_training_loss
from ohagi.losses import masked_mean
from ohagi.losses import mse
from ohagi.model_types import ModelType
from purin.solvers import ProjectionMethod
from purin.solvers import prepare_projection
from ohagi.training import Adam
from ohagi.training import make_eval_step
from ohagi.training import make_train_step
from kkthn.api._utils import _copy_tree, _json_safe
from kkthn.api.data import _fixed_batches
from kkthn.api.state import _TrainedState
from .composition import projected_model
from .execution import epoch_plan
from ohagi.training.engine import Engine
from ohagi.training.runner import EpochTransition


def _canonical_for_metrics(self):
    if self._metric_canonical is None:
        if self.objective is None:
            self.objective = 0.0
        self._metric_canonical = self.compile()
    return self._metric_canonical


def _task_value(self, X, Y, values, dtype):
    canonical = self._canonical_for_metrics()
    theta = jnp.asarray(X, dtype=dtype)
    y = jnp.asarray(values, dtype=dtype)
    if self.model_type is ModelType.OPTIMIZATION:
        result = jax.vmap(canonical.objective)(theta, y)
        return float(jax.device_get(jnp.mean(result)))
    indices = self._observation_indices()
    prediction = y if indices is None else y[:, jnp.asarray(indices)]
    return float(
        jax.device_get(jnp.mean(jnp.square(prediction - jnp.asarray(Y, dtype=dtype))))
    )


def _output_metrics(self, X, Y, outputs, dtype):
    canonical = self._canonical_for_metrics()
    theta = jnp.asarray(X, dtype=dtype)
    values = jnp.asarray(outputs, dtype=dtype)
    equalities = jax.vmap(canonical.equalities)(theta, values)
    inequalities = jax.vmap(canonical.inequalities)(theta, values)
    eq = (
        jnp.mean(jnp.linalg.norm(equalities, axis=1))
        if canonical.equality_size
        else jnp.asarray(0.0, dtype=dtype)
    )
    ineq = (
        jnp.mean(jnp.linalg.norm(jnp.maximum(inequalities, 0.0), axis=1))
        if canonical.inequality_size
        else jnp.asarray(0.0, dtype=dtype)
    )
    eq_max = (
        jnp.max(jnp.abs(equalities))
        if canonical.equality_size
        else jnp.asarray(0.0, dtype=dtype)
    )
    ineq_max = (
        jnp.max(jnp.maximum(inequalities, 0.0))
        if canonical.inequality_size
        else jnp.asarray(0.0, dtype=dtype)
    )
    task = self._task_value(X, Y, outputs, dtype)
    eq, ineq, eq_max, ineq_max = jax.device_get((eq, ineq, eq_max, ineq_max))
    return {
        "loss": task,
        "eq": float(eq),
        "ineq": float(ineq),
        "eq_max": float(eq_max),
        "ineq_max": float(ineq_max),
    }


def _batch_violation_metrics(canonical, theta, values, mask, dtype):
    """Return scalar mean equality and inequality violations for one batch.

    These metrics are computed inside the already-running train/evaluation graph,
    so recording history does not require another projection pass over the
    training set.
    """
    mask = jnp.asarray(mask, dtype=bool)
    zero = jnp.asarray(0.0, dtype=dtype)

    if canonical.equality_size:
        equalities = jax.vmap(canonical.equalities)(theta, values)
        equality_per_sample = jnp.linalg.norm(equalities, axis=1)
        equality_violation = masked_mean(equality_per_sample, mask)
    else:
        equality_violation = zero

    if canonical.inequality_size:
        inequalities = jax.vmap(canonical.inequalities)(theta, values)
        inequality_per_sample = jnp.linalg.norm(jnp.maximum(inequalities, zero), axis=1)
        inequality_violation = masked_mean(inequality_per_sample, mask)
    else:
        inequality_violation = zero

    return equality_violation, inequality_violation


def _aggregate_epoch_metrics(metric_rows, physical_counts, keys):
    """Aggregate scalar device metrics with one host synchronization per epoch."""
    if not metric_rows:
        return {key: 0.0 for key in keys}

    first_value = None
    for row in metric_rows:
        for key in keys:
            if key in row:
                first_value = jnp.asarray(row[key])
                break
        if first_value is not None:
            break
    if first_value is None:
        return {key: 0.0 for key in keys}

    weights = jnp.asarray(physical_counts, dtype=first_value.dtype)
    denominator = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, weights.dtype))
    aggregated = {}
    for key in keys:
        values = jnp.stack(
            [jnp.asarray(row[key], dtype=first_value.dtype) for row in metric_rows]
        )
        aggregated[key] = jnp.sum(values * weights) / denominator

    host = jax.device_get(aggregated)
    return {key: float(np.asarray(value)) for key, value in host.items()}


def _evaluate_scalar_batches(
    parameters, eval_step, X, Y, batch_size, dtype, metric_keys
):
    """Evaluate only scalar metrics; never materialize full model outputs."""
    rows = []
    counts = []
    for xb, yb, mask, count in _fixed_batches(X, Y, batch_size, shuffle=False, seed=0):
        y_payload = (
            jnp.zeros((len(xb), 0), dtype=dtype)
            if yb is None
            else jnp.asarray(yb, dtype=dtype)
        )
        result = eval_step(
            parameters, (jnp.asarray(xb, dtype=dtype), y_payload, jnp.asarray(mask))
        )
        # Projected evaluation returns ``(outputs, metrics)`` so the same JIT
        # executable can serve scalar validation, readiness, and final output
        # collection. Raw evaluation returns the metrics mapping directly.
        rows.append(result[1] if isinstance(result, tuple) else result)
        counts.append(count)
    return _aggregate_epoch_metrics(rows, counts, metric_keys)


def _lightweight_history_metrics(metrics):
    """Normalize batch or detailed metrics to the history/logging schema."""
    task = metrics.get("task_loss", metrics.get("loss", 0.0))
    total = metrics.get("total_loss", metrics.get("loss", task))
    result = {
        # ``loss`` is retained as the historical task/data-loss key.
        "loss": float(task),
        "data_loss": float(task),
        "total_loss": float(total),
        "eq": float(metrics.get("eq", 0.0)),
        "ineq": float(metrics.get("ineq", 0.0)),
    }
    for key in (
        "primal_consistency_loss",
        "equality_dual_consistency_loss",
        "inequality_dual_consistency_loss",
    ):
        if key in metrics:
            result[key] = float(metrics[key])
    return result


def _raw_loss_builder(self):
    canonical = self._canonical_for_metrics()
    observation_indices = self._observation_indices()

    def loss_function(outputs, batch):
        raw = outputs["raw"]
        mask = batch.get("mask")
        if self.model_type is ModelType.OPTIMIZATION:
            values = jax.vmap(canonical.objective)(batch["theta"], raw)
            task = masked_mean(values, mask)
        else:
            prediction = raw
            if observation_indices is not None:
                prediction = prediction[:, jnp.asarray(observation_indices)]
            task = mse(prediction, batch["solution"], mask)
        return task, {"task_loss": task}

    return loss_function


def _raw_steps(self, backbone, optimizer, mean, scale, dtype, gamma):
    canonical = self._canonical_for_metrics()
    loss_builder = self._raw_loss_builder()
    model = projected_model(self, backbone, mean, scale, dtype, gamma=gamma)

    def loss_function(parameters, batch):
        X, Y, mask = batch
        outputs, _ = model.apply(parameters, {}, X)
        raw_y = outputs["raw"]
        loss, metrics = loss_builder(
            {"raw": raw_y}, {"theta": X, "solution": Y, "mask": mask}
        )
        eq, ineq = _batch_violation_metrics(canonical, X, raw_y, mask, dtype)
        return loss, {
            **metrics,
            "eq": eq,
            "ineq": ineq,
            "gradient_valid": jnp.all(jnp.isfinite(raw_y)),
        }

    train_step = jax.jit(make_train_step(loss_function, optimizer))
    eval_step = jax.jit(make_eval_step(loss_function))
    return train_step, eval_step


def _pretrain(
    self,
    *,
    backbone,
    parameters,
    optimizer,
    X_train,
    Y_train,
    X_validation,
    Y_validation,
    mean,
    scale,
    dtype,
    config,
):
    """Run KKTHN's pretraining phase using prepared shared training resources."""
    gamma = float(config["dual_pretrain_gamma"])
    train_step, eval_step = self._raw_steps(
        backbone, optimizer, mean, scale, dtype, gamma
    )
    optimizer_state = optimizer.init(parameters)
    history = []
    maximum = min(int(config["epoch_mlp"]), int(config["epochs"]))
    metric_keys = ("loss", "task_loss", "eq", "ineq")

    def raw_epoch(epoch, state):
        parameters, optimizer_state = state
        result = yield epoch_plan(
            parameters,
            optimizer_state,
            _fixed_batches(
                X_train,
                Y_train,
                int(config["batch_size"]),
                shuffle=True,
                seed=int(config["seed"]) + epoch,
                drop_last=bool(config["drop_last_batch"]),
            ),
            train_step,
            dtype,
        )
        parameters, optimizer_state = result.state
        batch_metrics, batch_counts = result.metrics, result.counts

        # Training metrics are reused from the train step. Validation performs
        # one scalar-only forward pass and transfers only a few scalars.
        train_metrics = _aggregate_epoch_metrics(
            batch_metrics, batch_counts, metric_keys
        )
        val_metrics = _evaluate_scalar_batches(
            parameters,
            eval_step,
            X_validation,
            Y_validation,
            int(config["batch_size"]),
            dtype,
            metric_keys,
        )
        train_view = _lightweight_history_metrics(train_metrics)
        val_view = _lightweight_history_metrics(val_metrics)
        history.append(
            {
                "stage": "raw",
                "epoch": epoch,
                "training_loss": train_view["loss"],
                "training_data_loss": train_view["data_loss"],
                "training_total_loss": train_view["total_loss"],
                "training_eq": train_view["eq"],
                "training_ineq": train_view["ineq"],
                "validation_loss": val_view["loss"],
                "validation_data_loss": val_view["data_loss"],
                "validation_total_loss": val_view["total_loss"],
                "validation_eq": val_view["eq"],
                "validation_ineq": val_view["ineq"],
            }
        )

        should_print = (
            epoch == 1
            or epoch == maximum
            or epoch % max(1, int(config["print_every"])) == 0
        )
        if should_print:
            self._print_training_row(
                epoch,
                int(config["epochs"]),
                {
                    "loss": train_view["total_loss"],
                    "data_loss": train_view["data_loss"],
                    "eq": train_view["eq"],
                    "ineq": train_view["ineq"],
                },
                {
                    "loss": val_view["total_loss"],
                    "data_loss": val_view["data_loss"],
                    "eq": val_view["eq"],
                    "ineq": val_view["ineq"],
                },
            )

        stop = config.get("eta") is not None and train_view["loss"] <= float(
            config["eta"]
        )
        return EpochTransition((parameters, optimizer_state), stop)

    result = Engine.run((parameters, optimizer_state), maximum, raw_epoch)
    parameters, optimizer_state = result.state
    return parameters, optimizer_state, history, result.completed_epoch


def _projection_steps(self, backbone, optimizer, projector, mean, scale, dtype):
    canonical = self._canonical_for_metrics()
    loss_builder = configured_training_loss(
        self.loss,
        objective=canonical.objective,
        observation_indices=self._observation_indices(),
    )

    shared_model = projected_model(self, backbone, mean, scale, dtype, projector)

    def evaluate_with_outputs(parameters, batch):
        X, Y, mask = batch
        outputs, _ = shared_model.apply(parameters, {}, X)
        raw_y, lambda_raw, mu_raw = (
            outputs["raw"],
            outputs["lambda_raw"],
            outputs["mu_raw"],
        )
        output = outputs["projection_result"]
        loss, metrics = loss_builder(
            {
                "raw": raw_y,
                "projected": output.y_tilde,
                "lambda_raw": lambda_raw,
                "lambda_projected": output.multipliers_eq,
                "mu_raw": mu_raw,
                "mu_projected": output.multipliers_ineq,
                "dual_valid": output.forward_gradient_eligible,
            },
            {"theta": X, "solution": Y, "mask": mask},
        )
        eq, ineq = _batch_violation_metrics(canonical, X, output.y_tilde, mask, dtype)
        finite = jnp.all(jnp.isfinite(raw_y)) & jnp.all(jnp.isfinite(output.y_tilde))
        scalar_metrics = {
            **metrics,
            "loss": loss,
            "eq": eq,
            "ineq": ineq,
            "gradient_valid": finite,
        }
        return (raw_y, lambda_raw, mu_raw, output), scalar_metrics

    def loss_function(parameters, batch):
        _outputs, metrics = evaluate_with_outputs(parameters, batch)
        return metrics["loss"], {
            key: value for key, value in metrics.items() if key != "loss"
        }

    train_step = jax.jit(make_train_step(loss_function, optimizer))
    evaluation_step = jax.jit(evaluate_with_outputs)
    return train_step, evaluation_step


def _evaluate_projector(self, parameters, evaluation_step, X, Y, batch_size, dtype):
    """Detailed projection evaluation used for readiness and final artifacts.

    Per-epoch history uses the scalar-only evaluation path instead. This
    detailed routine intentionally materializes full outputs, but performs one
    combined device transfer per batch rather than many synchronizing transfers.
    """
    raw_rows, projected_rows = [], []
    lambda_raw_rows, lambda_final_rows = [], []
    mu_raw_rows, mu_final_rows = [], []
    converged_rows, eligible_rows, residual_rows, iteration_rows = [], [], [], []
    for xb, yb, mask, count in _fixed_batches(X, Y, batch_size, shuffle=False, seed=0):
        y_payload = (
            jnp.zeros((len(xb), 0), dtype=dtype)
            if yb is None
            else jnp.asarray(yb, dtype=dtype)
        )
        (raw, lambda_raw, mu_raw, output), _scalar_metrics = evaluation_step(
            parameters, (jnp.asarray(xb, dtype=dtype), y_payload, jnp.asarray(mask))
        )
        host = jax.device_get(
            (
                raw,
                output.y_tilde,
                lambda_raw,
                output.multipliers_eq,
                mu_raw,
                output.multipliers_ineq,
                output.forward_status_codes,
                output.forward_gradient_eligible,
                output.forward_residual_inf_norm,
                output.forward_iterations,
            )
        )
        (
            raw_host,
            projected_host,
            lambda_raw_host,
            lambda_final_host,
            mu_raw_host,
            mu_final_host,
            status_host,
            eligible_host,
            residual_host,
            iteration_host,
        ) = (np.asarray(value)[:count] for value in host)

        raw_rows.append(raw_host)
        projected_rows.append(projected_host)
        lambda_raw_rows.append(lambda_raw_host)
        lambda_final_rows.append(lambda_final_host)
        mu_raw_rows.append(mu_raw_host)
        mu_final_rows.append(mu_final_host)
        converged_rows.append(status_host == 0)
        eligible_rows.append(eligible_host)
        residual_rows.append(residual_host)
        iteration_rows.append(iteration_host)

    raw = np.vstack(raw_rows)
    projected = np.vstack(projected_rows)
    lambda_raw = np.vstack(lambda_raw_rows)
    lambda_final = np.vstack(lambda_final_rows)
    mu_raw = np.vstack(mu_raw_rows)
    mu_final = np.vstack(mu_final_rows)
    converged = np.concatenate(converged_rows)
    eligible = np.concatenate(eligible_rows)
    residuals = np.concatenate(residual_rows)
    iterations = np.concatenate(iteration_rows)
    metrics = self._output_metrics(X, Y, projected, dtype)
    if np.any(eligible):
        eligible_Y = None if Y is None else Y[eligible]
        eligible_task_loss = self._task_value(
            X[eligible], eligible_Y, projected[eligible], dtype
        )
        primal_consistency_mse = float(
            np.mean(np.square(raw[eligible] - projected[eligible]))
        )
    else:
        eligible_task_loss = 0.0
        primal_consistency_mse = 0.0
    metrics.update(
        {
            "raw_task_loss": self._task_value(X, Y, raw, dtype),
            "eligible_projected_task_loss": eligible_task_loss,
            "primal_consistency_mse": primal_consistency_mse,
            "equality_dual_consistency_mse": (
                0.0
                if lambda_raw.shape[1] == 0
                else float(
                    np.mean(np.square(lambda_raw[eligible] - lambda_final[eligible]))
                )
                if np.any(eligible)
                else 0.0
            ),
            "inequality_dual_consistency_mse": (
                0.0
                if mu_raw.shape[1] == 0
                else float(np.mean(np.square(mu_raw[eligible] - mu_final[eligible])))
                if np.any(eligible)
                else 0.0
            ),
            "converged_fraction": float(np.mean(converged)),
            "gradient_eligible_fraction": float(np.mean(eligible)),
            "maximum_solver_residual": float(np.max(residuals)),
            "median_solver_residual": float(np.median(residuals)),
            "mean_iterations": float(np.mean(iterations)),
        }
    )
    metrics["total_loss"] = (
        metrics["eligible_projected_task_loss"]
        + self.loss.primal_consistency_weight * metrics["primal_consistency_mse"]
        + self.loss.equality_dual_consistency_weight
        * metrics["equality_dual_consistency_mse"]
        + self.loss.inequality_dual_consistency_weight
        * metrics["inequality_dual_consistency_mse"]
    )
    return raw, projected, metrics


def _prepare_training(self, train, method):
    """Prepare KKTHN data, backbone, loss and projection resources for the engine."""
    self._validate_composition(allow_incomplete=False)
    if self.objective is None:
        self.objective = 0.0
    values = self._train_config if train is None else train
    resume = None
    if values.get("resume_checkpoint"):
        from kkthn.api.resume import checkpoint_record

        resume = checkpoint_record(values["resume_checkpoint"])
        values = {**resume["config"], **values}
    config = self._normalize_train_config(values)
    X, Y = self._load_dataset()
    Xtr, Ytr, Xva, Yva = self._split(X, Y, config)

    batch_size = int(config["batch_size"])
    if bool(config["drop_last_batch"]) and len(Xtr) < batch_size:
        raise ValueError(
            "drop_last_batch=True would remove the complete training set: "
            f"train samples={len(Xtr)}, batch_size={batch_size}."
        )

    backbone, parameters, optimizer, mean, scale, dtype = self._network_setup(
        config, Xtr
    )
    self._print_training_header(
        method=method, config=config, X_train=Xtr, X_validation=Xva
    )

    if resume is not None:
        from kkthn.api.resume import restore_prepared

        return restore_prepared(self, locals(), resume)

    if bool(config["pretrain"]):
        parameters, opt_state, history, pretrain_epochs = self._pretrain(
            backbone=backbone,
            parameters=parameters,
            optimizer=optimizer,
            X_train=Xtr,
            Y_train=Ytr,
            X_validation=Xva,
            Y_validation=Yva,
            mean=mean,
            scale=scale,
            dtype=dtype,
            config=config,
        )
    else:
        # Do not construct or call the raw pretraining JIT graphs. Projected
        # training starts from the initialized primal/dual network at epoch 1.
        opt_state = optimizer.init(parameters)
        history = []
        pretrain_epochs = 0

    return locals()


def _train_one_method(self, *, method, projection_config, prepared):
    """Supply KKTHN's training phase policy and reports to the shared OHAGI engine."""
    (
        active_warm_start_mode,
        config,
        current_learning_rate,
        evaluation_step,
        history,
        metric_keys,
        objective_schedule,
        optimizer,
        optimizer_state,
        parameters,
        projection_manifest,
        projector,
        resume,
        train_step,
    ) = _prepare_method_training(self, method, prepared, projection_config)
    if resume is not None:
        # A stopped run uses detailed final validation. Recreate the scalar
        # boundary report used by an uninterrupted run before deciding the
        # following epoch's learning rate (including threshold edge cases).
        boundary = _lightweight_history_metrics(
            _evaluate_scalar_batches(
                parameters,
                evaluation_step,
                prepared["Xva"],
                prepared["Yva"],
                int(config["batch_size"]),
                prepared["dtype"],
                metric_keys,
            )
        )
        history[-1] = {
            **history[-1],
            **{f"validation_{k}": v for k, v in boundary.items()},
        }
        next_rate = float(history[-1]["learning_rate"])
        for threshold, rate in objective_schedule:
            if (
                np.isfinite(boundary["data_loss"])
                and boundary["data_loss"] <= threshold
            ):
                next_rate = min(next_rate, float(rate))
        if next_rate != current_learning_rate:
            current_learning_rate = next_rate
            optimizer = Adam(learning_rate=current_learning_rate)
            train_step, evaluation_step = self._projection_steps(
                prepared["backbone"],
                optimizer,
                projector,
                prepared["mean"],
                prepared["scale"],
                prepared["dtype"],
            )
        print(
            f"Resuming after epoch {resume['completed_epochs']}; target epoch {config['epochs']}."
        )

    readiness = None
    minimum = float(config["minimum_projection_convergence"])

    # Projection-readiness warm-up exists only as part of the optional raw
    # pretraining workflow. With pretrain=False, skip both the readiness
    # evaluation and every raw-training graph so projected training begins
    # immediately.
    if bool(config["pretrain"]) and resume is None:
        _raw, _projected, readiness = self._evaluate_projector(
            parameters,
            evaluation_step,
            prepared["Xtr"],
            prepared["Ytr"],
            int(config["batch_size"]),
            prepared["dtype"],
        )
        if readiness["converged_fraction"] < minimum:
            raw_step, _raw_eval_step = self._raw_steps(
                prepared["backbone"],
                optimizer,
                prepared["mean"],
                prepared["scale"],
                prepared["dtype"],
                float(config["dual_pretrain_gamma"]),
            )

            def readiness_epoch(extra, state):
                nonlocal readiness
                parameters, optimizer_state = state
                result = yield epoch_plan(
                    parameters,
                    optimizer_state,
                    _fixed_batches(
                        prepared["Xtr"],
                        prepared["Ytr"],
                        int(config["batch_size"]),
                        shuffle=True,
                        seed=int(config["seed"]) + 50_000 + extra,
                        drop_last=bool(config["drop_last_batch"]),
                    ),
                    raw_step,
                    prepared["dtype"],
                )
                parameters, optimizer_state = result.state
                if (
                    extra == 1
                    or extra % max(1, int(config["readiness_check_every"])) == 0
                ):
                    _raw, _projected, readiness = self._evaluate_projector(
                        parameters,
                        evaluation_step,
                        prepared["Xtr"],
                        prepared["Ytr"],
                        int(config["batch_size"]),
                        prepared["dtype"],
                    )
                    if readiness["converged_fraction"] >= minimum:
                        return EpochTransition((parameters, optimizer_state), True)
                return EpochTransition((parameters, optimizer_state))

            run = Engine.run(
                (parameters, optimizer_state),
                int(config["readiness_extra_epochs"]),
                readiness_epoch,
            )
            parameters, optimizer_state = run.state

        if readiness["converged_fraction"] < minimum:
            warnings.warn(
                f"{method} with predicted dual initialization "
                f"converged for only "
                f"{readiness['converged_fraction']:.1%} of raw "
                "predictions. Invalid projection gradients are zeroed.",
                RuntimeWarning,
                stacklevel=3,
            )

    projected_epochs = max(0, int(config["epochs"]) - int(prepared["pretrain_epochs"]))
    projection_start_epoch = (
        prepared["pretrain_epochs"] + 1 if projected_epochs else None
    )
    if projected_epochs:
        if bool(config["pretrain"]):
            print(
                f"Switching to projection phase at epoch "
                f"{projection_start_epoch} using "
                f"{ProjectionMethod.normalize(method).value}."
            )
            # print(
            #     "Removing the dual pretraining box and using predicted dual "
            #     "initialization from the first projected epoch."
            # )
            pass
        else:
            # print(
            #     "Raw pretraining disabled; starting projected training at "
            #     f"epoch 1 using {ProjectionMethod.normalize(method).value} "
            #     "with predicted dual initialization."
            # )
            pass
    start = time.perf_counter()
    final_validation = None

    completed_projected = 0 if resume is None else int(resume["projected_epochs"])
    next_learning_rate = current_learning_rate

    def projected_epoch(local_epoch, state):
        nonlocal \
            train_step, \
            evaluation_step, \
            current_learning_rate, \
            next_learning_rate, \
            final_validation, \
            optimizer
        parameters, optimizer_state = state
        result = yield epoch_plan(
            parameters,
            optimizer_state,
            _fixed_batches(
                prepared["Xtr"],
                prepared["Ytr"],
                int(config["batch_size"]),
                shuffle=True,
                seed=int(config["seed"]) + 10_000 + local_epoch,
                drop_last=bool(config["drop_last_batch"]),
            ),
            train_step,
            prepared["dtype"],
        )
        parameters, optimizer_state = result.state
        batch_metrics, batch_counts = result.metrics, result.counts

        epoch = prepared["pretrain_epochs"] + local_epoch
        tr = _aggregate_epoch_metrics(batch_metrics, batch_counts, metric_keys)
        tr_view = _lightweight_history_metrics(tr)

        # The final epoch reuses the detailed validation pass needed for saved
        # artifacts. Earlier epochs use the scalar-only evaluator.
        if local_epoch == projected_epochs:
            raw, projected, detailed_metrics = self._evaluate_projector(
                parameters,
                evaluation_step,
                prepared["Xva"],
                prepared["Yva"],
                int(config["batch_size"]),
                prepared["dtype"],
            )
            final_validation = (raw, projected, detailed_metrics)
            va_view = {
                "loss": float(detailed_metrics["loss"]),
                # Data/task component used in the composite total loss.
                "data_loss": float(detailed_metrics["eligible_projected_task_loss"]),
                "total_loss": float(detailed_metrics["total_loss"]),
                "eq": float(detailed_metrics["eq"]),
                "ineq": float(detailed_metrics["ineq"]),
            }
        else:
            va = _evaluate_scalar_batches(
                parameters,
                evaluation_step,
                prepared["Xva"],
                prepared["Yva"],
                int(config["batch_size"]),
                prepared["dtype"],
                metric_keys,
            )
            va_view = _lightweight_history_metrics(va)

        row = {
            "stage": "projected",
            "epoch": epoch,
            "dual_warm_start_mode": active_warm_start_mode,
            "training_loss": tr_view["loss"],
            "training_data_loss": tr_view["data_loss"],
            "training_total_loss": tr_view["total_loss"],
            "training_eq": tr_view["eq"],
            "training_ineq": tr_view["ineq"],
            "validation_loss": va_view["loss"],
            "validation_data_loss": va_view["data_loss"],
            "validation_total_loss": va_view["total_loss"],
            "validation_eq": va_view["eq"],
            "validation_ineq": va_view["ineq"],
            "learning_rate": current_learning_rate,
        }
        # Monitor the physical validation objective, excluding consistency losses.
        # Reductions are permanent and take effect in the following epoch.
        next_learning_rate = current_learning_rate
        for threshold, rate in objective_schedule:
            if np.isfinite(va_view["data_loss"]) and va_view["data_loss"] <= threshold:
                next_learning_rate = min(next_learning_rate, float(rate))
        if (
            next_learning_rate < current_learning_rate
            and local_epoch < projected_epochs
        ):
            print(
                f"Validation objective {va_view['data_loss']:.6g}: learning rate "
                f"{current_learning_rate:g} -> {next_learning_rate:g} for epoch {epoch + 1}."
            )
            current_learning_rate = next_learning_rate
            optimizer = Adam(learning_rate=current_learning_rate)
            train_step, evaluation_step = self._projection_steps(
                prepared["backbone"],
                optimizer,
                projector,
                prepared["mean"],
                prepared["scale"],
                prepared["dtype"],
            )
        for key in (
            "primal_consistency_loss",
            "equality_dual_consistency_loss",
            "inequality_dual_consistency_loss",
        ):
            if key in tr_view:
                row[f"training_{key}"] = tr_view[key]
            if key in va_view:
                row[f"validation_{key}"] = va_view[key]
        history.append(row)

        should_print = (
            local_epoch == 1
            or local_epoch == projected_epochs
            or epoch % max(1, int(config["print_every"])) == 0
        )
        if should_print:
            self._print_training_row(
                epoch,
                int(config["epochs"]),
                {
                    "loss": tr_view["total_loss"],
                    "data_loss": tr_view["data_loss"],
                    "eq": tr_view["eq"],
                    "ineq": tr_view["ineq"],
                },
                {
                    "loss": va_view["total_loss"],
                    "data_loss": va_view["data_loss"],
                    "eq": va_view["eq"],
                    "ineq": va_view["ineq"],
                },
            )

        return EpochTransition((parameters, optimizer_state))

    run = Engine.run(
        (parameters, optimizer_state),
        projected_epochs,
        projected_epoch,
        start_epoch=completed_projected + 1,
    )
    parameters, optimizer_state = run.state
    metrics, projected, raw, run_identity = _finish_method_validation(
        self, config, evaluation_step, final_validation, parameters, prepared
    )

    result = _save_method_training(
        self,
        config,
        current_learning_rate,
        history,
        method,
        metrics,
        next_learning_rate,
        optimizer_state,
        parameters,
        prepared,
        projected,
        projected_epochs,
        projection_manifest,
        projection_start_epoch,
        projector,
        raw,
        run_identity,
        start,
    )
    return result


def model(self, *, train=None, projection=None):
    """Train the selected projected neural model and save its artifacts.

    Parameters
    ----------
    train : mapping, optional
        Overrides for this training run.
    projection : mapping or dataclass, optional
        Overrides for the selected projection method and its solver settings.

    Returns
    -------
    dict
        Training results, metrics, history, and the saved ``metadata_path``.

    Notes
    -----
    Saving includes a CPU StableHLO program when export dependencies are
    available. Loading and inference use JAX. Failure to export does not
    discard the trained model.
    """
    projection_values = {
        **self.projection_config,
        **self._normalize_mapping(projection),
    }
    method = projection_values.pop(
        "method", projection_values.pop("projection_method", self.projection_method)
    )
    prepared = self._prepare_training(train, method)
    return self._train_one_method(
        method=method, projection_config=projection_values, prepared=prepared
    )


def compare_projections(
    self, *, methods: Iterable[str], train=None, projection_configs=None
):
    """Train separate models with a shared data preparation for each method.

    Parameters
    ----------
    methods : iterable of str
        Projection methods to compare.
    train : mapping, optional
        Training overrides shared by the runs.
    projection_configs : mapping, optional
        Per-method projection configurations, indexed by method name.

    Returns
    -------
    dict
        Method names mapped to their individual training results.
    """
    methods = tuple(methods)
    if not methods:
        raise ValueError("methods must not be empty.")
    prepared = self._prepare_training(train, methods[0])
    configs = dict(projection_configs or {})
    return {
        method: self._train_one_method(
            method=method, projection_config=configs.get(method, {}), prepared=prepared
        )
        for method in methods
    }


def _finish_method_validation(
    self, config, evaluation_step, final_validation, parameters, prepared
):
    """Complete final validation and retain the original prediction/metric blocks."""
    if final_validation is None:
        final_validation = self._evaluate_projector(
            parameters,
            evaluation_step,
            prepared["Xva"],
            prepared["Yva"],
            int(config["batch_size"]),
            prepared["dtype"],
        )
    raw, projected, metrics = final_validation
    from kkthn.api.resume import run_identity

    return metrics, projected, raw, run_identity


def _save_method_training(
    self,
    config,
    current_learning_rate,
    history,
    method,
    metrics,
    next_learning_rate,
    optimizer_state,
    parameters,
    prepared,
    projected,
    projected_epochs,
    projection_manifest,
    projection_start_epoch,
    projector,
    raw,
    run_identity,
    start,
):
    """Assemble restart state and save the completed frontend training result."""
    resume_state = {
        "version": 1,
        "completed_epochs": int(config["epochs"]),
        "pretrain_epochs": int(prepared["pretrain_epochs"]),
        "projected_epochs": projected_epochs,
        "next_learning_rate": (
            next_learning_rate if projected_epochs else current_learning_rate
        ),
        "history": history,
        "config": {k: v for k, v in config.items() if k != "resume_checkpoint"},
        "identity": run_identity(
            self, config, [prepared[k] for k in ("Xtr", "Ytr", "Xva", "Yva")]
        ),
        "method": ProjectionMethod.normalize(method).value,
        "projection": projection_manifest,
    }
    result = {
        "model_name": self.name,
        "model_type": self.model_type.value,
        "method": ProjectionMethod.normalize(method).value,
        "parameters": parameters,
        "projector": projector,
        "optimizer_state": optimizer_state,
        "resume_state": resume_state,
        "backbone": prepared["backbone"],
        "architecture": self.nn,
        "loss_config": self.loss,
        "history": history,
        "metrics": metrics,
        "validation_raw": raw,
        "validation_projected": projected,
        "training_seconds": time.perf_counter() - start,
        "input_mean": prepared["mean"],
        "input_scale": prepared["scale"],
        "projection_config": projection_manifest,
        "backend": str(config["backend"]),
        "batch_size": int(config["batch_size"]),
        "dtype_name": np.dtype(prepared["dtype"]).name,
        "dual_warm_start_schedule": (
            "boxed-pretrain-predicted-projection"
            if bool(config["pretrain"])
            else "predicted-projection-only"
        ),
        "dual_warm_start_switched_epoch": projection_start_epoch,
        "pretrain": bool(config["pretrain"]),
        "drop_last_batch": bool(config["drop_last_batch"]),
        "dual_pretrain_gamma": float(config["dual_pretrain_gamma"]),
    }
    state = _TrainedState(
        method=result["method"],
        backbone=prepared["backbone"],
        architecture=self.nn,
        parameters=parameters,
        projector=projector,
        input_mean=np.asarray(prepared["mean"]),
        input_scale=np.asarray(prepared["scale"]),
        dtype=prepared["dtype"],
        batch_size=int(config["batch_size"]),
        backend=str(config["backend"]),
        projection_config=projection_manifest,
        result=result,
    )
    self._save_training_result(result, config, state=state)
    state.metadata_path = result["metadata_path"]
    self._trained[result["method"]] = state
    self._last_method = result["method"]
    return result


def _prepare_method_training(self, method, prepared, projection_config):
    """Prepare the projection, optimizer and initial method-training resources."""
    config = prepared["config"]
    resume = prepared.get("resume")
    definition = self._projection_definition(method, projection_config)
    projection_manifest = (
        asdict(definition) if is_dataclass(definition) else dict(projection_config)
    )
    if resume is not None and (
        resume["method"] != ProjectionMethod.normalize(method).value
        or resume["projection"] != _json_safe(projection_manifest)
    ):
        raise ValueError(
            "Resume requires the checkpoint's projection method and solver settings."
        )
    projector = prepare_projection(
        self,
        method=method,
        config=definition,
        batch_size=int(config["batch_size"]),
        dtype=prepared["dtype"],
        backend=str(config["backend"]),
    )
    current_learning_rate = float(config["learning_rate"])
    if resume is not None:
        current_learning_rate = float(resume["next_learning_rate"])
    objective_schedule = config.get("objective_learning_rate_schedule", [])
    for threshold, rate in objective_schedule:
        if not np.isfinite(threshold) or not np.isfinite(rate) or rate <= 0:
            raise ValueError(
                "Objective schedule requires finite thresholds and positive rates."
            )
    optimizer = Adam(learning_rate=current_learning_rate)
    train_step, evaluation_step = self._projection_steps(
        prepared["backbone"],
        optimizer,
        projector,
        prepared["mean"],
        prepared["scale"],
        prepared["dtype"],
    )
    active_warm_start_mode = "predicted"
    parameters = _copy_tree(prepared["parameters"])
    optimizer_state = _copy_tree(prepared["opt_state"])
    history = list(prepared["history"])
    metric_keys = (
        "loss",
        "task_loss",
        "primal_consistency_loss",
        "equality_dual_consistency_loss",
        "inequality_dual_consistency_loss",
        "eq",
        "ineq",
    )
    return (
        active_warm_start_mode,
        config,
        current_learning_rate,
        evaluation_step,
        history,
        metric_keys,
        objective_schedule,
        optimizer,
        optimizer_state,
        parameters,
        projection_manifest,
        projector,
        resume,
        train_step,
    )
