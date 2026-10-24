"""Epoch-boundary restart state for the high-level KKTHN training workflow."""

import hashlib
import json
from pathlib import Path
import numpy as np
from kkthn.serialization.problem import symbolic_problem_spec
from kkthn.api._utils import _json_safe


def checkpoint_record(path):
    """Collect the training resources required for a resumable checkpoint."""
    record = json.loads((Path(path) / "hax.json").read_text(encoding="utf-8"))
    metadata = record.get("metadata", {})
    resume = metadata.get("resume")
    if (
        record.get("format") != "hax_orbax_v1"
        or metadata.get("framework") != "kkthn"
        or not resume
    ):
        raise ValueError("This is not a resumable KKTHN Orbax checkpoint.")
    if resume.get("version") != 1 or resume.get("projected_epochs", 0) < 1:
        raise ValueError(
            "Resume requires a checkpoint after at least one projected epoch."
        )
    return resume


def run_identity(model, config, arrays):
    """Bind a restart to the same problem, loss, data and shuffle settings."""
    problem = symbolic_problem_spec(model)
    problem.pop("name", None)
    settings = {
        k: v
        for k, v in config.items()
        if k not in {"epochs", "output_dir", "print_every", "resume_checkpoint"}
    }
    description = _json_safe(
        {
            "problem": problem,
            "architecture": model.nn.to_dict(),
            "loss": model.loss.to_dict(),
            "type": model.model_type.value,
            "settings": settings,
        }
    )
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode())
    for value in arrays:
        if value is None:
            digest.update(b"none")
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def restore_prepared(model, prepared, record):
    """Restore saved training state into a prepared KKTHN model."""
    from ohagi.io.orbax import load_orbax

    config = prepared["config"]
    identity = run_identity(
        model, config, [prepared[k] for k in ("Xtr", "Ytr", "Xva", "Yva")]
    )
    if record["identity"] != identity:
        raise ValueError(
            "Resume checkpoint does not match the problem, network, loss, dataset or training settings."
        )
    if int(config["epochs"]) <= record["completed_epochs"]:
        raise ValueError("epochs must exceed the checkpoint's completed epoch count.")
    template = {
        "parameters": prepared["parameters"],
        "optimizer_state": prepared["optimizer"].init(prepared["parameters"]),
        "input_mean": prepared["mean"],
        "input_scale": prepared["scale"],
    }
    tree, _ = load_orbax(config["resume_checkpoint"], template)
    prepared.update(
        parameters=tree["parameters"],
        opt_state=tree["optimizer_state"],
        mean=tree["input_mean"],
        scale=tree["input_scale"],
        history=record["history"],
        pretrain_epochs=record["pretrain_epochs"],
        resume=record,
    )
    return prepared


def resume_training(self, checkpoint, *, epochs, train=None, projection=None):
    """Continue to a cumulative epoch target with saved settings and shuffle order.

    Attach the same dataset and problem first. The checkpoint must follow at
    least one projected epoch; stopping inside raw/readiness warm-up is not
    supported by this epoch-boundary interface.
    """
    record = checkpoint_record(checkpoint)
    settings = {
        **record["config"],
        **dict(train or {}),
        "epochs": int(epochs),
        "resume_checkpoint": str(Path(checkpoint).expanduser().resolve()),
    }
    saved_projection = {**record["projection"], "method": record["method"]}
    return self.model(
        train=settings,
        projection=saved_projection if projection is None else projection,
    )
