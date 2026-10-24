"""Dataset loading, validation, splitting, and fixed-size batching."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Sequence
import numpy as np
from ohagi.model_types import ModelType


def _read_named_csv(path: str | Path, names: Sequence[str]) -> np.ndarray:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {target}")
        missing = [name for name in names if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV file {target} is missing required columns {missing}; "
                f"available columns are {reader.fieldnames}."
            )
        rows = [[float(row[name]) for name in names] for row in reader]
    if not rows:
        raise ValueError(f"CSV file contains no data rows: {target}")
    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"CSV file contains NaN or infinite values: {target}")
    return values


def _fixed_batches(
    X, Y, batch_size: int, *, shuffle: bool, seed: int, drop_last: bool = False
):
    """Yield fixed-shape batches.

    When ``drop_last`` is false, the final incomplete batch is padded by
    repeating its final physical row and a Boolean mask identifies the valid
    samples. When ``drop_last`` is true, the incomplete batch is omitted.

    Shuffling occurs before dropping, so the omitted samples can change from
    epoch to epoch instead of permanently excluding the same rows.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive.")

    indices = np.arange(len(X))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)

    if not len(X):
        return
    from ohagi.data import fixed_batches

    data = {"inputs": X}
    if Y is not None:
        data["targets"] = Y
    for batch in fixed_batches(data, batch_size, indices=indices, drop_last=drop_last):
        yield (
            batch["inputs"],
            batch.get("targets"),
            batch["mask"],
            int(batch["mask"].sum()),
        )


def _target_names(self) -> tuple[str, ...]:
    if self.model_type is ModelType.PARAMETER_ESTIMATION:
        if self.loss.observed_variables is not None:
            unknown = [
                name
                for name in self.loss.observed_variables
                if name not in self.variable_names
            ]
            if unknown:
                raise ValueError(f"Unknown observed variables: {unknown}.")
            return tuple(self.loss.observed_variables)
    return tuple(self.variable_names)


def _observation_indices(self) -> tuple[int, ...] | None:
    if self.model_type is not ModelType.PARAMETER_ESTIMATION:
        return None
    names = self._target_names()
    lookup = {name: index for index, name in enumerate(self.variable_names)}
    return tuple(lookup[name] for name in names)


def _load_dataset(self):
    if self.dataset_spec is None:
        raise ValueError("Call dataset(parameters=..., variables=...) before training.")
    X = _read_named_csv(self.dataset_spec.parameters_path, self.parameter_names)
    requires_targets = self.model_type is not ModelType.OPTIMIZATION
    Y = None
    if self.dataset_spec.variables_path is not None:
        Y = _read_named_csv(self.dataset_spec.variables_path, self._target_names())
        if len(Y) != len(X):
            raise ValueError("Parameter and variable datasets must have equal rows.")
    if requires_targets and Y is None:
        raise ValueError(f"{self.model_type.value} training requires a variables CSV.")
    return X, Y


def _split(self, X, Y, config):
    fraction = float(config["train_frac"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("train_frac must lie strictly between zero and one.")
    indices = np.arange(len(X))
    np.random.default_rng(int(config["seed"])).shuffle(indices)
    split = max(1, min(len(indices) - 1, int(fraction * len(indices))))
    tr, va = indices[:split], indices[split:]
    return X[tr], None if Y is None else Y[tr], X[va], None if Y is None else Y[va]
