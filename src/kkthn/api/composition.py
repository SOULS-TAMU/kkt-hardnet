"""Model composition, symbolic-problem attachment, and public configuration methods."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
import numpy as np
from ohagi.losses import Loss
from ohagi.model_types import ModelType
from ohagi.nn import MLP
from purin.problems.expressions import VectorExpression
from purin.problems.problem import Problem as CoreProblem
from purin.problems.projection import Projection
from kkthn.api.state import _TrainedState
from kkthn.api.types import DatasetSpec


if TYPE_CHECKING:
    from kkthn.api.model import KKTHardNet


def __init__(
    self,
    type: ModelType | str = ModelType.SURROGATE,
    nn: MLP | Mapping[str, Any] | None = None,
    loss: Loss | Mapping[str, Any] | None = None,
    projection: Projection | CoreProblem | None = None,
    *,
    name: str | None = None,
    train: Mapping[str, Any] | None = None,
) -> None:
    # Backward-compatible positional name: KKTHardNet("example").

    """Construct a symbolic problem and its projected neural model.

    Args:
        type: Surrogate, optimization or parameter-estimation model family.
        nn: Neural-network architecture and primal/dual output configuration.
        loss: Training loss specification. Defaults follow the selected task.
        projection: Symbolic problem or projection specification containing
            variables, parameters, constraints and solver configuration.
        name: Optional model name used for reporting and artifact directories.
        train: Optional training configuration; architecture belongs in ``nn``.

    Notes:
        The historical positional form ``KKTHardNet("example")`` is retained.
        Building this frontend supplies problem resources to the shared OHAGI
        engine and PURIN projection services.
    """

    if (
        projection is None
        and nn is None
        and loss is None
        and name is None
        and str(type).strip().lower().replace("-", "_")
        not in {item.value for item in ModelType}
    ):
        name = str(type)

        type = ModelType.SURROGATE

    self.model_type = ModelType.normalize(type)

    self.nn = nn if isinstance(nn, MLP) else MLP(**dict(nn or {}))

    if isinstance(loss, Loss):
        self.loss = loss

    elif loss is None:
        default_kind = (
            "objective"
            if self.model_type is ModelType.OPTIMIZATION
            else "observation"
            if self.model_type is ModelType.PARAMETER_ESTIMATION
            else "mse"
        )

        self.loss = Loss(kind=default_kind)

    else:
        self.loss = Loss.from_dict(dict(loss))

    source_name = getattr(projection, "name", None)

    CoreProblem.__init__(self, name=name or source_name or "kkthardnet")

    self._user_objective_defined = False

    if projection is not None:
        self._attach_projection(projection)

    self.projection_method = str(
        getattr(projection, "method", "fischer-burmeister-newton")
    )

    self.projection_config = dict(getattr(projection, "config", {}))

    self.dataset_spec: DatasetSpec | None = None

    self._train_config = self._normalize_train_config(train)

    self._trained: dict[str, _TrainedState] = {}

    self._last_method: str | None = None

    self._metadata_path: str | None = None

    self._metric_canonical = None

    self._validate_composition(allow_incomplete=True)


def _attach_projection(self, projection: CoreProblem) -> None:

    self.add_parameter(projection.parameter_names)

    self.add_variable(projection.variable_names)

    self.constraints.add(*list(projection.constraints))

    if projection.objective is not None:
        self.objective = projection.objective

        self._user_objective_defined = True

    elif self.model_type is not ModelType.OPTIMIZATION:
        # Numerical compilers require an objective, although the projection

        # subproblem itself uses Euclidean distance to the NN output.

        self.objective = 0.0

    if getattr(projection, "_sparsity_override", None) is not None:
        self.set_sparsity_override(*projection._sparsity_override)


def _validate_composition(self, *, allow_incomplete: bool = False) -> None:

    if self.model_type is ModelType.SURROGATE and self.loss.kind != "mse":
        raise ValueError("surrogate type requires Loss.mse(...).")

    if self.model_type is ModelType.PARAMETER_ESTIMATION and self.loss.kind not in {
        "observation",
        "mse",
    }:
        raise ValueError(
            "parameter_estimation requires Loss.parameter_estimation(...) or MSE."
        )

    if self.model_type is ModelType.OPTIMIZATION:
        if self.loss.kind != "objective":
            raise ValueError("optimization type requires Loss.objective(...).")

        if not allow_incomplete and not self._user_objective_defined:
            raise ValueError(
                "An optimization projection must define projection.objective."
            )


def as_projection_problem(self) -> CoreProblem:
    """Expose the KKTHN problem through PURIN's common projection contract."""
    target_names = [f"__target_{index}" for index in range(len(self.variable_names))]

    projected = CoreProblem(name=f"{self.name}_projection")

    projected.add_parameter(target_names + list(self.parameter_names))

    projected.add_variable(list(self.variable_names))

    target = VectorExpression(
        [getattr(projected.parameter, name) for name in target_names],
        "projection_target",
    )

    projected.set_projection_objective(variables=projected.variable, target=target)

    projected.constraints.add(*list(self.constraints))

    if getattr(self, "_sparsity_override", None) is not None:
        projected.set_sparsity_override(*self._sparsity_override)

    return projected


def dataset(
    self, *, parameters: str | Path, variables: str | Path | None = None
) -> "KKTHardNet":
    """Attach CSV dataset paths without reading or changing the source files.

    Args:
        parameters: Input CSV with headers matching declared parameter names.
        variables: Optional target CSV with headers matching variable names.

    Returns:
        This model, allowing chained configuration calls.
    """

    self.dataset_spec = DatasetSpec(
        parameters_path=str(parameters),
        variables_path=None if variables is None else str(variables),
    )

    return self


def set_train_config(self, config: Mapping[str, Any]) -> "KKTHardNet":
    """Set this model's training configuration.

    Args:
        config: Training options; omitted values receive the existing defaults.

    Returns:
        This model, allowing chained configuration calls.
    """

    self._train_config = self._normalize_train_config(config)

    return self


def set_projection_config(self, config: Mapping[str, Any]) -> "KKTHardNet":
    """Set the projection method and solver configuration.

    Args:
        config: Mapping or dataclass of solver options, optionally including
            ``method`` or ``projection_method``.

    Returns:
        This model, allowing chained configuration calls.
    """

    values = self._normalize_mapping(config)

    self.projection_method = str(
        values.pop("method", values.pop("projection_method", self.projection_method))
    )

    self.projection_config = values

    return self


def extract(self, path: str | Path):
    """Load named NPZ arrays and attach them to this problem instance."""
    target = Path(path).expanduser().resolve()

    if not target.exists():
        raise FileNotFoundError(target)

    with np.load(target, allow_pickle=False) as arrays:
        loaded = {name: np.asarray(arrays[name]) for name in arrays.files}

    for name, value in loaded.items():
        setattr(self, name, value)

    return loaded


def optimize(self, *, train=None, projection=None):
    """Build an optimization model with optional training and projection overrides."""
    if self.model_type is not ModelType.OPTIMIZATION:
        raise ValueError("optimize() requires type='optimization'.")

    return self.model(train=train, projection=projection)


def estimate(self, *, train=None, projection=None):
    """Build a parameter-estimation model with optional training and projection overrides."""
    if self.model_type is not ModelType.PARAMETER_ESTIMATION:
        raise ValueError("estimate() requires type='parameter_estimation'.")

    return self.model(train=train, projection=projection)


"""Translate the historical network contract into the shared model contract."""
from ohagi.models import ProjectedModel
from ohagi.nn.backbone import FunctionalBackbone


def projected_model(
    problem, backbone, mean, scale, dtype, projector=None, *, gamma=None
):
    """Adapt KKTHN's network outputs and scaling to the shared projected model."""
    if gamma is None:
        apply = lambda parameters, inputs: problem._network_outputs(
            backbone, parameters, inputs, mean, scale, dtype
        )
    else:
        apply = lambda parameters, inputs: problem._network_primal_output(
            backbone, parameters, inputs, mean, scale, dtype, gamma
        )
    return ProjectedModel(
        FunctionalBackbone(None, apply), projector, return_projection=True
    )
