"""KKT-HardNet: JAX neural models with differentiable sparse projections."""

from kkthn import api
from ohagi import export
from ohagi import losses
from ohagi import nn
from ohagi import solvers as opt
from ohagi import problems
from kkthn.api import training as train
from ohagi import _utils as utils
from kkthn.api import DatasetSpec
from kkthn.api import KKTHardNet
from kkthn.api import ObjectiveGradientResult
from kkthn.api import PredictionWithGradient
from kkthn.api import ProblemBuilder
from ohagi.export import load_predictor
from ohagi.export import save_predictor
from ohagi.losses import Loss
from ohagi.model_types import ModelType
from ohagi.nn import MLP
from ohagi.nn import MLPBackbone
from purin.solvers import ADMMConfig
from purin.solvers import ADMMProjection
from purin.solvers import ChambollePockConfig
from purin.solvers import ChambollePockProjection
from purin.solvers import FischerBurmeisterConfig
from purin.solvers import ImplicitGradientConfig
from purin.solvers import LinearEqualityConfig
from purin.solvers import LinearSolverConfig
from purin.solvers import ProjectionConfig
from purin.solvers import ProjectionMethod
from purin.solvers import ProjectionResult
from purin.solvers import SolveStatus
from purin.solvers import prepare_projection
from purin.solvers.implicit import ImplicitProjectionLayer
from purin.solvers.implicit import make_implicit_projection_layer
from purin.solvers.primal_dual_interior_point.derivatives import (
    CompiledBatchProblem as CompiledBatchProblem,
)
from purin.solvers.primal_dual_interior_point.derivatives import (
    CompiledProblem as CompiledProblem,
)
from purin.solvers.primal_dual_interior_point.solver import (
    PreparedSparseProjectionBatchSolver as PreparedSparseProjectionBatchSolver,
)
from purin.solvers.primal_dual_interior_point.types import (
    BatchProblemSolution as BatchProblemSolution,
)
from purin.solvers.primal_dual_interior_point.types import (
    ProblemSolution as ProblemSolution,
)
from purin.solvers.types import ProjectionLayerOutput
from purin.solvers.types import ProjectionTrainingOutput
from purin.problems import FunctionalProblem
from purin.problems import MatrixProblem
from purin.problems import ParametricLP
from purin.problems import ParametricQP
from purin.problems import Problem
from purin.problems import Projection
from purin.problems import SymbolicProblem
from purin.problems.expressions import Constraint as Constraint
from purin.problems.expressions import ConstraintList as ConstraintList
from purin.problems.expressions import Expression as Expression
from purin.problems.expressions import VectorExpression as VectorExpression
from purin.problems.problem import OptimizationProblem as OptimizationProblem
from purin.problems.problem import PrimalDualProblem as PrimalDualProblem
from purin.problems.problem import Problem as CoreProblem  # noqa: F401 - supported public alias
from purin.problems.sparsity import ProjectionStructure as ProjectionStructure
from purin.problems.types import BatchNamedValues as BatchNamedValues
from purin.problems.types import NamedValues as NamedValues
from ohagi.training import Adam
from ohagi.training import AdamState as AdamState
from ohagi.training import BuildReport as BuildReport
from ohagi.training import CompiledProjectedGraphs as CompiledProjectedGraphs
from ohagi.training import PredictionOutput as PredictionOutput
from ohagi.training import ProjectedMetrics as ProjectedMetrics
from ohagi.training import apply_updates as apply_updates
from ohagi.training import configure_compilation_cache as configure_compilation_cache
from ohagi.training.projected import ReportedProjectedModel


class ProjectedNeuralModel(ReportedProjectedModel):
    """KKTHN frontend for the shared reported projected-model workflow."""

    framework = "kkthn"


__all__ = [
    "api",
    "nn",
    "opt",
    "problems",
    "losses",
    "train",
    "utils",
    "export",
    "Problem",
    "Projection",
    "KKTHardNet",
    "ProblemBuilder",
    "DatasetSpec",
    "PredictionWithGradient",
    "ObjectiveGradientResult",
    "ModelType",
    "MLP",
    "Loss",
    "SymbolicProblem",
    "FunctionalProblem",
    "MatrixProblem",
    "ParametricLP",
    "ParametricQP",
    "ADMMConfig",
    "ADMMProjection",
    "ChambollePockConfig",
    "ChambollePockProjection",
    "ProjectionMethod",
    "SolveStatus",
    "ProjectionResult",
    "ProjectionTrainingOutput",
    "ProjectionConfig",
    "FischerBurmeisterConfig",
    "LinearEqualityConfig",
    "LinearSolverConfig",
    "ImplicitGradientConfig",
    "prepare_projection",
    "Adam",
    "MLPBackbone",
    "ProjectedNeuralModel",
    "ImplicitProjectionLayer",
    "ProjectionLayerOutput",
    "make_implicit_projection_layer",
    "save_predictor",
    "load_predictor",
]

__version__ = "0.8.0"
