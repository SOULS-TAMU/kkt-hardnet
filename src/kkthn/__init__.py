"""Backward-compatible kkthn frontend powered by OHAGI."""

from ohagi.compat import (
    install_aliases as _install_aliases,
)
from ohagi.compat import (
    reexport_module as _reexport_module,
)
from ._aliases import ALIASES as _ALIASES

_install_aliases(_ALIASES)
from ohagi.compat.plugins import register_symbols

register_symbols(
    {
        "ohagi.KKTHardNet": "kkthn:KKTHardNet",
        "ohagi.ProblemBuilder": "kkthn:ProblemBuilder",
        "ohagi.export.Predictor": "kkthn.serialization.predictor:Predictor",
        "ohagi.export.load_predictor": "kkthn.serialization.predictor:load_predictor",
        "ohagi.export.save_predictor": "kkthn.serialization.predictor:save_predictor",
        "ohagi.export.load_problem": "kkthn.serialization.problem:load_problem",
        "ohagi.export.load_sparse_structure": "kkthn.serialization.problem:load_sparse_structure",
        "ohagi.export.load_symbolic_problem": "kkthn.serialization.problem:load_symbolic_problem",
        "ohagi.export.save_problem": "kkthn.serialization.problem:save_problem",
        "ohagi.export.save_sparse_structure": "kkthn.serialization.problem:save_sparse_structure",
        "ohagi.export.symbolic_problem_spec": "kkthn.serialization.problem:symbolic_problem_spec",
        "ohagi.export.projected_mlp_factory": "kkthn.serialization.registry:projected_mlp_factory",
        "ohagi.export.register": "kkthn.serialization.registry:register",
        "ohagi.export.resolve": "kkthn.serialization.registry:resolve",
    }
)
_reexport_module(globals(), "kkthn.api.root")
