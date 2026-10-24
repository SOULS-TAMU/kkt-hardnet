"""Historical reporting defaults over OHAGI table rendering."""

from dataclasses import dataclass
from datetime import datetime as datetime
from ohagi.training.logging import (
    print_postprocessing_start as print_postprocessing_start,
)
from ohagi.training.logging import print_research_citation as _print_citation
from ohagi.training.logging import KKTTableLogger as _TableLogger

_CITATION = """@article{iftakher2025physics,
  title={Physics-informed neural networks with hard nonlinear equality and inequality constraints},
  author={Iftakher, Ashfaq and Golder, Rahul and Roy, Bimol Nath and Hasan, MM Faruque},
  journal={Computers \\& Chemical Engineering},
  pages={109418},
  year={2025},
  publisher={Elsevier}
}"""
_CONTACT = "bimolnathroy@tamu.edu, rahulgolder8420@tamu.edu, hasan@tamu.edu"


def print_research_citation():
    """Print KKTHN's research citation through the shared logger."""
    return _print_citation(citation=_CITATION, contact=_CONTACT)


def print_postprocessing_finish(*, include_citation=True):
    """Report completion of KKTHN postprocessing through the shared logger."""
    print("Done.")
    if include_citation:
        print_research_citation()


@dataclass
class KKTTableLogger(_TableLogger):
    """KKTHN progress-table adapter over the shared OHAGI logger."""

    title: str = "KKTHN"
