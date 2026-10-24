"""Public training conveniences with KKT-HardNet reporting defaults."""

from ohagi.compat import reexport_module

reexport_module(globals(), "ohagi.training")
from kkthn.api.logging import (
    KKTTableLogger as KKTTableLogger,
    print_postprocessing_start as print_postprocessing_start,
    print_postprocessing_finish as print_postprocessing_finish,
    print_research_citation as print_research_citation,
)
