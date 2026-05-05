from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kkthn" / "src"

sys.path.insert(0, str(SRC))

project = "KKT-HardNet"
copyright = "2026, KKT-HardNet Authors"
author = "Ashfaq Iftakher, Rahul Golder, Bimol Nath Roy, MM Faruque Hasan"
release = "1.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "KKT-HardNet Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

source_suffix = ".rst"

html_context = {
    "display_github": True,
    "github_user": "SOULS-TAMU",
    "github_repo": "kkt-hardnet",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

latex_elements = {
    "extraclassoptions": "openany,oneside",
}
