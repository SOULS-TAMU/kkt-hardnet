"""Sphinx configuration; release versions come from package metadata."""

import os
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]
project = "KKT-HardNet"
release = metadata["version"]
version = release
copyright = "2026, SOULS-TAMU contributors"
author = "SOULS-TAMU contributors"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples/*.py"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autoclass_content = "class"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
# Companion-site inventories can be enabled once those projects are live.
# A local build must not depend on another project's site being available.
intersphinx_mapping = {}
html_theme = "sphinx_rtd_theme"
html_title = project + " Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_context = {
    "display_github": True,
    "github_user": "SOULS-TAMU",
    "github_repo": "kkt-hardnet",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
if os.environ.get("READTHEDOCS_CANONICAL_URL"):
    html_baseurl = os.environ["READTHEDOCS_CANONICAL_URL"]
html_copy_source = True

# Preserve previously published tutorial/API URLs without reviving retired APIs.
import json

templates_path = ["_templates"]
legacy_redirects = json.loads(
    (ROOT / "docs/redirects.json").read_text(encoding="utf-8")
)
html_context["legacy_redirects"] = legacy_redirects
html_additional_pages = {page: "redirect.html" for page in legacy_redirects}
