# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bnl import __version__

# -- Project information -----------------------------------------------------
project = "bnl"
copyright = "2025, Qingyang (Tom) Xi"
author = "Qingyang (Tom) Xi"

# The full version, including alpha/beta/rc tags

release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "myst_parser",
    "autoclasstoc",
]

# Napoleon settings
napoleon_numpy_docstring = False
napoleon_include_special_with_doc = False  # false because we set autoclass_content = "both".

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autoclass_content = "both"  # both because we set napoleon_include_init_with_doc = False.

# Enable autosummary to use __all__ to generate stub pages
autosummary_generate = True
autosummary_ignore_module_all = False

# Templates
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
# Custom navigation links
html_context = {
    "display_github": True,
}

autoclasstoc_sections = ["public-attrs", "public-methods-without-dunders"]
