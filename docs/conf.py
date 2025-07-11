# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bnl import __version__

# -- Project information -----------------------------------------------------
project = "bnl"
copyright = "2025, Tom Xi"
author = "Tom Xi"

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
    "sphinx_autodoc_typehints",  # Add type hints support
]


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autoclass_content = "class"

# Enable autosummary to generate stub pages for each documented item.
autosummary_generate = True
autosummary_ignore_module_all = False

# Configure autoclasstoc to show attributes and methods in desired order
autoclasstoc_sections = [
    "public-attrs",  # Shows properties and dataclass fields together
    "public-methods-without-dunders",  # Shows methods without dunder clutter
]

# Configure sphinx-autodoc-typehints for type annotation display
typehints_fully_qualified = False  # Use short type names (e.g., 'List' not 'typing.List')
typehints_document_rtype = True  # Show return type annotations
always_document_param_types = True  # Show type info even for undocumented parameters
typehints_use_signature = True  # Show type hints in function signatures
typehints_use_signature_return = True  # Show return type hints in signatures

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
