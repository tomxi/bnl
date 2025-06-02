# Configuration file for the Sphinx documentation builder.
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# -- Project information -----------------------------------------------------
project = 'BNL'
copyright = '2025, Tom Xi'
author = 'Tom Xi'

# The full version, including alpha/beta/rc tags
from bnl import __version__
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx_copybutton',
    'myst_parser',
]


# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Numpydoc settings
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'groupwise'
autodoc_typehints = 'description'

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Add custom CSS
def setup(app):
    app.add_css_file('custom.css')
