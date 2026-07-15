# Configuration file for the Sphinx documentation builder.
#
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from blueqat._version import __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = 'blueqat'
copyright = '2018-2026, The Blueqat Developers'
author = 'The Blueqat Developers'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- Options for HTML output --------------------------------------------------

html_theme = 'furo'
html_title = f'blueqat {__version__}'
html_static_path = ['_static']
html_theme_options = {
    'source_repository': 'https://github.com/blueqat/blueqatSDK/',
    'source_branch': 'main',
    'source_directory': 'doc/source/',
}
