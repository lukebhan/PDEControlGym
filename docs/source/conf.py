# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PDEControlGym'
copyright = '2023, PDEContRoLGym'
author = 'Luke Bhan'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.viewcode", 
        "sphinx.ext.ifconfig", 
        "sphinx.ext.autosummary", 
        "sphinx.ext.autodoc", 
        "sphinx_autodoc_typehints",
        "sphinx.ext.mathjax",
]

autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []
pygments_style = 'sphinx'
source_suffix = ".rst"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def setup(app):
    app.add_css_file("css/pdecg_theme.css")
