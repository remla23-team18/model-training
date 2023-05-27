"""Sphinx configuration."""
# pylint: disable-all
project = "Model Training"
author = "remla23-team18"
copyright = "2023, remla23-team18"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
