# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import date

project = "ngff-zarr"
copyright = f"{date.today().year}, NumFOCUS"
author = "Matt McCormick"

extensions = [
    "sphinx.ext.autosummary",
    "autodoc2",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence", "fieldlist"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc2_packages = [
    {
        "path": "../py/ngff_zarr",
        "exclude_files": ["__about__.py"],
    },
]
autodoc2_render_plugin = "myst"

maximum_signature_line_length = 80

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "itkwasm": ("https://itkwasm.readthedocs.io/en/latest/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_title = f"{project}"

# Furo options
html_theme_options = {
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/thewtex/ngff-zarr",
    "source_branch": "main",
    "source_directory": "docs",
}
