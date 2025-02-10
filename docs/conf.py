"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from datetime import datetime
from typing import Any

import pytz

from pollux import __version__

# -- Project information -----------------------------------------------------

author = "Pollux Developers"
project = "pollux"
copyright = f"{datetime.now(pytz.timezone('UTC')).year}, {author}"
version = __version__

master_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    # "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx-prompt",
    "sphinxext.opengraph",
    "sphinx_togglebutton",
    # "sphinx_tippy",
]

python_use_unqualified_type_names = True

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

source_suffix = [".md", ".rst", ".ipynb"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "unxt": ("http://unxt.readthedocs.io/en/latest/", None),
}

# -- Autodoc settings ---------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

always_document_param_types = True
typehints_use_signature = True


nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

# -- MyST Setting -------------------------------------------------

myst_enable_extensions = [
    "amsmath",  # for direct LaTeX math
    "attrs_block",  # enable parsing of block attributes
    "attrs_inline",  # apply syntax highlighting to inline code
    "colon_fence",
    "deflist",
    "dollarmath",  # for $, $$
    # "linkify",  # identify “bare” web URLs and add hyperlinks:
    "smartquotes",  # convert straight quotes to curly quotes
    "substitution",  # substitution definitions
]
myst_heading_anchors = 3

# myst_substitutions = {
#     "ArrayLike": ":obj:`jaxtyping.ArrayLike`",
#     "Any": ":obj:`typing.Any`",
# }


# -- HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "pollux"
html_logo = "_static/Pollux-logo.png"
html_copy_source = True
html_favicon = "_static/favicon.png"

html_static_path = ["_static"]
html_css_files = []

html_theme_options: dict[str, Any] = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/adrn/pollux",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/adrn/pollux",
            "icon": "fa-brands fa-github",
        },
        # {
        #     "name": "PyPI",
        #     "url": "https://pypi.org/project/unxt/",
        #     "icon": "https://img.shields.io/pypi/v/unxt",
        #     "type": "url",
        # },
        # {
        #     "name": "Zenodo",
        #     "url": "https://doi.org/10.5281/zenodo.10850455",
        #     "icon": "fa fa-quote-right",
        # },
    ],
}
