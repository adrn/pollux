"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

from pollux import __version__

# -- Tutorial data -----------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "_data"
_TUTORIAL_DATA = {
    "rgb-highSNR-1k-1chip.h5": "https://users.flatironinstitute.org/~apricewhelan/pollux/rgb-highSNR-1k-1chip.h5",
}

_DATA_DIR.mkdir(exist_ok=True)
for _filename, _url in _TUTORIAL_DATA.items():
    _path = _DATA_DIR / _filename
    if not _path.exists():
        print(f"Downloading tutorial data: {_filename}")
        urllib.request.urlretrieve(_url, _path)

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
    "rtds_action",
]

python_use_unqualified_type_names = True

exclude_patterns = [
    "_build",
    "_data",
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
    "numpyro": ("https://num.pyro.ai/en/stable", None),
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

nb_execution_mode = "off"


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

# -- rtds-action --

if "GITHUB_TOKEN" in os.environ:
    print("GitHub Token found: retrieving artifact")

    # The name of your GitHub repository
    rtds_action_github_repo = "adrn/pollux"

    # The path where the artifact should be extracted
    # Note: this is relative to the CWD when sphinx runs (the repo root on RTD)
    rtds_action_path = "docs/tutorials"

    # The "prefix" used in the `upload-artifact` step of the action
    rtds_action_artifact_prefix = "notebooks-for-"

    # A GitHub personal access token is required, more info below
    rtds_action_github_token = os.environ["GITHUB_TOKEN"]

    # Whether or not to raise an error on ReadTheDocs if the
    # artifact containing the notebooks can't be downloaded (optional)
    rtds_action_error_if_missing = True

else:
    print("No GitHub Token found: skipping artifact retrieval")
    rtds_action_github_repo = ""
    rtds_action_github_token = ""
    rtds_action_path = ""


# -- Check for executed tutorials and only add to toctree if they exist ------

# Note: list the expected .ipynb filename for each tutorial
tutorial_files = [
    "tutorials/Lux-linear-simulated-data.ipynb",
    "tutorials/Lux-getting-started-apogee.ipynb",
    "tutorials/Lux-iterative-optimization.ipynb",
    "tutorials/Lux-simulated-data-underestimated-err.ipynb",
]

_not_executed = []
_tutorial_toctree_items = []
for fn in tutorial_files:
    if not Path(fn).exists() and "GITHUB_TOKEN" not in os.environ:
        _not_executed.append(fn)
        continue
    _tutorial_toctree_items.append(fn)

if _tutorial_toctree_items:
    _items = "\n   ".join(_tutorial_toctree_items)
    _tutorial_toctree = f"""\
.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   {_items}
"""
else:
    _tutorial_toctree = "No tutorials found!\n"

if _not_executed:
    print(
        "\n-------- Pollux warning --------\n"
        "Some tutorial notebooks could not be found! This is likely because "
        "the tutorial notebooks have not been executed. If you are building "
        "the documentation locally, you may want to execute the notebooks "
        "before running the sphinx build.\n"
        f"Missing tutorials: {', '.join(_not_executed)}\n"
    )

with Path("_tutorials.rst").open("w", encoding="utf-8") as f:
    f.write(_tutorial_toctree)
