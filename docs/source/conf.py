# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

import pydata_sphinx_theme
import sphinx_book_theme
import sphinx_rtd_theme
import sphinx_theme

project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "diff_binom_confint"
docs_root = Path(__file__).resolve().parents[0]

sys.path.insert(0, str(project_root))

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DBCI"
copyright = "2023, WEN Hao"
author = "WEN Hao"

# The full version, including alpha/beta/rc tags
release = Path(src_root / "version.py").read_text().split("=")[1].strip()[1:-1]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # 'sphinx.ext.autosectionlabel',
    "sphinx_multiversion",
    # "numpydoc",
    "sphinxemoji.sphinxemoji",
    "sphinx_emoji_favicon",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest/", None),
}

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "DeepPSP",  # Username
    "github_repo": "DBCI",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

_theme_name = "sphinx_book_theme"  # "pydata_sphinx_theme", "stanford_theme", etc.

if _theme_name == "stanford_theme":
    html_theme = "stanford_theme"
    html_theme_path = [sphinx_theme.get_html_theme_path("stanford-theme")]
    html_theme_options = {
        "collapse_navigation": False,
        "display_version": True,
    }
elif _theme_name == "sphinx_rtd_theme":
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_theme_options = {
        "collapse_navigation": False,
        "display_version": True,
    }
elif _theme_name == "sphinx_book_theme":
    html_theme = "sphinx_book_theme"
    html_theme_path = [sphinx_book_theme.get_html_theme_path()]
    html_theme_options = {
        "repository_url": "https://github.com/DeepPSP/DBCI",
        "use_repository_button": True,
        "use_issues_button": True,
        "use_edit_page_button": True,
        "use_download_button": True,
        "use_fullscreen_button": True,
        "path_to_docs": "docs/source",
        "repository_branch": "master",
    }
elif _theme_name == "pydata_sphinx_theme":
    html_theme = "pydata_sphinx_theme"
    html_theme_path = [pydata_sphinx_theme.get_html_theme_path()]
    html_theme_options = {
        "collapse_navigation": False,
        "display_version": True,
    }
else:
    raise ValueError(f"Unknown theme name: {_theme_name}")

html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

master_doc = "index"

numfig = False

emoji_favicon = ":abaque:"

linkcheck_ignore = [
    r"https://doi.org/*",  # 418 Client Error
]
