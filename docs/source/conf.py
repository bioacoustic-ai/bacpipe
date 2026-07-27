# docs/source/conf.py
import os
import shutil
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../"))  # root so `import bacpipe` works

# -- Project info -----------------------------------------------------------
project = "bacpipe"
author = "Vincent S. Kather"
copyright = f"{datetime.now().year}, {author}"
release = "1.3.3"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",  # add links to source
    "sphinx_autodoc_typehints",  # show type hints
    "myst_parser",  # enable Markdown
    "nbsphinx",  # <- ADDED: Render Jupyter Notebooks
]

autosummary_generate = True
autosummary_imported_members = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
templates_path = ["_templates"]

# ADDED: Exclude notebook checkpoints
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# docs/source/conf.py

# -- Syntax Highlighting ----------------------------------------------------
highlight_language = "python"
pygments_style = "sphinx"

# Force nbsphinx to lex code cells as Python if metadata is missing
nbsphinx_codecell_lexer = "python"

# -- nbsphinx Configuration --------------------------------------------------
# "never": Use outputs already present in the notebook (Fastest & safest)
# "auto": Execute notebook only if missing outputs
# "always": Re-execute notebooks every doc build
nbsphinx_execute = "never"

html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "searchbox.html",
    ]
}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/bacpipe_logo.png"
html_favicon = "_static/bacpipe_favicon_white.png"

# ADDED: Keep left sidebar ToC expanded on all pages
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}


def run_apidoc(_):
    from sphinx.ext.apidoc import main

    here = os.path.abspath(os.path.dirname(__file__))  # docs/source/
    root = os.path.abspath(os.path.join(here, "../.."))  # repo root
    api_out = os.path.join(here, "api")  # docs/source/api
    src_dir = os.path.join(root, "bacpipe")  # your package

    main(["-o", api_out, src_dir, "--force"])
# docs/source/conf.py

# ... (Path setup & copy_examples definition) ...

def copy_examples():
    """Copy bacpipe/examples into docs/source/examples immediately."""
    here = os.path.abspath(os.path.dirname(__file__))  # docs/source/
    root = os.path.abspath(os.path.join(here, "../.."))  # repo root
    src_examples = os.path.join(root, "bacpipe", "examples")
    dst_examples = os.path.join(here, "examples")

    if os.path.exists(src_examples):
        if os.path.exists(dst_examples):
            shutil.rmtree(dst_examples)
        shutil.copytree(src_examples, dst_examples)

# Run immediately when conf.py is loaded (BEFORE Sphinx scans files)
copy_examples()

def setup(app):
    app.connect("builder-inited", run_apidoc)
