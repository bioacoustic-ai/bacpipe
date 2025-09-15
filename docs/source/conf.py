# docs/source/conf.py
import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Ensure the package can be imported
# If your package is in src/, adjust accordingly: sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../'))  # root so `import bacpipe` works

# -- Project info -----------------------------------------------------------
project = "bacpipe"
author = "Vincent S. Kather"
copyright = f"{datetime.now().year}, {author}"
release = "1.1.2"  # keep in sync with your package/version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",         # Google/NumPy style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",        # add links to source
    "sphinx_autodoc_typehints",   # show type hints
    "myst_parser",                # enable Markdown
]

autosummary_generate = True  # generate stub files for autosummary
autosummary_imported_members = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_sidebars = {
    "**": [
        "globaltoc.html",   # <- Always show full ToC
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
html_logo = "_static/bacpipe_logo.png"      # optional logo
html_favicon = "_static/bacpipe_favicon_white.png"  # optional favicon

def run_apidoc(_):
    from sphinx.ext.apidoc import main

    here = os.path.abspath(os.path.dirname(__file__))  # docs/source/
    root = os.path.abspath(os.path.join(here, "../.."))  # repo root
    api_out = os.path.join(here, "api")  # docs/source/api
    src_dir = os.path.join(root, "bacpipe")  # your package

    main(["-o", api_out, src_dir, "--force"])

def setup(app):
    app.connect("builder-inited", run_apidoc)

