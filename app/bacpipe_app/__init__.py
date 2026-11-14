"""
bacpipe_app - the desktop application wrapper around bacpipe.

The package turns ``python run_pipeline.py`` into a three stage browser
application:

1. a form generated from ``bacpipe/config.yaml`` and ``bacpipe/settings.yaml``
2. a terminal view showing the output of the running pipeline
3. the regular bacpipe dashboard

Start it with ``python -m bacpipe_app`` (or ``python app/run_app.py`` from a
checkout). See ``app/README.md`` for the full documentation.

Importing this package is cheap on purpose: it must not import ``bacpipe``
(which pulls in torch and tensorflow), so that the window appears quickly.
"""

APP_NAME = "bacpipe"
REPO_URL = "https://github.com/bioacoustic-ai/bacpipe"
DOCS_URL = "https://bacpipe.readthedocs.io/en/latest/"

#: Panel extensions that have to be loaded before the first page is served.
#: "plotly" is needed by the bacpipe dashboard, "terminal" by the run page.
#: ``app/tests/test_contract.py`` fails if bacpipe starts using an extension
#: that is not listed here.
PANEL_EXTENSIONS = ("plotly", "terminal")
