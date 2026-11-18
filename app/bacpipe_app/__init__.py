"""
Desktop application for bacpipe: a configuration form generated from
``config.yaml`` and ``settings.yaml``, a live view of the pipeline output and
the bacpipe dashboard, all served in one browser page.

The package must not import ``bacpipe`` at module level, since that loads
torch and tensorflow.
"""

APP_NAME = "bacpipe"
REPO_URL = "https://github.com/bioacoustic-ai/bacpipe"
DOCS_URL = "https://bacpipe.readthedocs.io/en/latest/"

#: Panel extensions loaded before the first page is served: "plotly" for the
#: dashboard, "terminal" for the pipeline output.
PANEL_EXTENSIONS = ("plotly", "terminal")
