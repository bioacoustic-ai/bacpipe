"""
Run the application straight from the repository, without building it:

    python app/run_app.py

This is the quickest way to work on the application: it uses the bacpipe
code of the checkout and the Python environment you start it with (the same
one you use for ``python run_pipeline.py``).

All options of the built program are available, e.g.

    python app/run_app.py --no-browser --port 5010
    python app/run_app.py --selftest
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# make `bacpipe_app` and the `bacpipe` of this checkout importable
for entry in (str(REPO_ROOT), str(APP_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from bacpipe_app.server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
