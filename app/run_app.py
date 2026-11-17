"""
Run the bacpipe application from a repository checkout.

    python app/run_app.py [--no-browser] [--port PORT] [--selftest]
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

for entry in (str(REPO_ROOT), str(APP_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from bacpipe_app.server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
