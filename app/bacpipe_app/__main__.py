"""Entry point: ``python -m bacpipe_app`` (this is what the launcher runs)."""

import sys

from bacpipe_app.server import main

if __name__ == "__main__":
    sys.exit(main())
