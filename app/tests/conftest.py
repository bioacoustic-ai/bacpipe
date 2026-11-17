"""
Shared setup of the application tests.

    python -m pytest app/tests                        tests of the source code
    python -m pytest app/tests --app-dist app/dist    ... plus the built app

See "Tests" in app/README.md.
"""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent

# make `bacpipe_app`, `build` and the `bacpipe` of this checkout importable
for entry in (str(REPO_ROOT), str(APP_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def pytest_addoption(parser):
    group = parser.getgroup("bacpipe application")
    group.addoption(
        "--app-dist",
        default=None,
        help="output folder of app/build.py; enables the tests of the built application",
    )
    group.addoption(
        "--app-e2e-model",
        default=None,
        help="also run the pipeline inside the built application with this "
        "model (e.g. batdetect2_clip_avg, which needs no download)",
    )


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def bacpipe_dir():
    """The bacpipe package of this checkout."""
    return REPO_ROOT / "bacpipe"


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    """Never touch the real ~/bacpipe folder of the person running the tests."""
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("BACPIPE_WORKSPACE", str(workspace))
    monkeypatch.setenv("BACPIPE_NO_UPDATE_CHECK", "1")
    return workspace
