"""
Where things live: the bacpipe package, the bundled resources and the folders
the application writes to.

Nothing in here imports ``bacpipe`` itself. The package is only *located*
(``importlib.util.find_spec``), because importing it loads torch and
tensorflow and takes several seconds.
"""

import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

#: Environment variable to move the workspace somewhere else than ~/bacpipe.
WORKSPACE_ENV = "BACPIPE_WORKSPACE"

#: ``audio_dir`` value that makes ``bacpipe.play`` use the bundled test data.
TEST_DATA_AUDIO_DIR = "bacpipe/tests/test_data"


# --------------------------------------------------------------------
# The bacpipe package and the build information
# --------------------------------------------------------------------
@lru_cache(maxsize=None)
def bacpipe_package_dir():
    """Directory of the installed ``bacpipe`` package (without importing it)."""
    spec = importlib.util.find_spec("bacpipe")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(
            "The bacpipe package was not found. Start the app from the "
            "repository with `python app/run_app.py` or install bacpipe."
        )
    return Path(list(spec.submodule_search_locations)[0])


@lru_cache(maxsize=None)
def build_info():
    """
    Information written by ``app/build.py`` into ``build_info.json``.

    When the app runs from a checkout there is no such file, the version is
    then read from ``pyproject.toml`` and ``bundled`` is False.
    """
    info_file = PACKAGE_DIR / "build_info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        info["bundled"] = True
        return info

    version = "dev"
    pyproject = PACKAGE_DIR.parent.parent / "pyproject.toml"
    if pyproject.exists():
        import tomllib

        version = tomllib.loads(pyproject.read_text(encoding="utf-8"))[
            "project"
        ]["version"]
    return {"version": version, "commit": "", "bundled": False}


def is_bundled():
    """True inside a built application, False when run from a checkout."""
    return build_info()["bundled"]


def logo_path():
    """Big logo with the 'bacpipe' writing (sidebar and splash screen)."""
    return bacpipe_package_dir() / "imgs" / "bacpipe_unlabelled.png"


def favicon_path():
    """Small logo without writing (browser tab and program icon)."""
    return bacpipe_package_dir() / "imgs" / "bacpipe_favicon_white.png"


def button_icon_svg():
    """
    The small bacpipe drawing (src/bacpipe.svg) as inline SVG, for the
    "Play the bacpipe" button. '' if the file is not available.

    In a built app ``build.py`` copies the file into ``resources``.
    """
    for candidate in (
        PACKAGE_DIR / "resources" / "bacpipe.svg",
        PACKAGE_DIR.parent.parent / "src" / "bacpipe.svg",
    ):
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8")
            # drop the xml header in front of the <svg> element and let the
            # drawing take the text color of the button
            return text[text.index("<svg"):].replace('fill="#000000"', 'fill="currentColor"')
    return ""


def demo_url():
    """
    Link to the live demo. ``ref`` tells the demo page which application the
    visitor came from: windows_app, mac_app or linux_app.
    """
    system = build_info().get("system")
    if not system:
        system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    ref = {"windows": "windows_app", "macos": "mac_app"}.get(system, "linux_app")
    return f"https://bacpipe.siriusly.me/?ref={ref}"


def readme_root():
    """
    Directory containing ``README.md`` and the images it refers to.

    In a built app ``build.py`` copies both into ``resources/readme``. In a
    checkout the repository root is used directly.
    """
    bundled = PACKAGE_DIR / "resources" / "readme"
    if (bundled / "README.md").exists():
        return bundled
    repo_root = PACKAGE_DIR.parent.parent
    if (repo_root / "README.md").exists():
        return repo_root
    return None


# --------------------------------------------------------------------
# Folders the app writes to. Everything is inside the user's home
# directory, so the app never needs administrator rights.
# --------------------------------------------------------------------
def workspace_dir():
    """
    Folder holding results, model checkpoints and the app state.

    Relative paths in the settings (``bacpipe_results``,
    ``bacpipe_model_checkpoints``) are resolved against this folder, in the
    same way they are resolved against the repository folder when running
    ``python run_pipeline.py``.
    """
    custom = os.environ.get(WORKSPACE_ENV)
    path = Path(custom).expanduser() if custom else Path.home() / "bacpipe"
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_dir():
    """Folder for the remembered form values, logs and run files."""
    path = workspace_dir() / "app_state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def last_values_file():
    """File remembering what the user entered in the form last time."""
    return state_dir() / "last_values.yaml"


def instance_file():
    """File telling a second launch where the running app is listening."""
    return state_dir() / "instance.json"


def app_log_file():
    """Log of the application process itself (not of the pipeline)."""
    return state_dir() / "app.log"


# --------------------------------------------------------------------
# Starting helper processes with the same Python
# --------------------------------------------------------------------
def python_executable():
    """
    The console Python next to the running interpreter.

    On Windows the app runs with ``pythonw.exe`` (no console window). Helper
    processes need ``python.exe``, because only that one has stdout/stderr.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console_exe = exe.with_name("python.exe")
        if console_exe.exists():
            return str(console_exe)
    return str(exe)


def child_command(script, *args):
    """
    Command line to run one of the helper scripts of this package
    (``worker.py``, ``splash.py``, ``dialogs.py``) in a new process.

    The scripts are run by file path and are self-contained, so this works
    the same in a built app and in a checkout.
    """
    flags = ["-u", "-X", "utf8", "-s"]
    if is_bundled():
        # ignore PYTHONPATH/PYTHONHOME of the user's machine, the bundled
        # runtime must only ever see its own packages
        flags.append("-E")
    return [python_executable(), *flags, str(PACKAGE_DIR / script), *args]


def child_env(**extra):
    """
    Environment for helper processes.

    ``BACPIPE_APP_SYS_PATH`` hands the import path of this process to the
    child, so that a checkout that put the repository on ``sys.path`` (see
    ``app/run_app.py``) also works in the child.
    """
    env = os.environ.copy()
    env["BACPIPE_APP_SYS_PATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONNOUSERSITE"] = "1"
    env.update({k: str(v) for k, v in extra.items()})
    return env


def child_popen_kwargs():
    """Keyword arguments that keep helper processes from opening a console."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
