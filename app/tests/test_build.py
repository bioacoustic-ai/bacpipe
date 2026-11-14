"""
Tests of a built application (the output of app/build.py).

    python app/build.py
    python -m pytest app/tests/test_build.py --app-dist app/dist

Without ``--app-dist`` all tests in this file are skipped. The tests work on
Windows, macOS and Linux; a few checks only apply to one system.

``--app-e2e-model batdetect2_clip_avg`` additionally runs the pipeline on
the example data inside the built application (a few minutes).
"""

import json
import os
import struct
import subprocess
import sys

import pytest

import build

IS_WINDOWS = sys.platform == "win32"


@pytest.fixture(scope="module")
def layout(request):
    dist = request.config.getoption("--app-dist")
    if not dist:
        pytest.skip("needs --app-dist <output folder of app/build.py>")
    layout = build.layout_for(os.path.abspath(dist))
    assert layout.root.is_dir(), f"no built application in {layout.root}"
    return layout


@pytest.fixture(scope="module")
def build_info(layout):
    for info_file in layout.runtime.rglob("bacpipe_app/build_info.json"):
        return json.loads(info_file.read_text(encoding="utf-8"))
    pytest.fail("build_info.json not found in the runtime")


def clean_env(workspace):
    """
    The environment of a machine that has its own, different Python: the
    application must not be influenced by it.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace / "not_a_python_path")
    env["PYTHONHOME"] = str(workspace / "not_a_python_home")
    env["BACPIPE_WORKSPACE"] = str(workspace)
    env["BACPIPE_NO_UPDATE_CHECK"] = "1"
    return env


def run_python(layout, workspace, *arguments, timeout=600):
    """Run the Python of the application the way the launcher does."""
    return subprocess.run(
        [str(layout.python), "-E", "-s", "-X", "utf8", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=clean_env(workspace),
        cwd=str(workspace),
    )


@pytest.fixture
def workspace(isolated_workspace):
    isolated_workspace.mkdir(parents=True, exist_ok=True)
    return isolated_workspace


# --------------------------------------------------------------------
# The folder
# --------------------------------------------------------------------
def test_folder_contains_launcher_runtime_and_readme(layout):
    assert layout.launcher.is_file()
    assert layout.python.is_file()
    assert (layout.info_dir / "README.txt").is_file()
    assert (layout.info_dir / "LICENSE").is_file()
    if not IS_WINDOWS:
        assert os.access(layout.launcher, os.X_OK), "the launcher is not executable"


def test_version_is_the_version_of_pyproject(build_info):
    assert build_info["version"] == build.project_version()


def test_runtime_is_the_pinned_python(layout, workspace):
    result = run_python(layout, workspace, "-c", "import platform; print(platform.python_version())")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == build.load_config()["python"]["version"]


def test_runtime_does_not_see_packages_of_the_machine(layout, workspace):
    """Every import must come from inside the application folder."""
    code = (
        "import sys, json, panel, yaml, bacpipe_app;"
        "print(json.dumps([sys.prefix, panel.__file__, yaml.__file__, bacpipe_app.__file__]))"
    )
    result = run_python(layout, workspace, "-c", code)
    assert result.returncode == 0, result.stderr
    runtime = os.path.realpath(layout.runtime)
    for location in json.loads(result.stdout.strip().splitlines()[-1]):
        assert os.path.realpath(location).startswith(runtime), location


def test_help_page_resources_are_shipped(layout):
    readme = list(layout.runtime.rglob("bacpipe_app/resources/readme/README.md"))
    assert readme, "README.md for the help page is missing"


# --------------------------------------------------------------------
# bacpipe inside the application
# --------------------------------------------------------------------
def test_bacpipe_and_its_dependencies_import(layout, workspace):
    """
    The import of bacpipe loads torch, tensorflow and all model pipelines
    modules that are imported at start. This is the test that notices a
    dependency that is missing from the requirements, or a [prune] pattern
    that removed too much.
    """
    code = (
        "import bacpipe, torch, tensorflow, panel, plotly, librosa, umap, sklearn;"
        "from importlib.metadata import version;"
        "print('bacpipe', version('bacpipe'), len(bacpipe.supported_models), 'models')"
    )
    result = run_python(layout, workspace, "-c", code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"bacpipe {build.project_version()}" in result.stdout


def test_form_matches_the_bacpipe_inside_the_application(layout, workspace):
    """The form is generated from the YAML files that were actually shipped."""
    code = (
        "import json;"
        "from bacpipe_app import schema, paths;"
        "d = schema.load_form_definitions(paths.bacpipe_package_dir());"
        "print(json.dumps({k: len(v) for k, v in d.items()}))"
    )
    result = run_python(layout, workspace, "-c", code)
    assert result.returncode == 0, result.stderr
    counts = json.loads(result.stdout.strip().splitlines()[-1])
    assert counts["config"] > 3 and counts["settings"] > 20


# --------------------------------------------------------------------
# Starting it
# --------------------------------------------------------------------
def test_launcher_starts_the_application(layout, workspace):
    """
    Start the real launcher (bacpipe.exe on Windows). `--selftest` starts
    the web server, requests the form page and the help page and exits; the
    launcher passes the exit code on.
    """
    result = subprocess.run(
        [str(layout.launcher), "--selftest"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        env=clean_env(workspace),
        cwd=str(workspace),  # not the application folder
    )
    log = workspace / "app_state" / "app.log"
    details = result.stdout + result.stderr
    if log.exists():
        details += log.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, details


def test_launcher_reports_a_failing_start(layout, workspace):
    """A broken start must not look like success."""
    result = subprocess.run(
        [str(layout.launcher), "--this-option-does-not-exist"],
        capture_output=True,
        timeout=300,
        env=clean_env(workspace),
        cwd=str(workspace),
    )
    assert result.returncode != 0


def test_application_does_not_write_into_its_own_folder(layout, workspace):
    """
    The folder may be in a read-only location. Everything the application
    writes has to go to the workspace in the home directory.
    """

    def snapshot():
        return {str(f) for f in layout.root.rglob("*")}

    before = snapshot()
    result = run_python(layout, workspace, "-m", "bacpipe_app", "--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    new_files = sorted(snapshot() - before)
    assert not new_files, new_files[:10]
    assert (workspace / "app_state" / "app.log").is_file()


# --------------------------------------------------------------------
# Windows specifics
# --------------------------------------------------------------------
@pytest.mark.skipif(not IS_WINDOWS, reason="Windows launcher")
def test_windows_launcher_needs_no_console_and_no_admin_rights(layout):
    data = layout.launcher.read_bytes()
    assert data[:2] == b"MZ"
    pe_header = struct.unpack("<I", data[0x3C:0x40])[0]
    subsystem = struct.unpack("<H", data[pe_header + 24 + 68 : pe_header + 24 + 70])[0]
    assert subsystem == 2, "bacpipe.exe must be a GUI program (no console window)"
    assert b"asInvoker" in data, "the manifest must request the rights of the user"
    assert b"requireAdministrator" not in data
    # version information, shown in the properties dialog of the file
    assert "ProductName".encode("utf-16-le") in data


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows launcher")
def test_windows_console_launcher(layout):
    script = (layout.root / "bacpipe_debug.cmd").read_bytes()
    assert b"runtime\\python.exe" in script
    assert b"\r\n" in script, ".cmd files need Windows line endings"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows path limit")
def test_paths_are_short_enough_for_windows(layout):
    limit = build.load_config()["windows"]["max_path_length"]
    length, path = build.longest_paths(layout, count=1)[0]
    assert length <= limit, f"{length} characters: {path}"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows runtime libraries")
def test_cpp_runtime_libraries_are_shipped(layout):
    """
    torch and tensorflow need the Microsoft C++ runtime (msvcp140.dll). It
    has to be inside the folder, a clean Windows does not have it and
    installing it requires administrator rights.
    """
    names = {f.name.lower() for f in layout.runtime.rglob("*.dll")}
    assert "vcruntime140.dll" in names
    assert "msvcp140.dll" in names, (
        "msvcp140.dll is not shipped by any package. Add a package that "
        "contains it to [extra_requirements] windows in build_config.toml."
    )


# --------------------------------------------------------------------
# The pipeline inside the application (optional, takes minutes)
# --------------------------------------------------------------------
def test_pipeline_and_dashboard_inside_the_application(request, layout, workspace):
    model = request.config.getoption("--app-e2e-model")
    if not model:
        pytest.skip("needs --app-e2e-model <model>")
    result = run_python(
        layout, workspace, "-m", "bacpipe_app.e2e_check", "--model", model, timeout=3600
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    assert "e2e_check: pipeline ok" in result.stdout
    assert "e2e_check: dashboard ok" in result.stdout
    assert any((workspace / "bacpipe_results").rglob("*.npy")), "no embeddings were written"
