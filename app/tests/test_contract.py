"""
The contract between the application and bacpipe.

The application calls bacpipe through a handful of names (see
``BACPIPE_CONTRACT`` in ``bacpipe_app/dashboard.py``). These tests check
that the names still exist, by reading the source files - bacpipe itself is
not imported, so the tests need neither torch nor tensorflow and run in a
second.

If one of them fails, bacpipe was changed in a way the application has to
follow. The failing test says what to adapt.
"""

import ast
import inspect

import pytest

from bacpipe_app import PANEL_EXTENSIONS, paths
from bacpipe_app.dashboard import BACPIPE_CONTRACT


def parse(file):
    return ast.parse(file.read_text(encoding="utf-8"))


def defined_names(tree):
    """Top level functions, classes, variables and 'Class.method' names."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                names.update(
                    f"{node.name}.{member.name}"
                    for member in node.body
                    if isinstance(member, ast.FunctionDef)
                )
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize("file", sorted(BACPIPE_CONTRACT))
def test_names_used_by_the_application_exist(bacpipe_dir, file):
    names = defined_names(parse(bacpipe_dir / file))
    missing = [name for name in BACPIPE_CONTRACT[file] if name not in names]
    assert not missing, (
        f"bacpipe/{file} no longer provides {missing}. Adapt "
        "app/bacpipe_app/dashboard.py and worker.py, then update BACPIPE_CONTRACT."
    )


def test_play_hands_over_to_the_dashboard_with_keyword_arguments(bacpipe_dir):
    """
    worker.py replaces ``workflows.visualize_using_dashboard`` with a
    function that stores its keyword arguments. That only works while
    ``play`` calls it by that name and with keyword arguments only.
    """
    play = function(parse(bacpipe_dir / "core" / "workflows.py"), "play")
    calls = [
        node
        for node in ast.walk(play)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "visualize_using_dashboard"
    ]
    assert calls, "play() no longer calls visualize_using_dashboard(), adapt worker.py"
    for call in calls:
        assert not call.args, "play() passes positional arguments to the dashboard, adapt worker.py"


def test_play_accepts_bool_save_logs(bacpipe_dir):
    play = function(parse(bacpipe_dir / "core" / "workflows.py"), "play")
    assert "bool_save_logs" in [a.arg for a in play.args.args], "adapt worker.py"


def test_dashboard_function_takes_models_first(bacpipe_dir):
    """dashboard.py pops `models` and hands the rest to DashBoard."""
    tree = parse(bacpipe_dir / "embedding_evaluation" / "visualization" / "dashboard.py")
    arguments = [a.arg for a in function(tree, "visualize_using_dashboard").args.args]
    assert arguments[0] == "models", "adapt build_dashboard in app/bacpipe_app/dashboard.py"


def test_test_data_shortcut_is_still_understood(bacpipe_dir):
    """The default audio_dir is a special value that selects the example data."""
    source = (bacpipe_dir / "core" / "workflows.py").read_text(encoding="utf-8")
    assert paths.TEST_DATA_AUDIO_DIR in source
    assert (bacpipe_dir / "tests" / "test_data").is_dir()


def test_panel_extensions_of_bacpipe_are_loaded_by_the_application(bacpipe_dir):
    """
    Panel extensions must be loaded before the first page is shown. The
    dashboard appears later inside the page of the application, so the
    application has to load everything bacpipe asks for up front.
    """
    used = set()
    for file in bacpipe_dir.rglob("*.py"):
        if "tests" in file.parts:
            continue
        for node in ast.walk(parse(file)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "extension"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("pn", "panel")
            ):
                used.update(a.value for a in node.args if isinstance(a, ast.Constant))
    missing = used - set(PANEL_EXTENSIONS)
    assert not missing, f"add {sorted(missing)} to PANEL_EXTENSIONS in app/bacpipe_app/__init__.py"


def test_files_the_application_reads_exist(bacpipe_dir):
    for relative in (
        "config.yaml",
        "settings.yaml",
        "core/constants.py",
        "imgs/bacpipe_unlabelled.png",
        "imgs/bacpipe_favicon_white.png",
        "model_pipelines/dimensionality_reduction",
    ):
        assert (bacpipe_dir / relative).exists(), relative


def test_images_are_part_of_the_installed_package(repo_root):
    """The logo must be included in the wheel, the application loads it from there."""
    import tomllib

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["include"]
    assert "/bacpipe" in include


def test_helper_scripts_do_not_import_the_application():
    """worker.py, splash.py and dialogs.py are started by file path."""
    for script in ("worker.py", "splash.py", "dialogs.py"):
        tree = parse(paths.PACKAGE_DIR / script)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0 and not (node.module or "").startswith("bacpipe_app"), script


def test_worker_and_dashboard_use_the_same_handover():
    from bacpipe_app import dashboard, worker

    assert "dashboard_kwargs_file" in inspect.getsource(worker)
    assert "dashboard_kwargs_file" in inspect.signature(dashboard.build_dashboard).parameters
