"""
Tests of the application code, run from the checkout.

They need panel (see app/requirements-build.txt) but not the heavy
dependencies of bacpipe: the application never imports bacpipe before the
dashboard is shown.
"""

import subprocess
import sys
import time

import pytest
import yaml

from bacpipe_app import paths, readme_links, schema, updates
from bacpipe_app.runner import OutputBuffer, PipelineRun, absolute_path_values

pn = pytest.importorskip("panel")

from bacpipe_app import form, ui  # noqa: E402


@pytest.fixture(scope="module")
def definitions(bacpipe_dir):
    return schema.load_form_definitions(bacpipe_dir)


# --------------------------------------------------------------------
# The form
# --------------------------------------------------------------------
@pytest.mark.parametrize("name", ["config", "settings"])
def test_untouched_form_returns_the_yaml_values(definitions, bacpipe_dir, name):
    """The widgets must hand back exactly the values and types of the file."""
    values = form.Form(definitions[name]).values()
    expected = yaml.safe_load((bacpipe_dir / f"{name}.yaml").read_text(encoding="utf-8"))
    assert values == expected
    for key, value in expected.items():
        if isinstance(value, bool):
            assert isinstance(values[key], bool), key


def test_widget_types(definitions):
    config = form.Form(definitions["config"])
    settings = form.Form(definitions["settings"])
    # boolean -> radio buttons
    assert isinstance(config.field("overwrite").widget, pn.widgets.RadioBoxGroup)
    # list -> dropdown with multiple selection
    assert isinstance(config.field("models").widget, pn.widgets.MultiChoice)
    assert isinstance(settings.field("audio_suffixes").widget, pn.widgets.MultiChoice)
    # fixed set of options -> dropdown
    assert isinstance(config.field("dim_reduction_model").widget, pn.widgets.Select)
    assert isinstance(settings.field("device").widget, pn.widgets.Select)
    # everything else -> input fields
    assert isinstance(settings.field("global_batch_size").widget, pn.widgets.IntInput)
    assert isinstance(settings.field("classifier_threshold").widget, pn.widgets.FloatInput)
    assert isinstance(settings.field("label_column").widget, pn.widgets.TextInput)


def test_values_of_changed_widgets(definitions):
    settings = form.Form(definitions["settings"])
    settings.field("nr_parallel_workers").widget.value = "4"
    settings.field("max_labels_per_timestamp").widget.value = "False"
    settings.field("device").widget.value = "cuda"
    probe = settings.field("probe_configs")
    probe.children[0].children[0].widget.value = False  # config_1.bool
    values = settings.values()
    assert values["nr_parallel_workers"] == 4
    assert values["max_labels_per_timestamp"] is False
    assert values["device"] == "cuda"
    assert values["probe_configs"]["config_1"]["bool"] is False
    assert values["probe_configs"]["config_2"]["bool"] is True
    # 'None' stays the string that bacpipe expects
    assert values["clust_configs"]["config_1"]["params"]["n_clusters"] == "None"


def test_restore_ignores_stale_values(definitions):
    config = form.Form(definitions["config"])
    config.restore(
        {
            "models": ["birdnet", "a_model_that_was_removed"],  # not restorable
            "overwrite": "yes",  # wrong type
            "dim_reduction_model": "pca",
            "a_key_that_was_removed": 1,
        }
    )
    values = config.values()
    assert values["models"] == ["birdnet", "perch_bird"]
    assert values["overwrite"] is False
    assert values["dim_reduction_model"] == "pca"
    config.reset()
    assert config.values()["dim_reduction_model"] == "umap"


def test_hidden_fields_are_not_shown_but_returned(definitions):
    settings = form.Form(definitions["settings"])
    assert settings.values()["dashboard_port"] == 5006
    shown = str(settings.layout)
    assert settings.field("dashboard_port").spec.hidden
    assert "dashboard_port" not in shown


# --------------------------------------------------------------------
# Before and around a run
# --------------------------------------------------------------------
def test_validate(tmp_path):
    ok = {"audio_dir": paths.TEST_DATA_AUDIO_DIR, "models": ["birdnet"]}
    assert ui.validate(ok, {}) == []
    assert ui.validate({**ok, "audio_dir": str(tmp_path)}, {}) == []
    assert ui.validate({**ok, "audio_dir": str(tmp_path / "missing")}, {})
    assert ui.validate({**ok, "audio_dir": ""}, {})
    assert ui.validate({**ok, "models": []}, {})
    assert ui.validate({**ok, "models": [], "already_computed": True}, {}) == []


def test_relative_folders_are_resolved_against_the_workspace(isolated_workspace):
    values = {"main_results_dir": "bacpipe_results", "audio_dir": paths.TEST_DATA_AUDIO_DIR}
    result = absolute_path_values(values, ["main_results_dir", "audio_dir"])
    assert result["main_results_dir"] == str(isolated_workspace / "bacpipe_results")
    assert result["audio_dir"] == paths.TEST_DATA_AUDIO_DIR  # special value is kept


def test_output_buffer_serves_readers_at_different_positions():
    buffer = OutputBuffer(max_chars=10)
    buffer.write("abc")
    text, cursor = buffer.read_from(0)
    assert (text, cursor) == ("abc", 3)
    buffer.write("def")
    assert buffer.read_from(cursor) == ("def", 6)
    assert buffer.read_from(6) == ("", 6)
    buffer.write("0123456789")  # older text is dropped
    text, cursor = buffer.read_from(0)
    assert text.endswith("0123456789") and cursor == 16


def test_pipeline_run_collects_output_and_exit_code(monkeypatch, tmp_path):
    """The process handling, with a tiny script standing in for worker.py."""
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import sys, yaml\n"
        "run = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))\n"
        "print('models:', run['config']['models'])\n"
        "print('progress 1\\rprogress 2', file=sys.stderr)\n"
        "open(run['dashboard_kwargs_file'], 'wb').close()\n"
        "sys.exit(run['settings']['exit_code'])\n",
        encoding="utf-8",
    )
    original = paths.child_command
    monkeypatch.setattr(
        paths, "child_command", lambda name, *args: original(str(script), *args)
    )

    finished = []
    run = PipelineRun({"models": ["birdnet"]}, {"exit_code": 0}, on_exit=finished.append)
    run.start()
    run._process.wait(timeout=60)
    for _ in range(100):
        if finished:
            break
        time.sleep(0.05)
    assert finished == [run]
    output = run.output.read_from(0)[0]
    assert "models: ['birdnet']" in output
    assert "progress 1\rprogress 2" in output  # carriage returns survive
    assert run.succeeded and run.wants_dashboard
    assert run.log_file.read_bytes().decode("utf-8") == output

    failing = PipelineRun({"models": []}, {"exit_code": 3})
    failing.start()
    failing._process.wait(timeout=60)
    for _ in range(100):
        if not failing.running:
            break
        time.sleep(0.05)
    assert failing.returncode == 3 and not failing.succeeded
    assert "exit code 3" in failing.output.read_from(0)[0]


# --------------------------------------------------------------------
# The annotations file, the theme, the sidebar link
# --------------------------------------------------------------------
def test_annotations_path_follows_the_audio_folder(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    shown = ui.default_annotations_path(str(audio), "annotations.csv")
    assert shown == str(audio / "annotations.csv")
    # inside the audio folder: bacpipe gets the name relative to it
    assert ui.annotations_for_bacpipe(shown, str(audio)) == "annotations.csv"
    assert ui.annotations_for_bacpipe(str(audio / "sub" / "a.csv"), str(audio)) == "sub/a.csv"
    # somewhere else: bacpipe gets the complete path
    elsewhere = str(tmp_path / "other" / "labels.csv")
    assert ui.annotations_for_bacpipe(elsewhere, str(audio)) == elsewhere
    # the example data
    shown = ui.default_annotations_path(paths.TEST_DATA_AUDIO_DIR, "annotations.csv")
    assert ui.annotations_for_bacpipe(shown, paths.TEST_DATA_AUDIO_DIR) == "annotations.csv"


def test_annotations_file_existence(tmp_path):
    assert not ui.annotations_file_exists(str(tmp_path / "annotations.csv"), str(tmp_path))
    (tmp_path / "annotations.csv").write_text("a", encoding="utf-8")
    assert ui.annotations_file_exists(str(tmp_path / "annotations.csv"), str(tmp_path))
    shown = ui.default_annotations_path(paths.TEST_DATA_AUDIO_DIR, "annotations.csv")
    assert ui.annotations_file_exists(shown, paths.TEST_DATA_AUDIO_DIR)


def test_annotations_field_is_a_file_picker(definitions):
    settings = form.Form(definitions["settings"])
    assert isinstance(settings.field("annotations_filename"), form.FileField)
    assert settings.field("annotations_filename").dialog_mode == "file"
    assert settings.field("main_results_dir").dialog_mode == "folder"


def test_sections_unfolded_by_default(definitions):
    settings = form.Form(definitions["settings"], expanded=schema.expanded_sections("settings"))
    cards = [c for c in settings.layout if isinstance(c, pn.Card)]
    unfolded = [card.title for card in cards if not card.collapsed]
    assert unfolded == schema.expanded_sections("settings")
    headings = [p.object for p in settings.layout if isinstance(p, pn.pane.Markdown)]
    assert headings == ["### Evaluation Settings"]
    # no list given: everything unfolded
    config = form.Form(definitions["config"], expanded=None)
    assert not any(card.collapsed for card in config.layout if isinstance(card, pn.Card))


def test_theme_file_is_complete_and_falls_back_to_defaults(tmp_path):
    from bacpipe_app import theme

    loaded = theme.load_theme()
    assert set(loaded) == set(theme.DEFAULTS)
    for group, values in theme.DEFAULTS.items():
        assert set(loaded[group]) == set(values)
    assert loaded["terminal"]["background"].startswith("#")
    broken = tmp_path / "theme.yaml"
    broken.write_text("terminal: {text: '#123456'}\nhelp_text: [oops", encoding="utf-8")
    assert theme.load_theme(broken) == theme.DEFAULTS
    partial = tmp_path / "partial.yaml"
    partial.write_text("terminal: {text: '#123456'}", encoding="utf-8")
    assert theme.load_theme(partial)["terminal"]["text"] == "#123456"
    assert theme.load_theme(partial)["terminal"]["font_size"] == theme.DEFAULTS["terminal"]["font_size"]


def test_demo_link_names_the_system(monkeypatch):
    for system, ref in (("windows", "windows_app"), ("macos", "mac_app"), ("linux", "linux_app")):
        monkeypatch.setattr(paths, "build_info", lambda system=system: {"system": system})
        assert paths.demo_url() == f"https://bacpipe.siriusly.me/?ref={ref}"


def test_play_button_drawing_is_available():
    svg = paths.button_icon_svg()
    assert svg.startswith("<svg") and "currentColor" in svg


def test_plain_addresses_become_links():
    text = 'under this link: https://buttondown.com/vskode. And [x](https://a.b) <img src="https://c.d/e.png">'
    result = readme_links.link_bare_urls(text)
    assert "<https://buttondown.com/vskode>." in result
    assert "[x](https://a.b)" in result and 'src="https://c.d/e.png"' in result


# --------------------------------------------------------------------
# Help page and update notice
# --------------------------------------------------------------------
def test_readme_links_are_rewritten(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "logo.png").write_bytes(b"png")
    text = (
        "![logo](src/logo.png) [docs](docs/page.md) [web](https://example.org) "
        '[top](#top) <img src="src/logo.png"> <img src="src/missing.png">'
    )
    result = readme_links.rewrite_links(text, tmp_path, "https://github.com/o/r")
    assert "](readme_assets/src/logo.png)" in result
    assert "](https://github.com/o/r/blob/main/docs/page.md)" in result
    assert "](https://example.org)" in result and "](#top)" in result
    assert 'src="readme_assets/src/logo.png"' in result
    assert 'src="https://github.com/o/r/blob/main/src/missing.png"' in result
    assert readme_links.local_images(text, tmp_path) == ["src/logo.png"]


def test_help_page_renders_the_readme(repo_root):
    from bacpipe_app.help_page import render_help_page

    page = render_help_page()
    assert "bacpipe help" in page and "<h1" in page
    assert '<a href="https://buttondown.com/vskode">' in page  # the newsletter link
    for image in readme_links.local_images(
        (repo_root / "README.md").read_text(encoding="utf-8"), repo_root
    ):
        assert f"readme_assets/{image}" in page


def test_version_comparison():
    assert updates.is_newer("v1.4.0", "1.3.5")
    assert updates.is_newer("1.3.10", "1.3.9")
    assert not updates.is_newer("v1.3.5", "1.3.5")
    assert not updates.is_newer("v1.2.9", "1.3.5")
    assert not updates.is_newer("nightly", "1.3.5")


# --------------------------------------------------------------------
# The whole application, started like a user would (without a browser)
# --------------------------------------------------------------------
def test_application_starts_and_serves_its_pages(repo_root):
    """`--selftest` starts the server, requests the pages and exits."""
    result = subprocess.run(
        [sys.executable, str(repo_root / "app" / "run_app.py"), "--selftest"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for page in ("/", "/help", "/ping"):
        assert f"selftest: {page} ok" in result.stdout
