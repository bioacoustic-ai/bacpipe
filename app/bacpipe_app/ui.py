"""
The pages of the application.

There are three stages, all inside the same browser page:

1. ``config``   - the form for config.yaml and settings.yaml with the big
                  "Run pipeline" button
2. ``running``  - the terminal output of the pipeline
                  (``loading_dashboard`` and ``ended`` show the same view
                  with a different status line)
3. ``dashboard``- the regular bacpipe dashboard

``AppState`` exists once per application and knows which stage is active.
``Page`` exists once per browser tab and shows the active stage. A page looks
at the state a few times per second (``Page.tick``), which is how the output
of the pipeline gets into the terminal and how every open tab follows along
when the stage changes.
"""

import logging
import threading
import time
import traceback
from pathlib import Path

import panel as pn
import yaml

from . import DOCS_URL, REPO_URL, paths, schema
from .form import Form
from .runner import PipelineRun, absolute_path_values
from .theme import load_theme

logger = logging.getLogger("bacpipe_app")

SITE_TITLE = "bacpipe dashboard"
PAGE_TITLE = "Explore embeddings of audio data"
SIDEBAR_WIDTH = 180

# same text as in the sidebar of the bacpipe dashboard
CONTACT_TEXT = f"""
**Contact**

If you run into problems, please raise issues on github

Please collaborate and help make bacpipe as convenient for many as possible

🌍 [github]({REPO_URL})

📖 [documentation]({DOCS_URL})

To stay updated with new releases, subscribe to the [newsletter](https://buttondown.com/vskode)
"""

HELP_LINK_CSS = """
a.bacpipe-help {
  color: white; text-decoration: none; font-weight: 600;
  border: 1px solid rgba(255, 255, 255, 0.7); border-radius: 6px;
  padding: 5px 14px; white-space: nowrap;
}
a.bacpipe-help:hover { background: rgba(255, 255, 255, 0.2); }
"""

THEME = load_theme()

# the size of the text and of the drawing on the button come from theme.yaml
PLAY_BUTTON_CSS = """
:host .bk-btn {{
  font-size: {font_size}; font-weight: 600; line-height: 1;
  display: flex; align-items: center; justify-content: center; gap: 0.35em;
}}
""".format(**THEME["play_button"])

ANNOTATIONS_KEY = "annotations_filename"


# --------------------------------------------------------------------
# The annotations file
# --------------------------------------------------------------------
# bacpipe wants the *name* of the annotations file and looks for it in the
# audio folder. The form shows the complete path instead (so that it can be
# chosen with a file dialog) and these functions translate between the two.
def default_annotations_path(audio_dir, filename):
    """'/data/audio' + 'annotations.csv' -> '/data/audio/annotations.csv'"""
    return str(Path(audio_dir) / Path(filename).name) if audio_dir else filename


def annotations_for_bacpipe(shown_path, audio_dir):
    """
    The value for ``settings.annotations_filename``: the path relative to
    the audio folder if the file is inside it, otherwise the complete path
    (bacpipe joins it with the audio folder, which keeps a complete path as
    it is).
    """
    try:
        return Path(shown_path).relative_to(Path(audio_dir)).as_posix()
    except ValueError:
        return shown_path


def annotations_file_exists(shown_path, audio_dir):
    """Whether the annotations file shown in the form exists."""
    path = Path(shown_path).expanduser()
    if audio_dir == paths.TEST_DATA_AUDIO_DIR:
        # the example data lives inside the bacpipe package
        try:
            relative = path.relative_to(Path(paths.TEST_DATA_AUDIO_DIR))
            return (paths.bacpipe_package_dir() / "tests" / "test_data" / relative).is_file()
        except ValueError:
            pass
    return path.is_file()


# --------------------------------------------------------------------
# Checks before a run
# --------------------------------------------------------------------
def validate(config, settings):
    """
    Check the values of the form before the pipeline is started.

    Only problems that would make the pipeline fail right away are reported,
    everything else is left to bacpipe.

    Returns
    -------
    list of str
        messages for the user, empty if everything is fine
    """
    problems = []
    audio_dir = config.get("audio_dir", "")
    if audio_dir != paths.TEST_DATA_AUDIO_DIR:
        if not audio_dir:
            problems.append("Please choose the folder containing your audio files (audio_dir).")
        elif not Path(audio_dir).is_dir():
            problems.append(f"The audio folder does not exist: {audio_dir}")
    if not config.get("models") and not config.get("already_computed"):
        problems.append(
            "Please select at least one model (or set already_computed to "
            "True to use existing embeddings)."
        )
    return problems


# --------------------------------------------------------------------
# State shared by all browser tabs
# --------------------------------------------------------------------
class AppState:
    """What the application is currently doing."""

    def __init__(self):
        self.definitions = schema.load_form_definitions(paths.bacpipe_package_dir())
        self.stage = "config"
        self.run = None
        self.dashboard_layout = None
        self.update_notice = ""
        #: set by server.py, closes the application
        self.shutdown = lambda: None
        self._lock = threading.Lock()

    def folder_keys(self, name):
        """Keys of the fields of kind "folder" in 'config' or 'settings'."""
        return [s.key for s in self.definitions[name] if s.kind == "folder"]

    # -- stage changes -----------------------------------------------
    def start_run(self, config, settings):
        """Stage 1 -> 2: start the pipeline with the values of the form."""
        with self._lock:
            if self.run is not None and self.run.running:
                return
            self.dashboard_layout = None
            self.run = PipelineRun(config, settings, on_exit=self._run_ended)
            self.run.start()
            self.stage = "running"

    def _run_ended(self, run):
        """Called from the output thread of the run when the process ended."""
        if not run.wants_dashboard:
            self.stage = "ended"
            return
        # Stage 2 -> 3. Building the dashboard imports bacpipe into this
        # process and loads the results, which takes a few seconds. It
        # happens here, in the background thread, so the pages stay alive.
        self.stage = "loading_dashboard"
        run.output.write("\nLoading the dashboard ...\n")
        try:
            from .dashboard import build_dashboard

            self.dashboard_layout = build_dashboard(
                run.config, run.settings, run.dashboard_kwargs_file
            )
            self.stage = "dashboard"
        except BaseException:
            logger.exception("building the dashboard failed")
            run.output.write(
                "\n--- the dashboard could not be created ---\n"
                + traceback.format_exc()
            )
            self.stage = "ended"

    def back_to_config(self):
        """Back to stage 1 (the results of the run stay on disk)."""
        with self._lock:
            if self.run is not None and self.run.running:
                return
            self.dashboard_layout = None
            self.stage = "config"

    def stop_run(self):
        if self.run is not None:
            self.run.stop()

    @property
    def busy(self):
        """True while closing the app would interrupt something."""
        return self.stage in ("running", "loading_dashboard")


# --------------------------------------------------------------------
# One browser tab
# --------------------------------------------------------------------
class Page:
    """The page shown in one browser tab."""

    def __init__(self, state):
        self.state = state
        self.shown_stage = None
        self.loaded = False
        self.main = pn.Column(sizing_mode="stretch_width")

        self.template = pn.template.BootstrapTemplate(
            site=SITE_TITLE,
            title=PAGE_TITLE,
            favicon=str(paths.favicon_path()),
            header=[
                pn.pane.HTML(
                    '<a class="bacpipe-help" href="help" target="_blank">'
                    "❔ Help</a>",
                    stylesheets=[HELP_LINK_CSS],
                    align=("end", "center"),
                    margin=(0, 20),
                )
            ],
            main=[self.main],
        )
        self.render()
        pn.state.onload(self._on_load)
        pn.state.add_periodic_callback(self.tick, period=300)

    def _on_load(self):
        self.loaded = True

    # -- following the application state ------------------------------
    def tick(self):
        """Runs a few times per second in every open tab."""
        try:
            if self.state.stage != self.shown_stage:
                self.render()
            if self.shown_stage in ("running", "loading_dashboard", "ended"):
                self._update_run_view()
        except Exception:
            logger.exception("updating the page failed")

    def render(self):
        """Show the view of the active stage."""
        stage = self.state.stage
        previous, self.shown_stage = self.shown_stage, stage
        run_stages = ("running", "loading_dashboard", "ended")
        if stage in run_stages and previous in run_stages:
            return  # same view, only the status line changes
        if stage == "config":
            self.main.objects = [self._config_view()]
        elif stage == "dashboard":
            self.main.objects = [self._dashboard_view()]
        else:
            self.main.objects = [self._run_view()]

    # -- the sidebar shown in stage 1 and 2 ---------------------------
    def _sidebar(self):
        """
        Logo, contact text and close button, like the sidebar of the
        dashboard (which adds the same elements below its own widgets).
        """
        info = paths.build_info()
        version = f"bacpipe {info['version']}"
        if info.get("commit"):
            version += f" ({info['commit'][:7]})"

        close_button = pn.widgets.Button(name="❌ close bacpipe")
        close_button.on_click(lambda event: self.state.shutdown())

        parts = [
            pn.pane.PNG(str(paths.logo_path()), sizing_mode="scale_width"),
            pn.pane.Markdown(f"[live demo with explanations]({paths.demo_url()})"),
            pn.Spacer(height=10),
            pn.pane.Markdown(CONTACT_TEXT),
            pn.pane.Markdown(version),
        ]
        if self.state.update_notice:
            parts.append(
                pn.pane.Alert(self.state.update_notice, alert_type="info", margin=(5, 0))
            )
        parts.append(close_button)
        return pn.Column(*parts, width=SIDEBAR_WIDTH, margin=(10, 10))

    # -- stage 1: the form --------------------------------------------
    def _config_view(self):
        definitions = self.state.definitions
        self.config_form = Form(
            definitions["config"], expanded=schema.expanded_sections("config")
        )
        self.settings_form = Form(
            definitions["settings"], expanded=schema.expanded_sections("settings")
        )
        self._show_absolute_folders()
        self._restore_last_values()
        self._link_annotations_to_audio_dir()

        self.problems = pn.pane.Alert(
            "", alert_type="danger", visible=False, sizing_mode="stretch_width"
        )
        play = THEME["play_button"]
        run_button = pn.widgets.Button(
            name="Play the bacpipe",
            button_type="primary",
            icon=paths.button_icon_svg() or None,
            icon_size=str(play["icon_size"]),
            width=int(play["width"]),
            height=int(play["height"]),
            stylesheets=[PLAY_BUTTON_CSS],
            margin=(10, 10),
        )
        run_button.on_click(self._on_run)
        reset_button = pn.widgets.Button(
            name="Reset to defaults", width=150, margin=(int(play["height"]) // 2 - 6, 10)
        )
        reset_button.on_click(self._on_reset)

        content = pn.Column(
            pn.pane.Markdown("## Configure the pipeline"),
            pn.pane.Markdown(
                "These are the options of `config.yaml` and `settings.yaml`. "
                "Results and model checkpoints are stored in "
                f"`{paths.workspace_dir()}` unless you choose other folders. "
                "Your entries are remembered for the next start."
            ),
            pn.Tabs(
                ("Configuration", self.config_form.layout),
                ("Settings", self.settings_form.layout),
                # only the visible tab takes up space; otherwise the shorter
                # page would be as tall as the longer one, leaving a gap
                # above the button
                dynamic=True,
                sizing_mode="stretch_width",
            ),
            self.problems,
            pn.Row(run_button, reset_button),
            pn.Spacer(height=40),
            sizing_mode="stretch_width",
            max_width=1000,
        )
        return pn.Row(self._sidebar(), content, sizing_mode="stretch_width")

    def _show_absolute_folders(self):
        """Show where relative folders like 'bacpipe_results' really are."""
        for name, form in (("config", self.config_form), ("settings", self.settings_form)):
            keys = self.state.folder_keys(name)
            defaults = absolute_path_values(schema.defaults(form_specs(form)), keys)
            for key in keys:
                form.field(key).set_value(defaults[key])

    def _annotations_field(self):
        """The annotations field, None if bacpipe no longer has that setting."""
        try:
            field = self.settings_form.field(ANNOTATIONS_KEY)
        except StopIteration:
            return None
        return field if field.spec.kind == "file" else None

    def _link_annotations_to_audio_dir(self):
        """
        Keep the annotations file next to the audio folder: when the audio
        folder changes, the annotations path follows - unless the user chose
        a file somewhere else. A missing file is pointed out, but only if
        `only_embed_annotations` needs it, and it never blocks a run.
        """
        annotations = self._annotations_field()
        if annotations is None:
            return
        audio_dir = self.config_form.field("audio_dir").widget

        # a bare file name (the YAML default, or a value remembered by an
        # older version) is shown as a path inside the audio folder
        if Path(annotations.widget.value).name == annotations.widget.value:
            annotations.set_value(
                default_annotations_path(audio_dir.value, annotations.widget.value)
            )
        self._followed_audio_dir = audio_dir.value

        def audio_dir_changed(event):
            shown = Path(annotations.widget.value)
            if shown.parent == Path(self._followed_audio_dir):
                # still the automatic location: move along
                annotations.set_value(default_annotations_path(event.new, shown.name))
            self._followed_audio_dir = event.new
            self._check_annotations()

        audio_dir.param.watch(audio_dir_changed, "value")
        annotations.widget.param.watch(lambda event: self._check_annotations(), "value")
        try:
            only_annotations = self.settings_form.field("only_embed_annotations").widget
            only_annotations.param.watch(lambda event: self._check_annotations(), "value")
        except StopIteration:
            pass
        self._check_annotations()

    def _check_annotations(self):
        annotations = self._annotations_field()
        if annotations is None:
            return
        settings = self.settings_form.values()
        audio_dir = self.config_form.field("audio_dir").value
        missing = settings.get("only_embed_annotations") and not annotations_file_exists(
            annotations.value, audio_dir
        )
        annotations.show_note(
            "This file does not exist, and `only_embed_annotations` is True: "
            "without it there is nothing to embed."
            if missing
            else ""
        )

    def _restore_last_values(self):
        try:
            last = yaml.safe_load(paths.last_values_file().read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return
        if isinstance(last, dict):
            self.config_form.restore(last.get("config"))
            self.settings_form.restore(last.get("settings"))

    def _on_reset(self, event):
        self.config_form.reset()
        self.settings_form.reset()
        self._show_absolute_folders()
        annotations = self._annotations_field()
        if annotations is not None:
            audio_dir = self.config_form.field("audio_dir").value
            annotations.set_value(default_annotations_path(audio_dir, annotations.value))
            self._followed_audio_dir = audio_dir
            self._check_annotations()
        self.problems.visible = False

    def _on_run(self, event):
        config = absolute_path_values(
            self.config_form.values(), self.state.folder_keys("config")
        )
        settings = absolute_path_values(
            self.settings_form.values(), self.state.folder_keys("settings")
        )
        problems = validate(config, settings)
        if problems:
            self.problems.object = "\n\n".join(problems)
            self.problems.visible = True
            return
        try:
            # remembered the way they are shown in the form
            paths.last_values_file().write_text(
                yaml.safe_dump({"config": config, "settings": settings}, sort_keys=False),
                encoding="utf-8",
            )
        except OSError:
            logger.exception("could not remember the form values")
        if self._annotations_field() is not None:
            settings[ANNOTATIONS_KEY] = annotations_for_bacpipe(
                settings[ANNOTATIONS_KEY], config["audio_dir"]
            )
        self.state.start_run(config, settings)
        self.render()

    # -- stage 2: the terminal ----------------------------------------
    def _run_view(self):
        look = THEME["terminal"]  # colors and font size: theme.yaml
        self.terminal = pn.widgets.Terminal(
            options={
                "cursorBlink": False,
                "disableStdin": True,
                "scrollback": 20000,
                "fontSize": int(look["font_size"]),
                "theme": {
                    "background": look["background"],
                    "foreground": look["text"],
                    "cursor": look["background"],  # no visible cursor
                },
            },
            height=int(look["height"]),
            sizing_mode="stretch_width",
        )
        self.cursor = 0
        # give the browser a moment to create the terminal before the first
        # text is sent, text sent earlier than that would be dropped
        self.terminal_ready_at = time.monotonic() + 1.0

        self.spinner = pn.indicators.LoadingSpinner(
            value=True, size=28, color="primary", margin=(10, 10)
        )
        self.status = pn.pane.Markdown("", margin=(8, 10))
        self.stop_button = pn.widgets.Button(
            name="■ Stop", button_type="danger", width=120
        )
        self.stop_button.on_click(self._on_stop)
        self.back_button = pn.widgets.Button(
            name="⬅ Back to configuration", button_type="primary", width=220, visible=False
        )
        self.back_button.on_click(self._on_back)
        self.log_hint = pn.pane.Markdown("", margin=(8, 10))

        content = pn.Column(
            pn.pane.Markdown("## Pipeline output"),
            pn.Row(self.spinner, self.status),
            self.terminal,
            pn.Row(self.stop_button, self.back_button),
            self.log_hint,
            sizing_mode="stretch_width",
        )
        self._update_status()
        return pn.Row(self._sidebar(), content, sizing_mode="stretch_width")

    def _update_run_view(self):
        """Copy new output into the terminal and refresh the status line."""
        run = self.state.run
        if run is None:
            return
        if self.loaded and time.monotonic() >= self.terminal_ready_at:
            text, self.cursor = run.output.read_from(self.cursor)
            if text:
                # the terminal wants "\r\n" for a new line; a single "\r"
                # (used by progress bars to redraw their line) is kept
                self.terminal.write(text.replace("\r\n", "\n").replace("\n", "\r\n"))
        self._update_status()

    def _update_status(self):
        run, stage = self.state.run, self.state.stage
        if run is None:
            return
        minutes, seconds = divmod(int(run.elapsed()), 60)
        elapsed = f"{minutes}:{seconds:02d} min"
        if stage == "running":
            status = f"**Running the pipeline** … {elapsed}"
        elif stage == "loading_dashboard":
            status = "**Pipeline finished.** Loading the dashboard …"
        elif run.stopped_by_user:
            status = "**Stopped.**"
        elif run.succeeded and not run.wants_dashboard:
            status = (
                f"**Finished** after {elapsed}. The results are in "
                f"`{run.settings.get('main_results_dir', '')}`. The dashboard "
                "was not opened because `dashboard` is set to False."
            )
        elif run.succeeded:
            status = "**The dashboard could not be created**, see the output below."
        else:
            status = "**The pipeline ended with an error**, see the output below."
        if self.status.object != status:
            self.status.object = status

        ended = stage == "ended"
        self.spinner.value = not ended
        self.spinner.visible = not ended
        self.stop_button.visible = stage == "running"
        self.back_button.visible = ended
        hint = f"Complete output: `{run.log_file}`" if ended else ""
        if self.log_hint.object != hint:
            self.log_hint.object = hint

    def _on_stop(self, event):
        self.stop_button.disabled = True
        # stopping waits for the process, do not block the page meanwhile
        threading.Thread(target=self.state.stop_run, daemon=True).start()

    def _on_back(self, event):
        self.state.back_to_config()
        self.render()

    # -- stage 3: the dashboard ---------------------------------------
    def _dashboard_view(self):
        back_button = pn.widgets.Button(name="⬅ Back to configuration", width=220)
        back_button.on_click(self._on_back)
        return pn.Column(
            pn.Row(back_button),
            # the dashboard brings its own sidebar with logo and contact text
            self.state.dashboard_layout,
            sizing_mode="stretch_width",
        )


def form_specs(form):
    """The FieldSpecs of the top level fields of a form."""
    return [f.spec for f in form.fields]
