"""
The configuration form: Panel widgets for the fields described in schema.py.

==============  =============================================================
field kind      widget
==============  =============================================================
``bool``        radio buttons True / False
``int``         integer input
``float``       float input
``text``        text input
``choice``      dropdown
``multichoice`` dropdown where several options can be selected
``folder``      text input with a "Browse" button (folder dialog of the system)
``file``        the same, the dialog selects a file
``yaml``        text input, the text is parsed as a YAML value
``group``       box containing the fields of a nested mapping
==============  =============================================================

Every field class has the same small interface: ``layout`` (what is shown),
``value`` (the python value to hand to bacpipe) and ``set_value``.
"""

import html
import math
import re
import subprocess
import threading
from pathlib import Path

import panel as pn
import yaml

from . import paths
from .theme import load_theme

_THEME = load_theme()

LABEL_STYLE = {"font-weight": "600", "font-size": "1em"}
# color and size of the help texts are set in theme.yaml
HELP_STYLE = {
    "color": _THEME["help_text"]["color"],
    "font-size": str(_THEME["help_text"]["font_size"]),
    "line-height": "1.4",
}
FIELD_WIDTH = 520
_URL = re.compile(r"(https?://[^\s)]+)")


def _help_pane(text):
    """Small grey help text below a widget (URLs become links)."""
    escaped = html.escape(text)
    escaped = _URL.sub(r'<a href="\1" target="_blank">\1</a>', escaped)
    return pn.pane.HTML(
        escaped,
        styles=HELP_STYLE,
        margin=(0, 10, 4, 10),
        sizing_mode="stretch_width",
        max_width=FIELD_WIDTH + 200,
    )


def _label_pane(text):
    return pn.pane.HTML(html.escape(text), styles=LABEL_STYLE, margin=(8, 10, 0, 10))


class Field:
    """Base class: label, widget and help text stacked in a column."""

    def __init__(self, spec):
        self.spec = spec
        self.widget = self.make_widget()
        self.set_value(spec.default)
        self.layout = self.make_layout()

    # -- to be provided by the subclasses ----------------------------
    def make_widget(self):
        raise NotImplementedError

    @property
    def value(self):
        return self.widget.value

    def set_value(self, value):
        self.widget.value = value

    def accepts(self, value):
        """Whether a remembered value can still be used for this field."""
        return True

    # -- shared ------------------------------------------------------
    def make_layout(self):
        parts = [_label_pane(self.spec.label), self.widget_row()]
        if self.spec.description:
            parts.append(_help_pane(self.spec.description))
        return pn.Column(*parts, margin=(0, 0, 8, 0), sizing_mode="stretch_width")

    def widget_row(self):
        return self.widget

    def restore(self, value):
        """Set a remembered value, silently keep the default if it is stale."""
        if self.accepts(value):
            try:
                self.set_value(value)
            except Exception:
                self.set_value(self.spec.default)


class BoolField(Field):
    def make_widget(self):
        return pn.widgets.RadioBoxGroup(
            options=[True, False], inline=True, margin=(4, 10)
        )

    def accepts(self, value):
        return isinstance(value, bool)


class IntField(Field):
    def make_widget(self):
        return pn.widgets.IntInput(width=200, margin=(4, 10))

    def accepts(self, value):
        return isinstance(value, int) and not isinstance(value, bool)


class FloatField(Field):
    def make_widget(self):
        default = self.spec.default
        # the step also defines how many decimals the input displays
        step = 10 ** (math.floor(math.log10(abs(default))) - 1) if default else 0.1
        return pn.widgets.FloatInput(step=min(step, 0.1), width=200, margin=(4, 10))

    def set_value(self, value):
        self.widget.value = float(value)

    def accepts(self, value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)


class TextField(Field):
    def make_widget(self):
        return pn.widgets.TextInput(width=FIELD_WIDTH, margin=(4, 10))

    @property
    def value(self):
        return self.widget.value.strip()

    def accepts(self, value):
        return isinstance(value, str)


class ChoiceField(Field):
    def make_widget(self):
        return pn.widgets.Select(
            options=list(self.spec.options), width=FIELD_WIDTH // 2, margin=(4, 10)
        )

    def accepts(self, value):
        return value in self.spec.options


class MultiChoiceField(Field):
    def make_widget(self):
        return pn.widgets.MultiChoice(
            options=list(self.spec.options),
            width=FIELD_WIDTH,
            margin=(4, 10),
            placeholder="click to select",
        )

    @property
    def value(self):
        return list(self.widget.value)

    def set_value(self, value):
        self.widget.value = list(value)

    def accepts(self, value):
        return isinstance(value, list) and all(v in self.spec.options for v in value)


class YamlField(TextField):
    """
    Text input for values without a fixed type, e.g. ``False`` or a number.
    The text is interpreted exactly like a value in the YAML file.
    """

    def make_widget(self):
        return pn.widgets.TextInput(width=200, margin=(4, 10))

    @property
    def value(self):
        text = self.widget.value.strip()
        try:
            return yaml.safe_load(text) if text else self.spec.default
        except yaml.YAMLError:
            return text

    def set_value(self, value):
        self.widget.value = str(value)

    def accepts(self, value):
        return not isinstance(value, (list, dict))


class FolderField(TextField):
    """Text input plus a button that opens the folder dialog of the system."""

    dialog_mode = "folder"

    def make_widget(self):
        self.browse_button = pn.widgets.Button(
            name="Browse …", width=90, margin=(4, 0)
        )
        self.browse_button.on_click(self._browse)
        # a note below the field that the pages can fill in, e.g. "this
        # file does not exist"; it never blocks anything
        self.note = pn.pane.Alert(
            "", alert_type="warning", visible=False, margin=(2, 10, 6, 10),
            max_width=FIELD_WIDTH + 200,
        )
        return pn.widgets.TextInput(width=FIELD_WIDTH, margin=(4, 10))

    def widget_row(self):
        return pn.Column(pn.Row(self.widget, self.browse_button), self.note)

    def show_note(self, text):
        """Show a warning below the field, an empty text hides it."""
        self.note.object = text
        self.note.visible = bool(text)

    def dialog_start_dir(self):
        """The folder the dialog opens in."""
        current = Path(self.widget.value).expanduser()
        for candidate in (current, current.parent):
            if candidate.is_dir():
                return str(candidate)
        return ""

    def _browse(self, event):
        """
        Show the file dialog of the system.

        The server runs on the user's own machine, so it can open a real
        dialog (a browser can not tell a web page the path of a folder). The
        dialog runs in its own process (dialogs.py) and a thread waits for
        it, so the page stays responsive while the dialog is open.
        """
        doc = pn.state.curdoc
        start_dir = self.dialog_start_dir()
        self.browse_button.disabled = True

        def wait_for_dialog():
            chosen = ""
            try:
                result = subprocess.run(
                    paths.child_command("dialogs.py", self.dialog_mode, start_dir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=paths.child_env(),
                    **paths.child_popen_kwargs(),
                )
                chosen = result.stdout.strip()
            except Exception:
                pass

            def apply_result():
                if chosen:
                    self.widget.value = chosen
                self.browse_button.disabled = False

            if doc is not None:
                doc.add_next_tick_callback(apply_result)
            else:
                apply_result()

        threading.Thread(target=wait_for_dialog, daemon=True).start()


class FileField(FolderField):
    """Text input plus a button that opens the file dialog of the system."""

    dialog_mode = "file"


class GroupField(Field):
    """A nested mapping: one field per entry, shown in a collapsible box."""

    def __init__(self, spec):
        self.children = [make_field(child) for child in spec.children]
        super().__init__(spec)

    def make_widget(self):
        return None

    @property
    def value(self):
        return {child.spec.key: child.value for child in self.children}

    def set_value(self, value):
        for child in self.children:
            if child.spec.key in value:
                child.set_value(value[child.spec.key])

    def restore(self, value):
        if isinstance(value, dict):
            for child in self.children:
                if child.spec.key in value:
                    child.restore(value[child.spec.key])

    def make_layout(self):
        visible = [c.layout for c in self.children if not c.spec.hidden]
        box = pn.Card(
            *visible,
            title=self.spec.label,
            collapsed=len(self.spec.path) > 1,
            sizing_mode="stretch_width",
            max_width=FIELD_WIDTH + 200,
            margin=(8, 10),
        )
        if self.spec.description:
            return pn.Column(
                box, _help_pane(self.spec.description), sizing_mode="stretch_width"
            )
        return box


FIELD_CLASSES = {
    "bool": BoolField,
    "int": IntField,
    "float": FloatField,
    "text": TextField,
    "choice": ChoiceField,
    "multichoice": MultiChoiceField,
    "folder": FolderField,
    "file": FileField,
    "yaml": YamlField,
    "group": GroupField,
}


def make_field(spec):
    """Create the field object for a FieldSpec."""
    return FIELD_CLASSES[spec.kind](spec)


class Form:
    """
    All fields of one YAML file, grouped into the sections of the file.

    Parameters
    ----------
    specs : list of FieldSpec
        result of ``schema.build_specs``
    expanded : list of str, optional
        titles of the sections that start unfolded. None = all of them.
    """

    def __init__(self, specs, expanded=None):
        self.fields = [make_field(spec) for spec in specs]

        sections = {}  # title -> layouts of its fields
        headings = {}  # title -> heading shown above that section
        for form_field in self.fields:
            if form_field.spec.hidden:
                continue
            title = form_field.spec.section or "Other"
            sections.setdefault(title, []).append(form_field.layout)
            if form_field.spec.heading:
                headings.setdefault(title, form_field.spec.heading)

        parts = []
        for title, layouts in sections.items():
            if title in headings:
                parts.append(
                    pn.pane.Markdown(f"### {headings[title]}", margin=(18, 10, 0, 10))
                )
            parts.append(
                pn.Card(
                    *layouts,
                    title=title,
                    collapsed=expanded is not None and title not in expanded,
                    sizing_mode="stretch_width",
                    margin=(6, 10),
                )
            )
        self.section_titles = list(sections)
        self.layout = pn.Column(*parts, sizing_mode="stretch_width")

    def values(self):
        """The current values as a dict with the same shape as the YAML file."""
        return {f.spec.key: f.value for f in self.fields}

    def restore(self, values):
        """Fill in remembered values, ignoring keys that no longer exist."""
        for form_field in self.fields:
            if form_field.spec.key in (values or {}):
                form_field.restore(values[form_field.spec.key])

    def reset(self):
        """Back to the defaults of the YAML file."""
        for form_field in self.fields:
            form_field.set_value(form_field.spec.default)

    def field(self, key):
        """The top level field for a key."""
        return next(f for f in self.fields if f.spec.key == key)
