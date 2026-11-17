"""
Turn ``config.yaml`` / ``settings.yaml`` into a description of form fields.

This module is the reason the application does not need to be touched when a
setting is added to bacpipe: every key in the YAML files becomes a field, the
widget type is derived from the type of the default value, the help text is
taken from the comment above the key and the section headings are taken from
the ``#####  TITLE  #####`` banners. A banner that is followed by an empty line and
another banner, without a key in between, is a heading above the following
sections (``EVALUATION SETTINGS`` in settings.yaml).

=================  ==========================================================
default value      field kind (see ``form.py`` for the widget)
=================  ==========================================================
``True``/``False``  ``bool``         radio buttons
integer             ``int``          number input
float               ``float``        number input
string              ``text``         text input
list                ``multichoice``  dropdown with multiple selection
mapping             ``group``        box containing one field per entry
=================  ==========================================================

Whatever cannot be derived from the YAML file (the options of a dropdown,
which text fields are folders, fields that make no sense in the app) is
declared in ``form_overrides.yaml``.

The module has no dependency on Panel or bacpipe and is unit tested in
``app/tests/test_schema.py``.
"""

import fnmatch
import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

KINDS = (
    "bool",
    "int",
    "float",
    "text",
    "choice",
    "multichoice",
    "folder",
    "file",
    "yaml",
    "group",
)

OVERRIDES_FILE = Path(__file__).resolve().parent / "form_overrides.yaml"
OVERRIDE_KEYS = {
    "widget",
    "options",
    "options_from",
    "description",
    "label",
    "hidden",
    "section",
}

# a line like  "#########   PATH SETTINGS   ########"  or  "####-----####"
_BANNER = re.compile(r"^\s*#{3,}(.*?)#{3,}\s*$")
_RULE = re.compile(r"^\s*#{3,}[#\-=\s]*$")
_TOP_LEVEL_KEY = re.compile(r"^([A-Za-z_]\w*)\s*:(.*)$")


@dataclass
class FieldSpec:
    """Description of one form field."""

    path: tuple  # ("probe_configs", "config_1", "name")
    default: Any
    kind: str
    description: str = ""
    section: str = ""
    heading: str = ""  # static heading shown above the section of this field
    options: list | None = None
    label: str = ""
    hidden: bool = False
    children: list = field(default_factory=list)

    @property
    def key(self):
        return self.path[-1]

    @property
    def dotted(self):
        return ".".join(self.path)


# --------------------------------------------------------------------
# Comments and section banners
# --------------------------------------------------------------------
def parse_comments(yaml_text):
    """
    Extract the help text and the section of every top level key.

    Parameters
    ----------
    yaml_text : str
        content of a bacpipe YAML file

    Returns
    -------
    dict
        ``{key: {"description": str, "section": str, "heading": str}}`` in
        file order. ``heading`` is only set for the first key of the first
        section below a heading banner.
    """
    result = {}
    section = ""
    heading = ""  # heading banner waiting for the next section
    comments = []
    banner_titles = None  # list while we are inside a block of banner lines
    blank_in_banner = False  # an empty line was seen inside the banner block
    flow_depth = 0  # > 0 while inside a multi-line  [ ... ]  value
    in_block_value = False  # inside an indented (nested) value

    for line in yaml_text.splitlines():
        stripped = line.strip()
        if not stripped:
            blank_in_banner = banner_titles is not None
            continue

        if flow_depth > 0:
            flow_depth += _bracket_balance(stripped)
            if flow_depth <= 0:
                flow_depth, in_block_value = 0, False
            continue

        banner = _BANNER.match(line) or _RULE.match(line)
        if banner and not line.startswith((" ", "\t")):
            if banner_titles is None:
                banner_titles = []
            elif blank_in_banner and banner_titles:
                # a titled banner, an empty line, another banner: the first
                # one is a heading for what follows, not a section
                heading = _title(banner_titles)
                banner_titles = []
            blank_in_banner = False
            title = banner.group(1) if banner.groups() else ""
            title = title.strip("#-= \t")
            if title:
                banner_titles.append(title)
            continue

        if banner_titles is not None:
            # first line after a block of banners: the section starts here
            section = _title(banner_titles)
            banner_titles = None
            comments = []

        if stripped.startswith("#"):
            if in_block_value and line.startswith((" ", "\t")):
                continue  # comment belonging to a nested value
            comments.append(stripped.lstrip("#").strip())
            continue

        key_match = _TOP_LEVEL_KEY.match(line)
        if key_match:
            key, rest = key_match.groups()
            result[key] = {
                "description": _clean_description(comments),
                "section": section,
                "heading": heading,
            }
            heading = ""
            comments = []
            in_block_value = False
            flow_depth = max(_bracket_balance(rest), 0)
        else:
            # indented content of a nested mapping
            in_block_value = True

    return result


def _title(banner_titles):
    """['SLOW DOWN AUDIO', 'FOR BATS'] -> 'Slow Down Audio For Bats'"""
    return " ".join(" ".join(banner_titles).split()).title()


def _bracket_balance(text):
    """Number of ``[`` minus number of ``]`` outside of quotes and comments."""
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "", text)
    text = text.split("#")[0]
    return text.count("[") - text.count("]")


def _clean_description(comment_lines):
    """Join comment lines into one paragraph."""
    text = " ".join(line for line in comment_lines if line)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------
# Option sources for dropdowns
# --------------------------------------------------------------------
def load_constants(bacpipe_dir):
    """
    Load ``bacpipe/core/constants.py`` as a standalone module.

    The file only contains plain data, so it can be executed directly. Doing
    ``import bacpipe.core.constants`` instead would run ``bacpipe/__init__``
    first, which imports torch and tensorflow.
    """
    constants_file = Path(bacpipe_dir) / "core" / "constants.py"
    spec = importlib.util.spec_from_file_location(
        "_bacpipe_constants", constants_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option_sources(bacpipe_dir):
    """
    The lists that ``options_from`` in ``form_overrides.yaml`` can refer to.

    Returns
    -------
    dict
        name of the source mapped to a list of options
    """
    bacpipe_dir = Path(bacpipe_dir)
    sources = {
        # bacpipe.core.constants.supported_models
        "supported_models": list(load_constants(bacpipe_dir).supported_models),
        # one python file per dimensionality reduction model
        "dim_reduction_models": sorted(
            f.stem
            for f in (
                bacpipe_dir / "model_pipelines" / "dimensionality_reduction"
            ).glob("*.py")
            if not f.stem.startswith("_")
        )
        + ["None"],
    }
    try:
        import plotly.colors

        sources["plotly_colorscales"] = sorted(
            name.capitalize() for name in plotly.colors.named_colorscales()
        )
    except Exception:  # plotly missing: the field falls back to a text input
        pass
    return sources


# --------------------------------------------------------------------
# Building the field descriptions
# --------------------------------------------------------------------
def _read_overrides_file(overrides_file=OVERRIDES_FILE):
    return yaml.safe_load(Path(overrides_file).read_text(encoding="utf-8")) or {}


def load_overrides(overrides_file=OVERRIDES_FILE):
    """The field hints: ``{file name: {key pattern: override}}``."""
    data = _read_overrides_file(overrides_file)
    return {name: hints for name, hints in data.items() if name != "sections"}


def expanded_sections(name, overrides_file=OVERRIDES_FILE):
    """
    Titles of the sections of 'config' or 'settings' that start unfolded
    (``sections: <name>: expanded:`` in form_overrides.yaml).

    Returns
    -------
    list or None
        None means: no list given, all sections start unfolded
    """
    sections = _read_overrides_file(overrides_file).get("sections", {})
    return (sections.get(name) or {}).get("expanded")


def _override_for(dotted, overrides):
    """Merge all overrides whose pattern matches the dotted key path."""
    merged = {}
    for pattern, override in overrides.items():
        if fnmatch.fnmatchcase(dotted, pattern):
            merged.update(override or {})
    return merged


def _infer_kind(value):
    """Field kind for a default value, see the table in the module docstring."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "text"
    if isinstance(value, (list, tuple)):
        return "multichoice"
    if isinstance(value, dict):
        return "group"
    return "yaml"


def _make_spec(path, value, description, section, overrides, sources, heading=""):
    """Create the FieldSpec for one key (recursively for mappings)."""
    dotted = ".".join(path)
    override = _override_for(dotted, overrides)

    kind = override.get("widget") or _infer_kind(value)
    if kind not in KINDS:
        raise ValueError(
            f"form_overrides.yaml: unknown widget '{kind}' for '{dotted}'. "
            f"Use one of {KINDS}."
        )

    options = override.get("options")
    source_name = override.get("options_from")
    if source_name:
        # an unavailable source simply leaves the inferred widget in place
        options = sources.get(source_name)
    if options is not None:
        options = list(options)
        if isinstance(value, (list, tuple)):
            kind = "multichoice"
            options += [v for v in value if v not in options]
        else:
            kind = "choice"
            if value not in options:
                options.insert(0, value)
    elif kind == "multichoice":
        options = list(value)
    elif kind == "choice":
        kind = _infer_kind(value)

    spec = FieldSpec(
        path=tuple(path),
        default=value,
        kind=kind,
        description=override.get("description", description),
        section=override.get("section", section),
        heading=heading,
        options=options,
        label=override.get("label", path[-1]),
        hidden=bool(override.get("hidden", False)),
    )
    if kind == "group":
        spec.children = [
            _make_spec(path + (key,), child, "", section, overrides, sources)
            for key, child in value.items()
        ]
    return spec


def build_specs(yaml_text, overrides=None, sources=None):
    """
    Describe the form for one YAML file.

    Parameters
    ----------
    yaml_text : str
        content of ``config.yaml`` or ``settings.yaml``
    overrides : dict, optional
        ``{key pattern: override}`` for this file, see ``form_overrides.yaml``
    sources : dict, optional
        option lists that ``options_from`` can refer to

    Returns
    -------
    list of FieldSpec
        one entry per top level key, in file order
    """
    values = yaml.safe_load(yaml_text) or {}
    comments = parse_comments(yaml_text)
    overrides = overrides or {}
    sources = sources or {}
    return [
        _make_spec(
            (key,),
            value,
            comments.get(key, {}).get("description", ""),
            comments.get(key, {}).get("section", ""),
            overrides,
            sources,
            heading=comments.get(key, {}).get("heading", ""),
        )
        for key, value in values.items()
    ]


def iter_specs(specs):
    """Yield every spec including the children of groups."""
    for spec in specs:
        yield spec
        yield from iter_specs(spec.children)


def unused_override_patterns(specs, overrides):
    """
    Override patterns that match no key. They point at a setting that was
    renamed or removed in bacpipe, the tests fail when this is not empty.
    """
    dotted = [spec.dotted for spec in iter_specs(specs)]
    return [
        pattern
        for pattern in overrides
        if not any(fnmatch.fnmatchcase(name, pattern) for name in dotted)
    ]


def defaults(specs):
    """The default values as a plain dict, equal to ``yaml.safe_load``."""
    return {spec.key: spec.default for spec in specs}


def load_form_definitions(bacpipe_dir):
    """
    Read both YAML files of the bacpipe package.

    Returns
    -------
    dict
        ``{"config": [FieldSpec, ...], "settings": [FieldSpec, ...]}``
    """
    bacpipe_dir = Path(bacpipe_dir)
    overrides = load_overrides()
    sources = option_sources(bacpipe_dir)
    return {
        name: build_specs(
            (bacpipe_dir / f"{name}.yaml").read_text(encoding="utf-8"),
            overrides.get(name, {}),
            sources,
        )
        for name in ("config", "settings")
    }
