"""
Reads the colors and font sizes from ``theme.yaml``. Missing or invalid
values fall back to ``DEFAULTS``.
"""

from pathlib import Path

import yaml

THEME_FILE = Path(__file__).resolve().parent / "theme.yaml"

DEFAULTS = {
    "terminal": {
        "background": "#1b1f3b",
        "text": "#f0ff66",
        "font_size": 16,
        "height": 600,
    },
    "help_text": {"color": "#333841", "font_size": "1em"},
    "play_button": {
        "font_size": "2.7em",
        "icon_size": "68px",
        "width": 480,
        "height": 88,
    },
}


def load_theme(theme_file=THEME_FILE):
    """
    Returns
    -------
    dict
        the content of theme.yaml, completed with the defaults
    """
    try:
        custom = yaml.safe_load(Path(theme_file).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        custom = {}
    theme = {}
    for group, defaults in DEFAULTS.items():
        values = custom.get(group)
        theme[group] = {**defaults, **(values if isinstance(values, dict) else {})}
    return theme
