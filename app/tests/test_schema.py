"""
The form is generated from config.yaml and settings.yaml. These tests run
against the real files, so they fail when a change in the YAML files does
not translate into a sensible form.
"""

import pytest
import yaml

from bacpipe_app import schema

FILES = ("config", "settings")


@pytest.fixture(scope="module")
def definitions(bacpipe_dir):
    return schema.load_form_definitions(bacpipe_dir)


@pytest.fixture(scope="module")
def yaml_values(bacpipe_dir):
    return {
        name: yaml.safe_load((bacpipe_dir / f"{name}.yaml").read_text(encoding="utf-8"))
        for name in FILES
    }


# --------------------------------------------------------------------
# Every key of the YAML files is part of the form
# --------------------------------------------------------------------
@pytest.mark.parametrize("name", FILES)
def test_every_key_becomes_a_field(definitions, yaml_values, name):
    assert [spec.key for spec in definitions[name]] == list(yaml_values[name])


@pytest.mark.parametrize("name", FILES)
def test_defaults_equal_the_yaml_file(definitions, yaml_values, name):
    """Running without touching the form must equal `python run_pipeline.py`."""
    assert schema.defaults(definitions[name]) == yaml_values[name]


@pytest.mark.parametrize("name", FILES)
def test_nested_mappings_are_complete(definitions, yaml_values, name):
    def check(spec, value):
        if isinstance(value, dict):
            assert spec.kind == "group"
            assert [child.key for child in spec.children] == list(value)
            for child in spec.children:
                check(child, value[child.key])

    for spec in definitions[name]:
        check(spec, yaml_values[name][spec.key])


@pytest.mark.parametrize("name", FILES)
def test_field_kinds_match_the_value_types(definitions, name):
    """bool -> radio buttons, list -> multi-select, options -> dropdown."""
    for spec in schema.iter_specs(definitions[name]):
        assert spec.kind in schema.KINDS
        if spec.kind == "bool":
            assert isinstance(spec.default, bool)
        if isinstance(spec.default, list):
            assert spec.kind == "multichoice", spec.dotted
        if spec.kind in ("choice", "multichoice"):
            assert spec.options, spec.dotted
            selected = spec.default if spec.kind == "multichoice" else [spec.default]
            assert all(value in spec.options for value in selected), spec.dotted


# --------------------------------------------------------------------
# The hints in form_overrides.yaml
# --------------------------------------------------------------------
def test_overrides_only_use_known_hints():
    for file_overrides in schema.load_overrides().values():
        for pattern, override in file_overrides.items():
            assert set(override) <= schema.OVERRIDE_KEYS, pattern


@pytest.mark.parametrize("name", FILES)
def test_overrides_refer_to_existing_keys(definitions, name):
    """
    Fails when a setting mentioned in form_overrides.yaml was renamed or
    removed in bacpipe: update or delete the entry.
    """
    overrides = schema.load_overrides().get(name, {})
    assert schema.unused_override_patterns(definitions[name], overrides) == []


def test_overrides_only_mention_known_files():
    assert set(schema.load_overrides()) <= set(FILES)


# --------------------------------------------------------------------
# The dropdowns the user asked for
# --------------------------------------------------------------------
def test_models_are_taken_from_supported_models(definitions, bacpipe_dir):
    models = next(spec for spec in definitions["config"] if spec.key == "models")
    assert models.kind == "multichoice"
    assert models.options == list(schema.load_constants(bacpipe_dir).supported_models)
    assert len(models.options) > 10


def test_dim_reduction_models_are_taken_from_the_package(definitions, bacpipe_dir):
    spec = next(s for s in definitions["config"] if s.key == "dim_reduction_model")
    files = (bacpipe_dir / "model_pipelines" / "dimensionality_reduction").glob("*.py")
    assert spec.kind == "choice"
    assert set(spec.options) == {f.stem for f in files} | {"None"}


def test_hidden_fields_keep_their_default(definitions):
    hidden = [s.key for s in definitions["settings"] if s.hidden]
    assert "dashboard_port" in hidden and "testing" in hidden


# --------------------------------------------------------------------
# Comments and banners
# --------------------------------------------------------------------
EXAMPLE = """\
#########------------------#########
#########   FIRST SECTION  #########
#########------------------#########

# help for alpha
# second line
alpha: 1

beta: [
    "x",
#   "commented out",
]

# help for gamma
gamma:
  nested:
    # comment inside a nested value
    value: 2

####### SECOND SECTION #######
# help for delta, see https://example.org
delta: text
epsilon: True
"""


def test_comments_become_help_texts():
    parsed = schema.parse_comments(EXAMPLE)
    assert parsed["alpha"]["description"] == "help for alpha second line"
    assert parsed["beta"]["description"] == ""
    assert parsed["gamma"]["description"] == "help for gamma"
    assert parsed["delta"]["description"] == "help for delta, see https://example.org"
    assert parsed["epsilon"]["description"] == ""


def test_banners_become_sections():
    parsed = schema.parse_comments(EXAMPLE)
    assert [parsed[k]["section"] for k in ("alpha", "beta", "gamma")] == ["First Section"] * 3
    assert [parsed[k]["section"] for k in ("delta", "epsilon")] == ["Second Section"] * 2


def test_a_new_setting_needs_no_code_change():
    """A key that the application has never heard of still gets a field."""
    text = EXAMPLE + "\n# a brand new option\nbrand_new_option: 0.25\nnew_list: [a, b]\n"
    specs = {spec.key: spec for spec in schema.build_specs(text)}
    assert specs["brand_new_option"].kind == "float"
    assert specs["brand_new_option"].description == "a brand new option"
    assert specs["new_list"].kind == "multichoice"
    assert specs["new_list"].options == ["a", "b"]


def test_every_visible_top_level_field_has_a_section(definitions):
    for name in FILES:
        for spec in definitions[name]:
            if not spec.hidden:
                assert spec.section, f"{name}.{spec.key} has no section banner above it"


# --------------------------------------------------------------------
# Headings and unfolded sections
# --------------------------------------------------------------------
def test_banner_followed_by_another_banner_is_a_heading():
    text = (
        "####### GROUP TITLE #######\n"
        "\n"
        "#####-----------#####\n"
        "#####   FIRST   #####\n"
        "#####-----------#####\n"
        "one: 1\n"
        "two: 2\n"
        "##### SECOND #####\n"
        "three: 3\n"
    )
    parsed = schema.parse_comments(text)
    assert parsed["one"] == {"description": "", "section": "First", "heading": "Group Title"}
    assert parsed["two"]["heading"] == "" and parsed["two"]["section"] == "First"
    assert parsed["three"]["section"] == "Second" and parsed["three"]["heading"] == ""


def test_evaluation_is_a_heading_above_the_classifier_settings(definitions):
    spec = next(s for s in definitions["settings"] if s.key == "max_labels_per_timestamp")
    assert spec.section == "Integrated Classifier Settings"
    assert spec.heading == "Evaluation Settings"


@pytest.mark.parametrize("name", FILES)
def test_unfolded_sections_exist(definitions, name):
    """
    Fails when a banner in the YAML file was renamed: update the list under
    `sections:` in form_overrides.yaml.
    """
    titles = {spec.section for spec in definitions[name]}
    for title in schema.expanded_sections(name) or []:
        assert title in titles, f"'{title}' is not a section of {name}.yaml"
