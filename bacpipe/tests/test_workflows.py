"""
Unit tests for the workflow helpers in ``bacpipe.core.workflows``.
"""

import pytest

from bacpipe.core.workflows import (
    confirm_model_name,
    evaluation_with_settings_already_exists,
    get_model_names,
)


class TestConfirmModelName:
    def test_lowercases_valid_model(self):
        assert confirm_model_name("BirdNet") == "birdnet"

    def test_non_string_input_raises(self):
        with pytest.raises(ValueError):
            confirm_model_name(123)

    def test_unsupported_model_raises(self):
        with pytest.raises(NameError):
            confirm_model_name("not_a_real_model")

    def test_custom_model_string(self):
        assert (
            confirm_model_name("mycustom", CustomModel="custom") == "mycustom"
        )

    def test_custom_model_list(self):
        assert (
            confirm_model_name("mycustom", CustomModel=["custom"])
            == "mycustom"
        )

    def test_custom_model_none_list_falls_through(self):
        with pytest.raises(NameError):
            confirm_model_name("mycustom", CustomModel=[None])


class TestGetModelNames:
    def test_confirms_models(self, monkeypatch):
        monkeypatch.setattr(
            "bacpipe.core.workflows.confirm_model_name", lambda m: m
        )
        names = get_model_names(
            ["birdnet", "beats"],
            audio_dir="audio",
            main_results_dir="results",
            embed_parent_dir="embeddings",
        )
        assert names == ["birdnet", "beats"]

    def test_already_computed_finds_existing_dirs(self, tmp_path):
        audio_dir = tmp_path / "audio"
        main = tmp_path / "results"
        # dataset folder name is derived from the audio_dir stem
        dataset = main / "audio" / "embeddings"
        (dataset / "2024-01-01___birdnet-birdset").mkdir(parents=True)
        (dataset / "2024-01-01___beats-birdset").mkdir()
        names = get_model_names(
            ["ignored"],
            audio_dir=audio_dir,
            main_results_dir=main,
            embed_parent_dir="embeddings",
            already_computed=True,
        )
        assert sorted(names) == ["beats", "birdnet"]

    def test_already_computed_no_results_raises(self, tmp_path):
        with pytest.raises(ValueError):
            get_model_names(
                ["x"],
                audio_dir=tmp_path / "audio",
                main_results_dir=tmp_path / "missing",
                embed_parent_dir="embeddings",
                already_computed=True,
            )


class TestEvaluationWithSettingsAlreadyExists:
    def test_testing_mode_returns_false(self):
        assert (
            evaluation_with_settings_already_exists(
                "audio_dir", "umap", ["birdnet"], testing=True
            )
            is False
        )
