"""
Unit tests for the workflow helpers in ``bacpipe.core.workflows``.
"""

from pathlib import Path

import pytest

from bacpipe.core.workflows import (
    _normalize_evaluation_task,
    confirm_model_name,
    evaluation_with_settings_already_exists,
    get_model_names,
)


class TestNormalizeEvaluationTask:
    """``evaluation_task`` may be passed as a single string by API users; it
    must be normalized to a list so the downstream ``in`` checks behave."""

    def test_string_is_wrapped_in_list(self):
        assert _normalize_evaluation_task("probing") == ["probing"]

    def test_list_passes_through(self):
        assert _normalize_evaluation_task(
            ["probing", "clustering"]
        ) == ["probing", "clustering"]

    def test_tuple_is_converted_to_list(self):
        assert _normalize_evaluation_task(("probing", "clustering")) == [
            "probing",
            "clustering",
        ]

    def test_none_becomes_empty_list(self):
        assert _normalize_evaluation_task(None) == []


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

    def test_skips_validation_for_custom_models(self, monkeypatch):
        # a custom model name that is not in ``supported_models`` must not
        # raise when a custom class is supplied for it
        names = get_model_names(
            ["birdnet", "mel"],
            audio_dir="audio",
            main_results_dir="results",
            embed_parent_dir="embeddings",
            CustomModels=[None, "MyModel"],
        )
        assert names == ["birdnet", "mel"]

    def test_custom_models_wrong_length_raises(self):
        with pytest.raises(AssertionError):
            get_model_names(
                ["birdnet", "mel"],
                audio_dir="audio",
                main_results_dir="results",
                embed_parent_dir="embeddings",
                CustomModels=["MyModel"],
            )

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


class TestEnsureModelsExist:
    def test_accepts_custom_model_name(self, monkeypatch, tmp_path):
        # ``mel`` is not part of ``supported_models``; with a custom class
        # provided the name must be accepted and no checkpoint download is
        # attempted because it is not in ``NEEDS_CHECKPOINT``.
        monkeypatch.setattr(
            "bacpipe.core.workflows.confirm_model_name", lambda m: m
        )
        from bacpipe.core.workflows import ensure_models_exist

        result = ensure_models_exist(
            model_base_path=str(tmp_path / "models"),
            model_names="mel",
            CustomModel=object,
        )
        assert result == tmp_path / "models"
        assert result.parent.exists()

    def test_custom_models_list_skips_only_custom_entries(
        self, monkeypatch, tmp_path
    ):
        validated = []
        monkeypatch.setattr(
            "bacpipe.core.workflows.confirm_model_name",
            lambda m: validated.append(m) or m,
        )
        from bacpipe.core.workflows import ensure_models_exist

        # Use the tiny torch-only ``bat`` model (no tensorflow import) and
        # pre-seed its checkpoint directory so the test stays offline.
        model_base_path = tmp_path / "models"
        (model_base_path / "bat").mkdir(parents=True)
        (model_base_path / "bat" / "model.pt").write_text("dummy")

        ensure_models_exist(
            model_base_path=model_base_path,
            model_names=["bat", "mel"],
            CustomModels=[None, object],
        )
        # only the built-in model name is validated against the supported list
        assert validated == ["bat"]

    def test_unknown_model_without_custom_raises(self, tmp_path):
        from bacpipe.core.workflows import ensure_models_exist

        with pytest.raises(NameError):
            ensure_models_exist(
                model_base_path=str(tmp_path / "models"),
                model_names="mel",
                CustomModels=[None],
            )


class TestRunPipelineForModelsCustomModels:
    def test_passes_custom_model_per_model(self, monkeypatch):
        # ``CustomModels`` must be consumed before the model names are
        # confirmed, otherwise a custom name that is not in
        # ``supported_models`` raises a NameError.
        confirm_calls = []
        single_model_calls = []

        def fake_confirm(model_name, **kwargs):
            confirm_calls.append((model_name, kwargs.get("CustomModel")))
            return model_name

        def fake_single_model(
            model_name, audio_dir, dim_reduction_model, CustomModel=None, **kwargs
        ):
            single_model_calls.append((model_name, CustomModel))

            class DummyLoader:
                files = ["some.npy"]

            return DummyLoader()

        monkeypatch.setattr(
            "bacpipe.core.workflows.confirm_model_name", fake_confirm
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.run_pipeline_for_single_model",
            fake_single_model,
        )
        from bacpipe.core.workflows import run_pipeline_for_models

        loader_dict = run_pipeline_for_models(
            models=["birdnet", "mel"],
            audio_dir="audio",
            dim_reduction_model=None,
            CustomModels=[None, "MyModel"],
        )

        assert set(loader_dict.keys()) == {"birdnet", "mel"}
        assert ("mel", "MyModel") in confirm_calls
        assert ("birdnet", None) in confirm_calls
        assert single_model_calls == [("birdnet", None), ("mel", "MyModel")]


class TestDimReductionNoneQuirk:
    def test_run_pipeline_for_single_model_none_skips_dim_reduction(
        self, monkeypatch
    ):
        # ``dim_reduction_model=None`` must mean the same as the string
        # ``"None"`` (no dimensionality reduction). Passing a python ``None``
        # used to trigger the dim-reduction branch, which crashed on
        # ``None.upper()`` further down the pipeline.
        from types import SimpleNamespace

        from bacpipe.core.workflows import run_pipeline_for_single_model

        monkeypatch.setattr(
            "bacpipe.core.workflows.confirm_model_name",
            lambda name, **kwargs: name,
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.replace_default_kwargs_with_user_kwargs",
            lambda **kwargs: kwargs,
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.make_set_paths_func",
            lambda *args, **kwargs: lambda model: SimpleNamespace(
                plot_path=Path("does/not/matter"), model_name=model
            ),
        )
        generate_calls = []

        def fake_generate_embeddings(**kwargs):
            generate_calls.append(kwargs)
            return SimpleNamespace(model_name="mel", embed_dir=Path("."))

        monkeypatch.setattr(
            "bacpipe.core.workflows.generate_embeddings", fake_generate_embeddings
        )

        loader = run_pipeline_for_single_model(
            model_name="mel",
            audio_dir="audio",
            dim_reduction_model=None,
            CustomModel="MyModel",
            testing=True,
        )

        # only the primary embedding generation runs; no dim-reduction call
        assert len(generate_calls) == 1
        assert "dim_reduction_model" not in generate_calls[0]
        assert loader.model_name == "mel"

    def test_cross_model_evaluation_none_skips_plot_comparison(
        self, monkeypatch, tmp_path
    ):
        # ``dim_reduction_model=None`` in ``cross_model_evaluation`` must not
        # attempt to plot a comparison (previously ``None == "None"`` is False,
        # so the plotting branch was entered and crashed downstream).
        from types import SimpleNamespace

        from bacpipe.core.workflows import cross_model_evaluation

        monkeypatch.setattr(
            "bacpipe.core.workflows.confirm_model_name",
            lambda name, **kwargs: name,
        )
        fake_get_paths = lambda model: SimpleNamespace(
            plot_path=tmp_path / "audio" / model
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.make_set_paths_func",
            lambda *args, **kwargs: fake_get_paths,
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.visualise_results_across_models",
            lambda *args, **kwargs: None,
        )
        plot_calls = []
        monkeypatch.setattr(
            "bacpipe.core.workflows.plot_comparison",
            lambda *args, **kwargs: plot_calls.append(True),
        )

        cross_model_evaluation(
            audio_dir=tmp_path / "audio",
            evaluation_task=["probing"],
            models=["mel", "perch_bird"],
            dim_reduction_model=None,
            CustomModels=["MyModel", None],
        )

        assert plot_calls == []


class TestGenerateEmbeddingsCustomModel:
    def test_passes_custom_model_to_ensure_models_exist(self, monkeypatch):
        # ``generate_embeddings`` must forward the custom model class so the
        # name is not validated against ``supported_models`` and no checkpoint
        # download is attempted for it.
        seen = {}
        monkeypatch.setattr(
            "bacpipe.core.workflows.confirm_model_name",
            lambda m, **kwargs: m,
        )

        def fake_ensure_models_exist(
            model_names, model_base_path=None, CustomModel=None
        ):
            seen["model_names"] = model_names
            seen["CustomModel"] = CustomModel
            return "models"

        monkeypatch.setattr(
            "bacpipe.core.workflows.ensure_models_exist", fake_ensure_models_exist
        )

        class DummyLoader:
            combination_already_exists = True
            embed_dir = None
            dim_reduction_model = False
            files = []

            def classifier_should_be_run(self, **kwargs):
                return False

        monkeypatch.setattr(
            "bacpipe.core.workflows.Loader",
            lambda **kwargs: DummyLoader(),
        )

        from bacpipe.core.workflows import generate_embeddings

        generate_embeddings(
            model_name="mel",
            audio_dir="audio",
            CustomModel="MyModel",
            testing=True,
        )
        assert seen["model_names"] == "mel"
        assert seen["CustomModel"] == "MyModel"


class TestPlayKwargsOverrideConfig:
    """
    ``bacpipe.play(...)`` must forward user-provided kwargs (e.g.
    ``audio_dir``) to the pipeline functions even when they differ from the
    defaults in ``config.yaml`` / ``settings.yaml``. This reproduces the
    setup of a real bug report: ``bacpipe.play(audio_dir=...)`` with an
    ``audio_dir`` that differed from ``config.audio_dir``.
    """

    def test_audio_dir_kwarg_overrides_config_default(self, monkeypatch, tmp_path):
        import bacpipe
        from bacpipe import config
        from bacpipe.core.workflows import play

        user_audio_dir = str(tmp_path / "user_audio")
        (tmp_path / "user_audio").mkdir()
        monkeypatch.setattr(config, "audio_dir", str(tmp_path / "config_audio"))
        assert config.audio_dir != user_audio_dir

        seen = {}
        monkeypatch.setattr(
            "bacpipe.core.workflows.ensure_models_exist",
            lambda *a, **k: str(tmp_path / "models"),
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.get_model_names", lambda **k: ["mel"]
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.evaluation_with_settings_already_exists",
            lambda **k: False,
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.run_pipeline_for_models",
            lambda **kwargs: seen.setdefault(
                "run_pipeline", kwargs.get("audio_dir")
            )
            or {},
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.model_specific_evaluation",
            lambda *args, **kwargs: seen.setdefault(
                "model_specific", kwargs.get("audio_dir")
            ),
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.cross_model_evaluation",
            lambda **kwargs: seen.setdefault(
                "cross_model", kwargs.get("audio_dir")
            ),
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.visualize_using_dashboard",
            lambda **kwargs: seen.setdefault(
                "dashboard", kwargs.get("audio_dir")
            ),
        )
        monkeypatch.setattr(
            "bacpipe.core.workflows.save_logs", lambda **kwargs: None
        )

        play(models=["mel"], audio_dir=user_audio_dir, overwrite=True)

        assert seen["run_pipeline"] == user_audio_dir
        assert seen["model_specific"] == user_audio_dir
        assert seen["cross_model"] == user_audio_dir
        assert seen["dashboard"] == user_audio_dir
