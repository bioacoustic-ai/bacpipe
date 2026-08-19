"""
Unit tests for the probe inference helpers in
``bacpipe.embedding_evaluation.probing.inference_probe``.

These cover the ``probing_a_model.ipynb`` example notebook workflow:
``prepare_probe_inference`` (loading a trained probe + label mapping) and
``run_probe_inference`` (applying the probe to embeddings, optionally
thresholding into a binary presence matrix).
"""

import json

import numpy as np
import pytest
import torch

from bacpipe.embedding_evaluation.probing.inference_probe import (
    prepare_probe_inference,
    run_probe_inference,
)
from bacpipe.embedding_evaluation.probing.train_probe import LinearProbe


def make_probe(in_dim=4, out_dim=2):
    return LinearProbe(in_dim=in_dim, out_dim=out_dim)


def make_embeds(n=4, dim=4):
    rng = np.random.RandomState(0)
    return rng.rand(n, dim)


class TestRunProbeInference:
    def test_binary_presence_shape_and_dtype(self):
        preds = run_probe_inference(
            "testmodel",
            make_probe(),
            threshold=0.5,
            embeds=make_embeds(),
            return_binary_presence=True,
            device="cpu",
        )
        # one prediction row per embedding, one column per class
        assert preds.shape == (4, 2)
        assert preds.dtype == np.int8
        assert set(np.unique(preds)) <= {0, 1}

    def test_probabilities_sum_to_one(self):
        preds = run_probe_inference(
            "testmodel",
            make_probe(),
            threshold=0.5,
            embeds=make_embeds(),
            return_binary_presence=False,
            device="cpu",
        )
        assert preds.dtype == np.float32
        assert np.allclose(preds.sum(axis=1), 1.0)

    def test_threshold_of_one_binarizes_to_zero(self):
        preds = run_probe_inference(
            "testmodel",
            make_probe(),
            threshold=1.0,
            embeds=make_embeds(),
            return_binary_presence=True,
            device="cpu",
        )
        assert np.all(preds == 0)

    def test_threshold_of_zero_binarizes_to_one(self):
        preds = run_probe_inference(
            "testmodel",
            make_probe(),
            threshold=0.0,
            embeds=make_embeds(),
            return_binary_presence=True,
            device="cpu",
        )
        assert np.all(preds == 1)

    def test_accepts_torch_tensor_input(self):
        embeds = torch.tensor(make_embeds(), dtype=torch.float32)
        preds = run_probe_inference(
            "testmodel",
            make_probe(),
            embeds=embeds,
            return_binary_presence=False,
            device="cpu",
        )
        assert preds.shape == (4, 2)

    def test_single_embedding_row(self):
        preds = run_probe_inference(
            "testmodel",
            make_probe(),
            embeds=np.array([[0.1, 0.2, 0.3, 0.4]]),
            return_binary_presence=True,
            device="cpu",
        )
        assert preds.shape == (1, 2)

    def test_kwargs_audio_dir_and_model_name_do_not_collide_with_loader(
        self, monkeypatch
    ):
        """
        Passing ``audio_dir`` (or ``model_name``) in kwargs must not crash
        when the Loader is constructed. Regression: the old
        ``**{**vars(settings), **kwargs}`` merge raised
        "TypeError: got multiple values for keyword argument 'audio_dir'"
        because ``audio_dir`` was passed both explicitly and via the merged
        dict.
        """
        import bacpipe

        captured = {}

        class FakeLoader:
            def __init__(self, audio_dir, model_name=None, **kwargs):
                captured["audio_dir"] = audio_dir
                captured["model_name"] = model_name
                captured["kwargs"] = kwargs

            def embeddings(self, return_type="array"):
                return make_embeds()

        monkeypatch.setattr(
            "bacpipe.core.experiment_manager.Loader", FakeLoader
        )

        preds = run_probe_inference(
            "testmodel",
            make_probe(),
            threshold=0.5,
            embeds=None,
            audio_dir="my_audio",
            model_name="ignored",
            device="cpu",
        )

        # the explicit ``model`` argument wins over the kwarg
        assert captured["model_name"] == "testmodel"
        assert captured["audio_dir"] == "my_audio"
        # the merged dict must not contain the explicitly-passed keys
        assert "audio_dir" not in captured["kwargs"]
        assert "model_name" not in captured["kwargs"]
        # settings defaults are still forwarded (merge behavior preserved)
        assert (
            captured["kwargs"].get("main_results_dir")
            == bacpipe.settings.main_results_dir
        )
        assert preds.shape == (4, 2)


class TestPrepareProbeInference:
    def test_loads_probe_and_label_mapping(self, tmp_path, monkeypatch):
        import bacpipe

        monkeypatch.setattr(bacpipe.settings, "device", "cpu")
        probe = make_probe(in_dim=4, out_dim=2)
        torch.save(probe.state_dict(), tmp_path / "linear_probe.pt")
        with open(tmp_path / "label2index.json", "w") as f:
            json.dump({"a": 0, "b": 1}, f)

        loaded, label2index = prepare_probe_inference(
            "testmodel", probe_path=str(tmp_path / "linear_probe.pt")
        )
        assert isinstance(loaded, LinearProbe)
        assert loaded.probe.in_features == 4
        assert loaded.probe.out_features == 2
        assert label2index == {"a": 0, "b": 1}

    def test_loaded_probe_predicts_like_original(self, tmp_path, monkeypatch):
        import bacpipe

        monkeypatch.setattr(bacpipe.settings, "device", "cpu")
        probe = make_probe(in_dim=4, out_dim=3)
        torch.save(probe.state_dict(), tmp_path / "linear_probe.pt")
        with open(tmp_path / "label2index.json", "w") as f:
            json.dump({"a": 0, "b": 1, "c": 2}, f)

        loaded, _ = prepare_probe_inference(
            "testmodel", probe_path=str(tmp_path / "linear_probe.pt")
        )
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(loaded(x), probe(x))


class TestProbingPipelineKwargsOverride:
    """
    Reproduces the bug found via ``bacpipe.play(..., audio_dir=<path>)`` where
    ``kwargs['audio_dir']`` differed from ``config.audio_dir``: downstream
    helpers that ignored the kwarg silently used the config default and wrote
    the probing results into the wrong directory.
    """

    def test_paths_use_kwarg_audio_dir_not_config_default(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        import pandas as pd

        import bacpipe
        from bacpipe import config, settings
        from bacpipe.embedding_evaluation.probing.probe import probing_pipeline

        # mimic ``bacpipe.play(audio_dir=...)``: the kwarg differs from the
        # config/settings defaults
        user_audio_dir = str(tmp_path / "user_audio")
        user_results_dir = str(tmp_path / "user_results")
        monkeypatch.setattr(config, "audio_dir", str(tmp_path / "config_audio"))
        monkeypatch.setattr(
            settings, "main_results_dir", str(tmp_path / "settings_results")
        )
        assert config.audio_dir != user_audio_dir
        assert settings.main_results_dir != user_results_dir

        seen = {}

        def fake_make_set_paths_func(audio_dir, main_results_dir=None, **kwargs):
            seen["audio_dir"] = audio_dir
            seen["main_results_dir"] = main_results_dir
            return lambda model: SimpleNamespace(
                probe_path=tmp_path / "probe",
                labels_path=tmp_path / "labels",
            )

        monkeypatch.setattr(bacpipe, "make_set_paths_func", fake_make_set_paths_func)
        monkeypatch.setattr(
            "bacpipe.embedding_evaluation.probing.probe."
            "generate_annotations_for_probing_task",
            lambda *a, **k: pd.DataFrame(
                {"predefined_set": ["train", "train"], "label": ["a", "b"]}
            ),
        )
        monkeypatch.setattr(
            "bacpipe.embedding_evaluation.probing.probe."
            "get_boolean_array_for_annotated_embeddings",
            lambda *a, **k: np.array([False, False]),
        )
        monkeypatch.setattr(
            "bacpipe.embedding_evaluation.probing.probe."
            "embeds_array_where_single_label",
            lambda embeds, ground_truth, bool_noise, df, **k: (df, embeds),
        )
        monkeypatch.setattr(
            "bacpipe.embedding_evaluation.probing.probe.train_probe",
            lambda *a, **k: "probe",
        )
        monkeypatch.setattr(
            "bacpipe.embedding_evaluation.probing.probe.eval_probe",
            lambda *a, **k: {},
        )

        _, label2index, _ = probing_pipeline(
            "mel",
            ground_truth="unused",
            embeds=np.zeros((2, 4)),
            paths=None,
            overwrite=True,
            name="linear",
            audio_dir=user_audio_dir,
            main_results_dir=user_results_dir,
        )

        # the probing paths must be derived from the kwargs, not from the
        # config/settings defaults
        assert seen["audio_dir"] == user_audio_dir
        assert seen["main_results_dir"] == user_results_dir
        assert label2index == {"a": 0, "b": 1}
