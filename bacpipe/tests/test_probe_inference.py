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
