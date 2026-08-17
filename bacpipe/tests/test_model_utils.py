"""
Unit tests for the model base class helpers in
``bacpipe.model_pipelines.model_utils``.
"""

import os

import pytest
import torch

from bacpipe.model_pipelines.model_utils import (
    ModelBaseClass,
    check_if_cudnn_tensorflow_compatible,
)


class TestModelBaseClass:
    def _make_model(self, **overrides):
        kwargs = dict(
            sr=22050,
            segment_length=22050,
            model_name="dummy",
            device="cpu",
            run_pretrained_classifier=False,
        )
        kwargs.update(overrides)
        return ModelBaseClass(**kwargs)

    def test_attributes_are_set(self):
        model = self._make_model(global_batch_size=8)
        assert model.sr == 22050
        assert model.segment_length == 22050
        assert model.device == "cpu"
        # batch_size = 100000 * global_batch_size / segment_length
        assert model.batch_size == int(100_000 * 8 / 22050)
        assert model.bool_classifier is False

    def test_no_batch_size_without_segment_length(self):
        model = self._make_model(segment_length=None)
        assert not hasattr(model, "batch_size")

    def test_bool_classifier_with_predictions(self):
        model = self._make_model(
            run_pretrained_classifier=True, classifier_predictions=True
        )
        assert model.bool_classifier is True

    def test_bool_classifier_false_without_predictions(self):
        model = self._make_model(run_pretrained_classifier=True)
        assert model.bool_classifier is False

    def test_cpu_device_sets_visible_devices(self):
        self._make_model()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"

    def test_model_base_path_is_set(self):
        model = self._make_model()
        assert model.model_base_path is not None

    def test_preprocessing_is_identity(self):
        model = self._make_model()
        audio = torch.zeros(2, 3)
        assert torch.equal(model.preprocessing(audio), audio)

    def test_call_is_identity(self):
        model = self._make_model()
        audio = torch.zeros(2, 3)
        assert torch.equal(model(audio), audio)

    def test_prepare_inference_handles_missing_model(self):
        model = self._make_model()
        # no self.model attribute -> logs and continues without raising
        model.prepare_inference()

    def test_tensorflow_model_cpu_stays_cpu(self):
        import bacpipe

        tf_model_name = bacpipe.TF_MODELS[0]
        model = self._make_model(model_name=tf_model_name)
        assert model.device == "cpu"


class TestCheckIfCudnnTensorflowCompatible:
    def test_returns_bool(self):
        assert isinstance(
            check_if_cudnn_tensorflow_compatible(), bool
        )
