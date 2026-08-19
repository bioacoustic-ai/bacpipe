"""
Unit tests for the model base class helpers in
``bacpipe.model_pipelines.model_utils``.
"""

import os

import pytest
import torch

import bacpipe
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
    """The cuDNN version is mocked so the checks behave the same on CUDA
    and CPU-only runners (where ``torch.backends.cudnn.version()`` returns
    ``None``)."""

    def test_cpu_only_build_returns_false(self, monkeypatch):
        # torch built without CUDA reports ``None`` as the cuDNN version
        monkeypatch.setattr(torch.backends.cudnn, "version", lambda: None)
        assert check_if_cudnn_tensorflow_compatible() is False

    def test_required_cudnn_version_returns_true(self, monkeypatch):
        # cuDNN 9.3 is the required version for tensorflow
        monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9300)
        assert check_if_cudnn_tensorflow_compatible() is True

    def test_incompatible_cudnn_version_returns_false(self, monkeypatch):
        # e.g. cuDNN 9.0 -> (9000 % 1000) // 100 = 0 < 3
        monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9000)
        assert check_if_cudnn_tensorflow_compatible() is False

    def test_returns_bool(self, monkeypatch):
        for cudnn_version in (None, 9000, 9300, 8700):
            monkeypatch.setattr(
                torch.backends.cudnn, "version", lambda: cudnn_version
            )
            assert isinstance(check_if_cudnn_tensorflow_compatible(), bool)


class TestClassifierKwargs:
    """The pretrained classifier reads its options from ``settings`` unless
    the matching kwarg is passed explicitly."""

    def test_max_labels_per_timestamp_kwarg(self):
        from types import SimpleNamespace

        from bacpipe.model_pipelines.runner import Classifier

        clf = Classifier(
            SimpleNamespace(),
            "testmodel",
            audio_dir=".",
            main_results_dir=".",
            classifier_threshold=0.5,
            use_folder_structure=False,
            max_labels_per_timestamp=7,
        )
        assert clf.max_labels_per_timestamp == 7

        clf_default = Classifier(
            SimpleNamespace(),
            "testmodel",
            audio_dir=".",
            main_results_dir=".",
            classifier_threshold=0.5,
            use_folder_structure=False,
        )
        assert (
            clf_default.max_labels_per_timestamp
            == bacpipe.settings.max_labels_per_timestamp
        )


class TestCustomModelEmbedder:
    """Drive a ``ModelBaseClass`` subclass through the multithreaded embedder,
    mirroring the ``using_a_custom_model.ipynb`` example notebook."""

    def test_custom_model_runs_through_multithreaded_embedder(self):
        import librosa as lb
        import numpy as np
        import torch

        from bacpipe.model_pipelines.runner import Embedder

        class MyModel(ModelBaseClass):
            SAMPLE_RATE = 22050
            SEGMENT_LENGTH = 22050  # 1 second

            def __init__(self, **kwargs):
                super().__init__(
                    sr=self.SAMPLE_RATE,
                    segment_length=self.SEGMENT_LENGTH,
                    **kwargs,
                )

            def preprocess(self, audio):
                return audio

            def __call__(self, audio):
                audio = audio.cpu().numpy()
                mel_spec = lb.feature.melspectrogram(
                    y=audio, sr=self.SAMPLE_RATE
                )
                # the return array needs to be 2D
                mel_spec = mel_spec.reshape(
                    [len(mel_spec), mel_spec.shape[-2] * mel_spec.shape[-1]]
                )
                return torch.tensor(mel_spec)

        # 2 seconds of audio -> two 1-second windows
        audio = np.zeros(2 * 22050, dtype=np.float32)
        embedder = Embedder(
            model_name="mel",
            CustomModel=MyModel,
            audio_dir="bacpipe/tests/test_data",
            device="cpu",
            run_pretrained_classifier=False,
            nr_parallel_workers=2,
        )
        embeddings = embedder.embeddings_using_multithreading(audio)
        # one 1D feature vector per 1-second window; the list is extended
        # with each row of the per-batch model output
        assert len(embeddings) == 2
        assert all(e.ndim == 1 for e in embeddings)
        assert all(e.size > 0 for e in embeddings)

    def test_custom_model_attribute_contract(self):
        class MyModel(ModelBaseClass):
            SAMPLE_RATE = 48000
            SEGMENT_LENGTH = 48000 * 3

            def __init__(self, **kwargs):
                super().__init__(
                    sr=self.SAMPLE_RATE,
                    segment_length=self.SEGMENT_LENGTH,
                    **kwargs,
                )

        model = MyModel(
            model_name="mymodel",
            device="cpu",
            run_pretrained_classifier=False,
            global_batch_size=8,
        )
        assert model.sr == 48000
        assert model.segment_length == 48000 * 3
        # batch_size = 100000 * global_batch_size / segment_length
        assert model.batch_size == int(100_000 * 8 / (48000 * 3))

