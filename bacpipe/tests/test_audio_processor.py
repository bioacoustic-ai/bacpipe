"""
Unit tests for the audio loading and windowing helpers in
``bacpipe.core.audio_processor``.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from bacpipe.core.audio_processor import AudioHandler

TEST_DATA_DIR = Path("bacpipe/tests/test_data")
TEST_AUDIO_FILE = (
    TEST_DATA_DIR / "audio/FewShot/CHE_01_20190101_163410.wav"
)


class DummyModel:
    """Minimal stand-in for a feature extractor model."""

    def __init__(
        self, sr=22050, segment_length=22050, only_embed_annotations=False
    ):
        self.sr = sr
        self.segment_length = segment_length
        self.only_embed_annotations = only_embed_annotations
        self.device = "cpu"
        self.model_name = "dummy"

    def preprocess(self, frames):
        return frames


def make_handler(model=None, **kwargs):
    if model is None:
        model = DummyModel()
    return AudioHandler(
        model,
        padding="wrap",
        audio_dir=TEST_DATA_DIR,
        **kwargs,
    )


class TestGetFileLength:
    def test_stores_duration_by_stem(self):
        handler = make_handler()
        handler.file_length = {}
        handler.get_file_length(TEST_AUDIO_FILE)
        assert TEST_AUDIO_FILE.stem in handler.file_length
        assert handler.file_length[TEST_AUDIO_FILE.stem] > 0

    def test_change_speed_divides_length(self):
        handler = make_handler(bool_change_speed=True, new_speed=2.0)
        handler.file_length = {}
        handler.get_file_length(TEST_AUDIO_FILE)
        normal = make_handler()
        normal.file_length = {}
        normal.get_file_length(TEST_AUDIO_FILE)
        assert handler.file_length[TEST_AUDIO_FILE.stem] == pytest.approx(
            normal.file_length[TEST_AUDIO_FILE.stem] / 2.0
        )


class TestLoadAndResample:
    def test_returns_mono_tensor_and_model_sr(self):
        handler = make_handler()
        handler.file_length = {}
        audio, sr = handler._load_and_resample(TEST_AUDIO_FILE)
        assert isinstance(audio, torch.Tensor)
        assert audio.shape[0] == 1
        assert sr == handler.model.sr
        assert audio.shape[1] > 0

    def test_missing_file_raises(self):
        handler = make_handler()
        handler.file_length = {}
        with pytest.raises(Exception):
            handler._load_and_resample(
                TEST_DATA_DIR / "does_not_exist.wav"
            )


class TestWindowAudio:
    def test_splits_into_segments_and_pads(self):
        handler = make_handler()
        audio = np.ones((1, 50_000))
        frames = handler._window_audio(audio)
        assert frames.shape == (3, handler.model.segment_length)

    def test_torch_input_is_supported(self):
        handler = make_handler()
        audio = torch.ones(1, 50_000)
        frames = handler._window_audio(audio)
        assert isinstance(frames, torch.Tensor)
        assert frames.shape == (3, handler.model.segment_length)


class TestLoadAudioBasedOnFixedSegmentLength:
    def test_computes_start_and_end_indices(self):
        handler = make_handler()
        audio = np.ones(50_000)
        starts, ends = handler._load_audio_based_on_fixed_segment_length(
            audio, segment_length=2.0
        )
        assert len(starts) == len(ends) == 50_000 // 2 + 1
        assert starts[0] == 0
        assert ends[0] == 2 * handler.model.sr


class TestLoadAndPadAudioBasedOnGrid:
    def test_pads_segments_to_model_length(self):
        handler = make_handler()
        handler.device = "cpu"
        audio = torch.ones(1, 50_000)
        starts = np.array([0, 44_100])
        ends = np.array([20_000, 60_000])
        segments = handler._load_and_pad_audio_based_on_grid(
            audio, starts, ends, Path("dummy.wav")
        )
        assert segments.shape == (2, handler.model.segment_length)


class TestOnlyLoadAnnotatedSegments:
    def test_loads_annotated_segments(self):
        handler = make_handler()
        handler.file_length = {}
        segments = handler._only_load_annotated_segments(TEST_AUDIO_FILE)
        assert isinstance(segments, torch.Tensor)
        assert segments.shape[1] == handler.model.segment_length
        assert segments.shape[0] > 0

    def test_no_annotations_raises(self):
        handler = make_handler()
        handler.file_length = {}
        with pytest.raises(AssertionError):
            handler._only_load_annotated_segments(
                TEST_DATA_DIR / "audio" / "unannotated_file.wav"
            )


class TestPrepareAudio:
    def test_full_audio_pipeline(self):
        handler = make_handler()
        handler.file_length = {}
        frames = handler.prepare_audio(TEST_AUDIO_FILE)
        assert isinstance(frames, torch.Tensor)
        assert frames.shape[1] == handler.model.segment_length
        assert handler.preprocessed_shape == tuple(frames.shape)

    def test_annotated_pipeline(self):
        model = DummyModel(only_embed_annotations=True)
        handler = make_handler(model=model)
        handler.file_length = {}
        frames = handler.prepare_audio(TEST_AUDIO_FILE)
        assert isinstance(frames, torch.Tensor)
        assert frames.shape[1] == handler.model.segment_length
        assert handler.preprocessed_shape == tuple(frames.shape)
