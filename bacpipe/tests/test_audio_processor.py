"""
Unit tests for the audio loading and windowing helpers in
``bacpipe.core.audio_processor``.
"""

import shutil
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
        audio_dir=TEST_DATA_DIR,
        padding="wrap",
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
        audio, sr = handler.load_and_resample(TEST_AUDIO_FILE)
        assert isinstance(audio, torch.Tensor)
        assert audio.shape[0] == 1
        assert sr == handler.model.sr
        assert audio.shape[1] > 0

    def test_missing_file_raises(self):
        handler = make_handler()
        handler.file_length = {}
        with pytest.raises(Exception):
            handler.load_and_resample(
                TEST_DATA_DIR / "does_not_exist.wav"
            )


class TestWindowAudio:
    def test_splits_into_segments_and_pads(self):
        handler = make_handler()
        audio = np.ones((1, 50_000))
        frames = handler.window_audio(audio)
        assert frames.shape == (3, handler.model.segment_length)

    def test_torch_input_is_supported(self):
        handler = make_handler()
        audio = torch.ones(1, 50_000)
        frames = handler.window_audio(audio)
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
        segments = handler.only_load_annotated_segments(TEST_AUDIO_FILE)
        assert isinstance(segments, torch.Tensor)
        assert segments.shape[1] == handler.model.segment_length
        assert segments.shape[0] > 0

    def test_no_annotations_raises(self):
        handler = make_handler()
        handler.file_length = {}
        with pytest.raises(AssertionError):
            handler.only_load_annotated_segments(
                TEST_DATA_DIR / "audio" / "unannotated_file.wav"
            )

    def test_duplicate_pairs_load_each_window_once(self, tmp_path):
        # Several species can share one time window, and annotations can even
        # re-use the same start value for different windows. Regression test
        # for the old ``Series.unique()``-per-column deduplication, which
        # mispaired starts with ends (negative durations -> exceptions) or
        # loaded one segment per duplicate row.
        shutil.copy(TEST_AUDIO_FILE, tmp_path / TEST_AUDIO_FILE.name)
        (tmp_path / "annotations.csv").write_text(
            "audiofilename,start,end,label:species\n"
            f"{TEST_AUDIO_FILE.name},0,5,Species A\n"
            f"{TEST_AUDIO_FILE.name},0,5,Species B\n"  # duplicate pair
            f"{TEST_AUDIO_FILE.name},0,10,Species C\n"  # shared start
            f"{TEST_AUDIO_FILE.name},5,10,Species D\n"
            f"{TEST_AUDIO_FILE.name},100,105,Species E\n"  # out of range
            f"{TEST_AUDIO_FILE.name},10,10,Species F\n"  # zero duration
        )
        handler = AudioHandler(
            DummyModel(), audio_dir=tmp_path, padding="wrap"
        )
        handler.file_length = {}
        segments = handler.only_load_annotated_segments(
            tmp_path / TEST_AUDIO_FILE.name
        )
        assert isinstance(segments, torch.Tensor)
        assert segments.shape[1] == handler.model.segment_length
        # (0,5), (0,10) and (5,10) survive; the duplicate row, the
        # out-of-range row and the zero-duration row are dropped
        assert segments.shape[0] == 3

    def test_only_out_of_range_annotations_raises(self, tmp_path):
        shutil.copy(TEST_AUDIO_FILE, tmp_path / TEST_AUDIO_FILE.name)
        (tmp_path / "annotations.csv").write_text(
            "audiofilename,start,end,label:species\n"
            f"{TEST_AUDIO_FILE.name},500,505,Species A\n"
        )
        handler = AudioHandler(
            DummyModel(), audio_dir=tmp_path, padding="wrap"
        )
        handler.file_length = {}
        with pytest.raises(AssertionError):
            handler.only_load_annotated_segments(
                tmp_path / TEST_AUDIO_FILE.name
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
