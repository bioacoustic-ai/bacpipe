"""
Unit tests for the audio loading and windowing helpers in
``bacpipe.core.audio_processor``.
"""

import shutil
from pathlib import Path

import librosa as lb
import numpy as np
import pandas as pd
import pytest
import torch

import bacpipe
from bacpipe.core.audio_processor import (
    AudioHandler,
    _ModelStub,
    _get_model_constants,
)

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


class TestWindowAudioStackedInput:
    """``window_audio`` also receives stacked arrays, e.g. when a user passes
    an array of audio segments to ``generate_embeddings_from_audio_array``.

    Regression: as soon as the input had more than one row it was returned
    without being reshaped. Rows that were longer than the model segment
    length therefore stayed glued together, which produced too few frames
    with the wrong number of samples (and made the model fail or return
    embeddings for the wrong audio).
    """

    def test_already_stacked_segments_are_returned_unchanged(self):
        handler = make_handler()
        seg_len = handler.model.segment_length
        audio = np.ones((5, seg_len))
        frames = handler.window_audio(audio)
        assert frames.shape == (5, seg_len)

    def test_stacked_short_segments_are_padded(self):
        handler = make_handler()
        seg_len = handler.model.segment_length
        audio = np.stack([np.ones(10_000) * (i + 1) for i in range(4)])
        frames = handler.window_audio(audio)
        # one padded frame per segment
        assert frames.shape == (4, seg_len)
        # the padding must not mix the segments with each other
        for i, frame in enumerate(frames):
            assert torch.unique(frame).tolist() == [float(i + 1)]

    def test_stacked_long_segments_are_split_into_windows(self):
        handler = make_handler()
        seg_len = handler.model.segment_length
        # every row holds one full segment plus a bit -> 2 windows per row
        audio = np.stack([np.ones(seg_len + 5_000) * (i + 1) for i in range(3)])
        frames = handler.window_audio(audio)
        assert frames.shape == (6, seg_len)
        # the two windows of a row belong to the same (padded) segment
        for i in range(3):
            assert torch.unique(frames[i * 2]).tolist() == [float(i + 1)]
            assert torch.unique(frames[i * 2 + 1]).tolist() == [float(i + 1)]

    def test_stacked_torch_input_returns_tensor(self):
        handler = make_handler()
        seg_len = handler.model.segment_length
        frames = handler.window_audio(torch.ones(3, seg_len + 5_000))
        assert isinstance(frames, torch.Tensor)
        assert frames.shape == (6, seg_len)


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


class TestModelPassedAsString:
    """``AudioHandler`` accepts either a model object or the name of a model
    supported by bacpipe.

    Loading, resampling and windowing audio only requires the sample rate
    and the segment length of a model, both of which are defined in the
    module of the model. Passing a name therefore neither instantiates the
    model nor requires its checkpoint.
    """

    def test_name_provides_sample_rate_and_segment_length(self):
        from bacpipe.model_pipelines.feature_extractors import aves_especies

        handler = AudioHandler(
            model="aves_especies", audio_dir=TEST_DATA_DIR
        )
        assert isinstance(handler.model, _ModelStub)
        assert handler.model.name == "aves_especies"
        assert handler.model.sr == aves_especies.SAMPLE_RATE
        assert (
            handler.model.segment_length == aves_especies.LENGTH_IN_SAMPLES
        )
        # the model itself (and with it its preprocessing) was not loaded
        assert not hasattr(handler.model, "preprocess")

    def test_documented_insect459_example(self):
        # the values of the example used in the docstrings and the notebook
        from bacpipe.model_pipelines.feature_extractors import insect459

        handler = AudioHandler(model="insect459", audio_dir=TEST_DATA_DIR)
        assert handler.model.sr == insect459.SAMPLE_RATE
        assert handler.model.segment_length == insect459.LENGTH_IN_SAMPLES

    def test_name_is_case_insensitive(self):
        handler = AudioHandler(
            model="AVES_Especies", audio_dir=TEST_DATA_DIR
        )
        assert handler.model.name == "aves_especies"

    def test_inherited_model_constants_are_found(self):
        # ``birdaves_especies`` only subclasses the model of
        # ``aves_especies`` and uses its SAMPLE_RATE / LENGTH_IN_SAMPLES.
        # Regression: reading the constants from the module of the model
        # itself raised an AttributeError for this supported model.
        from bacpipe.model_pipelines.feature_extractors import aves_especies

        handler = AudioHandler(
            model="birdaves_especies", audio_dir=TEST_DATA_DIR
        )
        assert handler.model.sr == aves_especies.SAMPLE_RATE
        assert (
            handler.model.segment_length == aves_especies.LENGTH_IN_SAMPLES
        )

    def test_all_supported_models_can_be_passed_as_string(self):
        # every supported model has to provide the two constants, either in
        # its own module or in the module of the model it subclasses
        for model_name in bacpipe.supported_models:
            if model_name in bacpipe.TF_MODELS:
                continue
            name, sr, segment_length = _get_model_constants(model_name)
            assert name == model_name
            assert sr > 0 and segment_length > 0

    def test_unsupported_name_raises_name_error(self):
        with pytest.raises(NameError) as excinfo:
            AudioHandler(model="insect459t", audio_dir=TEST_DATA_DIR)
        assert "insect459t" in str(excinfo.value)

    def test_model_object_is_used_unchanged(self):
        model = DummyModel()
        handler = AudioHandler(model, audio_dir=TEST_DATA_DIR)
        assert handler.model is model

    def test_windowing_uses_the_model_windows(self):
        handler = AudioHandler(
            model="aves_especies", audio_dir=TEST_DATA_DIR, padding="wrap"
        )
        audio, sr = handler.load_and_resample(TEST_AUDIO_FILE)
        assert sr == handler.model.sr
        frames = handler.window_audio(audio)
        assert frames.shape[1] == handler.model.segment_length

    def test_changed_sample_rate_and_segment_length_are_used(self):
        # the documented way of deviating from the model defaults:
        # ``aud.model.sr = ...``
        handler = AudioHandler(
            model="aves_especies", audio_dir=TEST_DATA_DIR, padding="wrap"
        )
        handler.model.sr = 32_000
        handler.model.segment_length = 3 * handler.model.sr
        audio, sr = handler.load_and_resample(TEST_AUDIO_FILE)
        assert sr == 32_000
        frames = handler.window_audio(audio)
        assert frames.shape[1] == 3 * 32_000

    def test_only_embed_annotations_is_taken_from_the_kwargs(self):
        handler = AudioHandler(
            model="aves_especies",
            audio_dir=TEST_DATA_DIR,
            only_embed_annotations=True,
        )
        assert handler.model.only_embed_annotations is True

    def test_only_embed_annotations_defaults_to_the_settings(self):
        handler = AudioHandler(
            model="aves_especies", audio_dir=TEST_DATA_DIR
        )
        assert (
            handler.model.only_embed_annotations
            == bacpipe.settings.only_embed_annotations
        )

    def test_change_speed_works_without_an_embedder(self):
        # Regression: ``load_and_resample`` read ``self.model_name``, which
        # is only set by ``Embedder``, so changing the speed with a
        # standalone AudioHandler raised an AttributeError.
        handler = AudioHandler(
            model="aves_especies",
            audio_dir=TEST_DATA_DIR,
            bool_change_speed=True,
            new_speed=2.0,
        )
        assert handler.model_name == "aves_especies"
        audio, _ = handler.load_and_resample(TEST_AUDIO_FILE)
        assert audio.shape[0] == 1 and audio.shape[1] > 0

    def test_change_speed_works_for_a_model_without_a_name(self):
        model = DummyModel()
        del model.model_name
        handler = AudioHandler(
            model,
            audio_dir=TEST_DATA_DIR,
            bool_change_speed=True,
            new_speed=2.0,
        )
        assert handler.model_name == ""
        audio, _ = handler.load_and_resample(TEST_AUDIO_FILE)
        assert audio.shape[0] == 1 and audio.shape[1] > 0



class TestPrepareAudioWithModelPassedAsString:
    """``prepare_audio`` applies the model specific preprocessing, so a
    model that was passed by name has to be loaded at that point.

    ``bacpipe.Embedder`` is replaced by a stand-in here, so no checkpoint is
    downloaded while testing.
    """

    @staticmethod
    def _patch_embedder(monkeypatch, **model_kwargs):
        loaded = []

        class FakeEmbedder:
            def __init__(self, model_name, **kwargs):
                loaded.append((model_name, kwargs))
                # the real model comes with its own sample rate and segment
                # length, which must not overwrite the values the audio was
                # loaded with
                self.model = DummyModel(**model_kwargs)

        monkeypatch.setattr(bacpipe, "Embedder", FakeEmbedder)
        return loaded

    def test_model_is_loaded_when_the_preprocessing_is_needed(
        self, monkeypatch
    ):
        loaded = self._patch_embedder(monkeypatch)
        handler = AudioHandler(
            model="aves_especies", audio_dir=TEST_DATA_DIR, padding="wrap"
        )
        assert isinstance(handler.model, _ModelStub)
        assert loaded == []

        frames = handler.prepare_audio(TEST_AUDIO_FILE)

        assert [name for name, _ in loaded] == ["aves_especies"]
        assert isinstance(handler.model, DummyModel)
        assert frames.shape[1] == handler.model.segment_length

    def test_model_is_only_loaded_once(self, monkeypatch):
        loaded = self._patch_embedder(monkeypatch)
        handler = AudioHandler(
            model="aves_especies", audio_dir=TEST_DATA_DIR, padding="wrap"
        )
        handler.prepare_audio(TEST_AUDIO_FILE)
        handler.prepare_audio(TEST_AUDIO_FILE)
        assert len(loaded) == 1

    def test_changed_sample_rate_survives_loading_the_model(
        self, monkeypatch
    ):
        # Regression: the model that was loaded replaced the stub entirely,
        # so a sample rate or segment length the user had changed was
        # silently reset to the model defaults.
        self._patch_embedder(monkeypatch, sr=16_000, segment_length=16_000)
        handler = AudioHandler(
            model="aves_especies", audio_dir=TEST_DATA_DIR, padding="wrap"
        )
        handler.model.sr = 32_000
        handler.model.segment_length = 2 * handler.model.sr

        frames = handler.prepare_audio(TEST_AUDIO_FILE)

        assert handler.model.sr == 32_000
        assert handler.model.segment_length == 2 * 32_000
        assert frames.shape[1] == 2 * 32_000

    def test_annotated_pipeline_with_model_passed_as_string(
        self, monkeypatch
    ):
        # Regression: the stub had no ``only_embed_annotations`` attribute,
        # so ``prepare_audio`` raised an AttributeError for a model that was
        # passed by name.
        self._patch_embedder(monkeypatch)
        handler = AudioHandler(
            model="aves_especies",
            audio_dir=TEST_DATA_DIR,
            padding="wrap",
            only_embed_annotations=True,
        )
        frames = handler.prepare_audio(TEST_AUDIO_FILE)
        assert frames.shape[1] == handler.model.segment_length
        # 13 annotated windows for this file, i.e. not the whole recording
        assert frames.shape[0] == 13

    def test_annotations_df_kwarg_is_forwarded(self, monkeypatch):
        self._patch_embedder(monkeypatch)
        annots = pd.read_csv(TEST_DATA_DIR / "annotations.csv")
        handler = AudioHandler(
            model="aves_especies",
            audio_dir=TEST_DATA_DIR,
            padding="wrap",
            only_embed_annotations=True,
            annotations_df=annots[annots.start < 10],
        )
        frames = handler.prepare_audio(TEST_AUDIO_FILE)
        # only the two annotated windows below 10 seconds
        assert frames.shape[0] == 2



class TestOnlyLoadAnnotatedSegmentsWithAnnotationsDf:
    """The annotations can be passed as a dataframe instead of being read
    from the annotations csv in the audio directory.

    Regression: the dataframe was used verbatim, so a dataframe covering
    several files loaded the time spans annotated for *other* files from the
    current file, and duplicated ``(start, end)`` pairs were loaded once per
    row.
    """

    @staticmethod
    def _annotations():
        return pd.read_csv(TEST_DATA_DIR / "annotations.csv")

    def test_dataframe_returns_the_same_segments_as_the_csv(self):
        handler = make_handler()
        from_csv = handler.only_load_annotated_segments(TEST_AUDIO_FILE)
        from_df = handler.only_load_annotated_segments(
            TEST_AUDIO_FILE, annotations_df=self._annotations()
        )
        assert from_df.shape == from_csv.shape
        assert torch.equal(from_df, from_csv)

    def test_only_the_annotated_segments_are_returned(self):
        handler = make_handler()
        annots = pd.DataFrame(
            {
                "audiofilename": [
                    "audio/FewShot/CHE_01_20190101_163410.wav"
                ]
                * 2,
                "start": [3, 10],
                "end": [4, 11],
            }
        )
        segments = handler.only_load_annotated_segments(
            TEST_AUDIO_FILE, annotations_df=annots
        )
        # one segment per annotation, not the whole recording
        assert segments.shape == (2, handler.model.segment_length)
        # and the returned audio is the audio of the annotated time spans
        for row, (start, end) in enumerate(
            zip(annots["start"], annots["end"])
        ):
            expected, _ = lb.load(
                str(TEST_AUDIO_FILE),
                sr=handler.model.sr,
                mono=True,
                offset=start,
                duration=end - start,
            )
            assert np.allclose(
                segments[row].numpy(),
                expected[: handler.model.segment_length],
                atol=1e-6,
            )

    def test_dataframe_of_all_files_is_filtered_by_file(self):
        handler = make_handler()
        annots = self._annotations()
        file_rows = annots[
            annots.audiofilename
            == "audio/FewShot/CHE_01_20190101_163410.wav"
        ]
        segments = handler.only_load_annotated_segments(
            TEST_AUDIO_FILE, annotations_df=annots
        )
        assert segments.shape[0] == len(file_rows)
        assert segments.shape[0] < len(annots)

    def test_duplicate_time_windows_are_loaded_once(self):
        handler = make_handler()
        annots = pd.DataFrame(
            {
                "start": [0, 0, 5],
                "end": [5, 5, 10],
                "label:species": ["Species A", "Species B", "Species C"],
            }
        )
        segments = handler.only_load_annotated_segments(
            TEST_AUDIO_FILE, annotations_df=annots
        )
        assert segments.shape[0] == 2

    def test_dataframe_without_filename_column_is_used_as_is(self):
        handler = make_handler()
        segments = handler.only_load_annotated_segments(
            TEST_AUDIO_FILE,
            annotations_df=pd.DataFrame({"start": [0, 5], "end": [1, 6]}),
        )
        assert segments.shape == (2, handler.model.segment_length)

    def test_invalid_annotations_are_skipped(self):
        handler = make_handler()
        annots = pd.DataFrame(
            {
                # zero duration, out of range and one valid annotation
                "start": [10, 500, 2],
                "end": [10, 505, 3],
            }
        )
        segments = handler.only_load_annotated_segments(
            TEST_AUDIO_FILE, annotations_df=annots
        )
        assert segments.shape == (1, handler.model.segment_length)

    def test_dataframe_without_rows_for_the_file_raises(self):
        handler = make_handler()
        annots = self._annotations()
        other_file = annots[
            annots.audiofilename
            == "audio/FewShot/CHE_02_20190101_183410.wav"
        ]
        with pytest.raises(AssertionError):
            handler.only_load_annotated_segments(
                TEST_AUDIO_FILE, annotations_df=other_file
            )

    def test_missing_start_and_end_columns_raise(self):
        handler = make_handler()
        annots = self._annotations().rename(columns={"start": "begin"})
        with pytest.raises(AssertionError) as excinfo:
            handler.only_load_annotated_segments(
                TEST_AUDIO_FILE, annotations_df=annots
            )
        assert "start" in str(excinfo.value)

    def test_empty_dataframe_raises(self):
        handler = make_handler()
        with pytest.raises(AssertionError):
            handler.only_load_annotated_segments(
                TEST_AUDIO_FILE,
                annotations_df=pd.DataFrame(columns=["start", "end"]),
            )

    def test_none_falls_back_to_the_annotations_csv(self):
        handler = make_handler()
        segments = handler.only_load_annotated_segments(
            TEST_AUDIO_FILE, annotations_df=None
        )
        assert segments.shape[0] == 13

    def test_file_path_can_be_a_string(self):
        handler = make_handler()
        segments = handler.only_load_annotated_segments(
            str(TEST_AUDIO_FILE), annotations_df=self._annotations()
        )
        assert segments.shape == (13, handler.model.segment_length)

