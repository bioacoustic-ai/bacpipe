"""
Unit tests for the Embedder / Classifier helpers in
``bacpipe.model_pipelines.runner``.

Covers the two regressions that surfaced after the pipeline refactors:

1. Duplicate / shared ``(start, end)`` annotations: the classifier used to
   deduplicate the start and end columns independently (``set`` + ``sort``),
   which either raised a "boolean index did not match ..." error or wrote
   timestamps that did not correspond to the embedded segments.
2. Device handling: batches were only moved to ``cuda``. On a Mac with
   ``mps`` the input stayed on the cpu while the weights lived on ``mps``,
   which raised "input and weight are not on the same device".
"""

import torch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bacpipe.model_pipelines.runner import Classifier, Embedder


class TestInitDataloader:
    class _BatchModel:
        batch_size = 4

    def test_tensor_input_still_returns_dataloader(self):
        embedder = object.__new__(Embedder)
        embedder.model = self._BatchModel()
        audio = torch.zeros(5, 22050)
        loader = embedder.init_dataloader(audio)
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 4


class TestFillDataframeWithClassiefierResults:
    """The annotation timestamps written next to classifier predictions must
    be the deduplicated (start, end) *pairs*, aligned with the number of
    embedded segments."""

    class _DummyModel:
        classes = ["Species A", "Species B"]
        device = "cpu"

    def _make_classifier(self, tmp_path):
        (tmp_path / "annotations.csv").write_text(
            "audiofilename,start,end,label:species\n"
            "rec.wav,0,5,Species A\n"
            "rec.wav,0,5,Species B\n"  # duplicate pair
            "rec.wav,0,10,Species C\n"  # shared start, different end
            "rec.wav,5,10,Species D\n"
        )
        classifier = Classifier(
            self._DummyModel(),
            model_name="dummy",
            audio_dir=tmp_path,
            main_results_dir=tmp_path / "results",
            classifier_threshold=0.5,
            use_folder_structure=False,
            only_embed_annotations=True,
            annotations_filename="annotations.csv",
        )
        classifier.predictions = torch.tensor(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
            ]
        )
        return classifier

    def test_duplicate_pairs_stay_aligned(self, tmp_path):
        classifier = self._make_classifier(tmp_path)

        class DummyLoader:
            audio_dir = tmp_path
            continue_incomplete_run = False

        classifier._fill_dataframe_with_classiefier_results(
            DummyLoader(), tmp_path / "rec.wav"
        )
        table = classifier.cumulative_annotations
        assert table["start"].tolist() == [0, 0, 5]
        assert table["end"].tolist() == [5, 10, 10]
        assert table["audiofilename"].tolist() == ["rec.wav"] * 3

    def test_mismatched_segment_count_does_not_raise(self, tmp_path):
        # Simulate the fallback path (e.g. offline classifier run) where the
        # number of embedded segments does not match the deduplicated
        # annotations: this must warn and skip, not raise.
        classifier = self._make_classifier(tmp_path)
        classifier.predictions = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

        class DummyLoader:
            audio_dir = tmp_path
            continue_incomplete_run = False

        classifier._fill_dataframe_with_classiefier_results(
            DummyLoader(), tmp_path / "rec.wav"
        )
        # no annotation rows were written for the un-alignable file
        assert not hasattr(classifier, "cumulative_annotations")


class TestSaveRavenTable:
    """In annotated-segment mode the Raven table must use the *per-file*
    ``(start, end)`` pairs (captured by ``_fill_dataframe...``) instead of
    the dataset-wide ``start_timestamps``/``end_timestamps``, which are in
    annotation-file order and would return another file's timestamps when
    indexed with the per-file prediction rows. When the per-file pairs are
    unavailable (alignment skipped) the table must be skipped, not written
    with wrong timestamps.
    """

    def _make_classifier(self, tmp_path, only_embed_annotations=True):
        classifier = object.__new__(Classifier)
        classifier.paths = SimpleNamespace(preds_path=tmp_path / "preds")
        classifier.paths.preds_path.mkdir(exist_ok=True, parents=True)
        classifier.model_name = "dummy"
        classifier.max_labels_per_timestamp = 1
        classifier.classifier_threshold = 0.5
        classifier.model = SimpleNamespace(
            classes=["Species A", "Species B", "Species C"], sr=22050
        )
        # Both rows have their top-1 species in class index 1 or 2 (class
        # index 0 is the below-threshold sentinel in ``save_Raven_table``).
        classifier.predictions = torch.tensor(
            [[0.1, 0.9, 0.2], [0.2, 0.1, 0.9]]
        )
        if only_embed_annotations:
            classifier.only_embed_annotations = True
            classifier.current_file_starts = np.array([0.0, 5.0])
            classifier.current_file_ends = np.array([5.0, 10.0])
        else:
            classifier.start_timestamps = np.array([0.0, 1.0])
            classifier.end_timestamps = np.array([1.0, 2.0])
        return classifier

    def _dest(self, tmp_path, file):
        return (
            tmp_path
            / "preds"
            / "raven_tables"
            / file.parent.relative_to(tmp_path)
            / f"{file.stem}_dummy.selection.table.txt"
        )

    def test_annotated_mode_uses_per_file_pairs(self, tmp_path):
        import numpy as np

        classifier = self._make_classifier(tmp_path, only_embed_annotations=True)
        file = tmp_path / "audio" / "rec.wav"
        file.parent.mkdir(parents=True, exist_ok=True)

        classifier.save_Raven_table(file, Path("audio"))

        dest = self._dest(tmp_path, file)
        assert dest.exists()
        table = pd.read_csv(dest, sep="\t", index_col=False)
        # per-file annotation timestamps, NOT the dataset-wide ones
        assert table["Begin Time (s)"].tolist() == [0.0, 5.0]
        assert table["End Time (s)"].tolist() == [5.0, 10.0]
        assert table["File Offset (s)"].tolist() == [0.0, 5.0]

    def test_annotated_mode_skips_when_pairs_missing(self, tmp_path, caplog):
        classifier = self._make_classifier(tmp_path, only_embed_annotations=True)
        classifier.current_file_starts = None
        classifier.current_file_ends = None
        file = tmp_path / "audio" / "rec.wav"
        file.parent.mkdir(parents=True, exist_ok=True)

        with caplog.at_level("WARNING"):
            classifier.save_Raven_table(file, Path("audio"))

        assert not self._dest(tmp_path, file).exists()
        assert any(
            "Raven table is skipped" in r.message for r in caplog.records
        )

    def test_full_mode_uses_segment_timestamps(self, tmp_path):
        classifier = self._make_classifier(tmp_path, only_embed_annotations=False)
        file = tmp_path / "audio" / "rec.wav"
        file.parent.mkdir(parents=True, exist_ok=True)

        classifier.save_Raven_table(file, Path("audio"))

        table = pd.read_csv(self._dest(tmp_path, file), sep="\t", index_col=False)
        assert table["Begin Time (s)"].tolist() == [0.0, 1.0]
        assert table["End Time (s)"].tolist() == [1.0, 2.0]


class TestGenerateEmbeddingsFromAudioArray:
    """``Embedder.generate_embeddings_from_audio_array`` embeds audio that is
    passed as an array instead of being read from files (see the
    ``simple_use_cases`` notebook).

    Regressions:
    1. The producer looped over single windows instead of batches, so the
       model was called once per window and the progress bar total did not
       match the number of batches.
    2. ``torch.tensor(...)`` was called on tensors (which warns and copies)
       and the batch was not moved to the model device, which broke ``mps``
       and ``cuda`` runs.
    3. Stacked segments that were longer than the model segment length were
       not windowed (see ``AudioHandler.window_audio``).
    """

    class _DummyModel:
        segment_length = 100
        batch_size = 4
        device = "cpu"
        sr = 100

        def __init__(self):
            self.batch_shapes = []
            self.devices = []

        def preprocess(self, audio):
            self.batch_shapes.append(tuple(audio.shape))
            self.devices.append(str(audio.device))
            return audio

    def _embedder(self, padding="wrap"):
        embedder = object.__new__(Embedder)
        embedder.model = self._DummyModel()
        embedder.padding = padding
        embedder.nr_parallel_workers = 1
        # one "embedding" per window: its mean, so that the order of the
        # embeddings can be checked against the order of the input
        embedder.get_embeddings_for_audio = lambda data: [
            float(row.mean()) for row in data
        ]
        return embedder

    def _long_recording(self, n_windows=10):
        # every window holds a constant value -> the mean identifies the window
        return np.concatenate(
            [np.full(100, float(i)) for i in range(n_windows)]
        )

    def test_long_recording_is_split_into_segments(self):
        embedder = self._embedder()
        embeddings = embedder.generate_embeddings_from_audio_array(
            self._long_recording()
        )
        # one embedding per segment, in the order of the recording
        assert embeddings == [float(i) for i in range(10)]

    def test_batches_use_the_model_batch_size(self):
        embedder = self._embedder()
        embedder.generate_embeddings_from_audio_array(self._long_recording())
        # 10 windows with batch_size 4 -> 3 batches (the progress bar total)
        assert embedder.model.batch_shapes == [(4, 100), (4, 100), (2, 100)]

    def test_two_dimensional_long_recording_is_supported(self):
        embedder = self._embedder()
        audio = self._long_recording().reshape(1, -1)
        embeddings = embedder.generate_embeddings_from_audio_array(audio)
        assert embeddings == [float(i) for i in range(10)]

    def test_stacked_segments_produce_one_embedding_each(self):
        embedder = self._embedder()
        audio = np.stack([np.full(100, float(i)) for i in range(5)])
        embeddings = embedder.generate_embeddings_from_audio_array(audio)
        assert embeddings == [float(i) for i in range(5)]
        assert embedder.model.batch_shapes == [(4, 100), (1, 100)]

    def test_stacked_short_segments_are_padded(self):
        embedder = self._embedder()
        # segments shorter than the model segment length are padded, one
        # window per segment
        audio = np.stack([np.full(40, float(i)) for i in range(3)])
        embeddings = embedder.generate_embeddings_from_audio_array(audio)
        assert embeddings == [float(i) for i in range(3)]
        assert embedder.model.batch_shapes == [(3, 100)]

    def test_stacked_long_segments_are_windowed(self):
        embedder = self._embedder()
        # segments longer than the model segment length are split up
        audio = np.stack([np.full(150, float(i)) for i in range(2)])
        embeddings = embedder.generate_embeddings_from_audio_array(audio)
        assert embeddings == [0.0, 0.0, 1.0, 1.0]
        assert embedder.model.batch_shapes == [(4, 100)]

    def test_single_window_batch_keeps_the_batch_dimension(self):
        embedder = self._embedder()
        embeddings = embedder.generate_embeddings_from_audio_array(
            np.full(100, 1.0)
        )
        # squeezing a batch of one window must not drop the batch dimension,
        # models expect (batch_size, samples)
        assert embedder.model.batch_shapes == [(1, 100)]
        assert embeddings == [1.0]

    def test_torch_input_is_supported(self):
        embedder = self._embedder()
        audio = torch.as_tensor(self._long_recording())
        embeddings = embedder.generate_embeddings_from_audio_array(audio)
        assert embeddings == [float(i) for i in range(10)]

    def test_list_input_is_supported(self):
        embedder = self._embedder()
        audio = [[float(i)] * 100 for i in range(3)]
        embeddings = embedder.generate_embeddings_from_audio_array(audio)
        assert embeddings == [float(i) for i in range(3)]

    def test_batches_are_moved_to_the_model_device(self):
        embedder = self._embedder()
        embedder.generate_embeddings_from_audio_array(self._long_recording())
        assert set(embedder.model.devices) == {"cpu"}

    class _FakeTqdm:
        """Minimal tqdm stand-in that records the total and the updates."""

        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total")
            self.updates = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, n=1):
            self.updates += n

    def _patch_tqdm(self, monkeypatch):
        import bacpipe.model_pipelines.runner as runner

        bars = []

        def fake_tqdm(*args, **kwargs):
            bar = self._FakeTqdm(*args, **kwargs)
            bars.append(bar)
            return bar

        monkeypatch.setattr(runner, "tqdm", fake_tqdm)
        return bars

    def test_progress_bar_counts_batches_not_windows(self, monkeypatch):
        bars = self._patch_tqdm(monkeypatch)
        embedder = self._embedder()
        embedder.generate_embeddings_from_audio_array(self._long_recording())
        # 10 windows with batch_size 4 -> 3 batches
        assert [bar.total for bar in bars] == [3]
        assert [bar.updates for bar in bars] == [3]

    def test_failing_batches_update_the_progress_bar_once(self, monkeypatch):
        bars = self._patch_tqdm(monkeypatch)
        embedder = self._embedder()

        def failing_embeddings(data):
            # the folder structure of a run is missing
            raise AttributeError("no folder structure")

        embedder.get_embeddings_for_audio = failing_embeddings
        embedder.generate_embeddings_from_audio_array(self._long_recording())
        # a skipped batch counts once, not twice
        assert [bar.updates for bar in bars] == [3]

    def test_failing_batches_are_skipped(self, caplog):
        embedder = self._embedder()

        def failing_preprocess(audio):
            raise ValueError("preprocessing failed")

        embedder.model.preprocess = failing_preprocess
        with caplog.at_level("WARNING", logger="bacpipe"):
            embeddings = embedder.generate_embeddings_from_audio_array(
                self._long_recording()
            )
        assert embeddings == []
        assert "preprocessing failed" in caplog.text

