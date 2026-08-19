"""
Round-trip and alignment tests for the CSV/parquet files that link
embeddings to labels, ground truth and visualization points.

The result files (``metadata_labels``, ``ground_truth_*``, ``*_all_predictions``)
are written without an index column (``index=False``) so that importing them
never produces a spurious ``Unnamed: 0`` data column. The row *order* of these
files is the only thing that associates a label (or a UMAP point) with its
embedding, so these tests guard both properties:

* written files import cleanly: no ``Unnamed`` columns and a plain
  ``RangeIndex`` starting at 0;
* row ``i`` of every metadata file still corresponds to embedding ``i``,
  i.e. a given audio file and time bin produce exactly the embedding, UMAP
  point and label that the user sees in the visualization.
"""

from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import pandas as pd
import pytest
import torch

from bacpipe import Loader
from bacpipe.core.workflows import generate_embeddings
from bacpipe.embedding_evaluation.label_embeddings import (
    create_metadata_labels,
    ground_truth_by_model,
    make_set_paths_func,
)
from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    collect_dim_reduced_embeds,
)
from bacpipe.model_pipelines.model_utils import ModelBaseClass
from bacpipe.model_pipelines.runner import Classifier

TEST_AUDIO_DIR = Path("bacpipe/tests/test_data")

# Name (key) under which the alignment test stores its results.
ALIGNMENT_MODEL = "mel"


class MelSpectrogramModel(ModelBaseClass):
    """Checkpoint-free feature extractor used by the end-to-end alignment
    test: mel-spectrograms are computed with librosa and treated as
    embeddings. No model checkpoint is required, so the test runs fast and
    works on a fresh checkout."""

    SAMPLE_RATE = 48_000
    SEGMENT_LENGTH = SAMPLE_RATE  # 1 second windows

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
        if audio.ndim == 1:
            audio = audio[None]
        mels = np.stack(
            [
                librosa.feature.melspectrogram(
                    y=segment,
                    sr=self.SAMPLE_RATE,
                    n_mels=64,
                    n_fft=1024,
                    hop_length=512,
                )
                for segment in audio
            ]
        )
        return torch.tensor(mels.reshape(len(mels), -1), dtype=torch.float32)


# Columns of a prediction table that are bookkeeping, not species.
_META_COLUMNS = {"audiofilename", "start", "end", "simultaneous_labels"}


def _make_paths(tmp_path):
    """Minimal result paths, mirroring the production directory layout."""
    paths = SimpleNamespace(
        audio_dir=tmp_path / "audio",
        main_embeds_path=tmp_path / "embeddings",
        labels_path=tmp_path / "labels",
        preds_path=tmp_path / "predictions",
    )
    paths.main_embeds_path.mkdir(exist_ok=True, parents=True)
    paths.labels_path.mkdir(exist_ok=True, parents=True)
    paths.preds_path.mkdir(exist_ok=True, parents=True)
    return paths


class TestMetadataLabelsRoundTrip:
    """``metadata_labels.csv`` / ``.parquet`` must round-trip without an
    ``Unnamed: 0`` column and must preserve the embedding row order."""

    def _fake_metadata(self):
        return {
            "files": {
                "audio_files": [
                    "audio/FewShot/CHE_01_20190101_163410.wav",
                    "audio/FewShot/CHE_02_20190101_183410.wav",
                ],
                "nr_embeds_per_file": [2, 3],
            },
            "nr_embeds_total": 5,
            "segment_length (samples)": 48000,
            "sample_rate (Hz)": 48000,
        }

    def test_csv_roundtrip_has_no_unnamed_and_preserves_order(
        self, tmp_path, monkeypatch
    ):
        import bacpipe.embedding_evaluation.label_embeddings as le

        paths = _make_paths(tmp_path)
        monkeypatch.setattr(
            le,
            "get_files_if_no_embeds",
            lambda *a, **k: ([], 1.0, self._fake_metadata()),
        )

        labels = create_metadata_labels(
            audio_dir=TEST_AUDIO_DIR,
            model="testmodel",
            paths=paths,
            overwrite=True,
            return_type="dataframe",
            default_label_keys=["audio_file_name"],
        )
        csv_path = paths.labels_path / "metadata_labels.csv"
        assert csv_path.exists()
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        # the first column is a real data column, not a written index
        assert header.split(",")[0] == "audio_file_name", header
        assert "Unnamed" not in header

        # read back through the exact production read path
        reloaded = create_metadata_labels(
            audio_dir=TEST_AUDIO_DIR,
            model="testmodel",
            paths=paths,
            overwrite=False,
            return_type="dataframe",
        )
        assert list(reloaded.columns) == list(labels.columns)
        assert not any("Unnamed" in str(c) for c in reloaded.columns)
        assert isinstance(reloaded.index, pd.RangeIndex)
        assert reloaded.index.tolist() == list(range(len(reloaded)))
        # exact row order is preserved: row i of the file == embedding i

    def test_parquet_reads_back_without_unnamed_and_preserves_order(
        self, tmp_path
    ):
        paths = _make_paths(tmp_path)
        labels = pd.DataFrame(
            {
                "audio_file_name": ["a.wav", "a.wav"],
                "start": [0.0, 1.0],
                "end": [1.0, 2.0],
            }
        )
        # production writes parquet with index=False (no stored index column)
        labels.to_parquet(
            paths.labels_path / "metadata_labels.parquet", index=False
        )

        reloaded = create_metadata_labels(
            audio_dir=TEST_AUDIO_DIR,
            model="testmodel",
            paths=paths,
            overwrite=False,
            return_type="dataframe",
        )
        assert not any("Unnamed" in str(c) for c in reloaded.columns)
        assert isinstance(reloaded.index, pd.RangeIndex)
        assert reloaded.index.tolist() == list(range(len(reloaded)))
        pd.testing.assert_frame_equal(reloaded, labels)


class TestGroundTruthRoundTrip:
    """``ground_truth_<label>.csv`` must round-trip without an ``Unnamed: 0``
    column and must keep exactly one row per embedding bin, keyed by
    ``(audiofilename, start)``."""

    def _make_paths(self, tmp_path):
        paths = _make_paths(tmp_path)
        # ``main_embeds_path`` is left empty on purpose so that
        # ``ground_truth_by_model`` falls back to ``get_files_if_no_embeds``.
        paths.main_embeds_path.rmdir()
        return paths

    def _annotations(self):
        return pd.DataFrame(
            {
                "audiofilename": [
                    "audio/FewShot/CHE_01_20190101_163410.wav",
                ] * 2,
                "start": [0.0, 3.0],
                "end": [2.0, 5.0],
                "label:species": ["sp_A", "sp_B"],
            }
        )

    def _fake_metadata(self):
        return {
            "files": {
                "audio_files": [
                    "audio/FewShot/CHE_01_20190101_163410.wav",
                ],
                "nr_embeds_per_file": [5],
            },
            "nr_embeds_total": 5,
            "segment_length (samples)": 48000,
            "sample_rate (Hz)": 48000,
        }

    def test_ground_truth_csv_roundtrip_has_no_unnamed(self, tmp_path, monkeypatch):
        import bacpipe.embedding_evaluation.label_embeddings as le

        paths = self._make_paths(tmp_path)
        files = [Path("CHE_01_20190101_163410_testmodel")]
        monkeypatch.setattr(
            le,
            "get_files_if_no_embeds",
            lambda *a, **k: (files, 1.0, self._fake_metadata()),
        )

        gt = ground_truth_by_model(
            model="testmodel",
            audio_dir=TEST_AUDIO_DIR,
            label_df=self._annotations(),
            paths=paths,
            overwrite=True,
            only_embed_annotations=False,
        )
        # one row per embedding bin
        assert len(gt) == 5
        assert isinstance(gt.index, pd.RangeIndex)
        assert gt.index.tolist() == list(range(5))

        csv_path = paths.labels_path / "ground_truth_species.csv"
        assert csv_path.exists()
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",")[0] == "start", header
        assert "Unnamed" not in header

        # the written file imports cleanly and is identical to what the
        # function returned (which is exactly what is stored on disk)
        from_file = pd.read_csv(csv_path, index_col=False)
        assert not any("Unnamed" in str(c) for c in from_file.columns)
        assert isinstance(from_file.index, pd.RangeIndex)
        pd.testing.assert_frame_equal(from_file, gt, check_dtype=False)

        # the (audiofilename, start) association maps annotations onto the
        # correct embedding bins: 0-2s -> bins 0,1 and 3-5s -> bins 3,4
        species_cols = [c for c in gt.columns if c not in _META_COLUMNS]
        assert set(species_cols) == {"sp_A", "sp_B"}
        assert gt.loc[gt["sp_A"] == 1, "start"].tolist() == [0.0, 1.0]
        assert gt.loc[gt["sp_B"] == 1, "start"].tolist() == [3.0, 4.0]
        middle = gt.loc[gt["start"] == 2.0].iloc[0]
        assert middle["sp_A"] == 0 and middle["sp_B"] == 0


class TestLoadExistingClfierOutputsLegacyColumnOrder:
    """Prediction files written by old bacpipe versions contain a leading
    index column (``Unnamed: 0``). Species columns must be selected by name so
    that this extra column cannot shift the association between a row's
    highest-probability species and the embedding it belongs to."""

    def _load_existing(self, preds_df, new_annots):
        classifier = object.__new__(Classifier)
        loader = SimpleNamespace(predictions=lambda return_type=None: preds_df)
        classifier._load_existing_clfier_outputs(loader, new_annots)
        return classifier.cumulative_annotations

    def test_old_file_with_unnamed_index_column(self):
        preds = pd.DataFrame(
            {
                "Unnamed: 0": [0, 1],
                "audiofilename": ["a.wav", "a.wav"],
                "start": [0.0, 1.0],
                "end": [1.0, 2.0],
                "simultaneous_labels": [0, 2],
                "sp_A": [0.8, 0.4],
                "sp_B": [0.2, 0.6],
            }
        )
        new_annots = pd.DataFrame(
            {
                "start": [2.0],
                "end": [3.0],
                "audiofilename": ["a.wav"],
                "label:default_classifier": ["sp_A"],
                "label:confidence": [0.9],
            }
        )
        cumulative = self._load_existing(preds, new_annots)

        # row 1 has simultaneous_labels=2 > sp_B=0.6, so a positional
        # selection (``iloc[:, 4:]``) would wrongly return
        # "simultaneous_labels" as the most probable species. The name-based
        # selection must return the actual species, keeping the row order.
        species = cumulative["label:default_classifier"].tolist()
        assert species[:2] == ["sp_A", "sp_B"]
        assert len(cumulative) == 3


class TestEmbeddingAlignment:
    """End-to-end check that the embedding array, the UMAP projection and
    the metadata labels are mutually aligned: point ``i`` of the scatter plot
    is the projection of embedding ``i`` and carries the metadata of row ``i``
    in ``metadata_labels``.

    Uses ``MelSpectrogramModel`` as a checkpoint-free stand-in for a real
    feature extractor (mel-spectrograms are treated as embeddings), so the
    test is fast and requires no model checkpoint on disk.
    """

    @pytest.mark.parametrize("annotations_only", [False, True])
    def test_embeddings_umap_and_metadata_labels_are_aligned(
        self, tmp_path, monkeypatch, annotations_only
    ):
        # ``ensure_models_exist`` only knows the built-in checkpoint models;
        # the mel model needs no checkpoint, so make the check a no-op.
        import bacpipe.core.workflows as workflows

        monkeypatch.setattr(
            workflows, "ensure_models_exist", lambda *a, **k: None
        )

        results_dir = tmp_path / "results"
        # Set the module-global ``get_paths`` before any pipeline code runs so
        # that the timestamps written into the UMAP JSON resolve against the
        # correct audio directory and results directory.
        make_set_paths_func(TEST_AUDIO_DIR, main_results_dir=results_dir)

        common = dict(
            audio_dir=TEST_AUDIO_DIR,
            main_results_dir=results_dir,
            device="cpu",
            run_pretrained_classifier=False,
            check_if_combination_exists=False,
            only_embed_annotations=annotations_only,
            CustomModel=MelSpectrogramModel,
        )
        loader = generate_embeddings(model_name=ALIGNMENT_MODEL, **common)
        loader_umap = generate_embeddings(
            model_name=ALIGNMENT_MODEL, dim_reduction_model="umap", **common
        )

        embeds = loader.embeddings(return_type="array")
        assert embeds.ndim == 2

        # the dim-reduction loader must still find the actual .npy files:
        # ``embeddings()`` returns the model embeddings, not the reduced
        # coordinates stored in the separate .json files
        embeds_after_umap = loader_umap.embeddings(return_type="array")
        assert embeds_after_umap.shape == embeds.shape
        np.testing.assert_allclose(embeds_after_umap, embeds)

        # the scatter plot is fed from the very JSON the visualization reads
        # (``collect_dim_reduced_embeds``): point i of the plot is the UMAP
        # projection of embedding i
        plot_points = collect_dim_reduced_embeds(
            ALIGNMENT_MODEL, loader_umap.embed_dir, "umap"
        )
        assert len(plot_points["x"]) == len(plot_points["y"]) == embeds.shape[0]
        assert len(plot_points["timestamp"]) == embeds.shape[0]

        # both loaders describe the same files, in the same order, with the
        # same number of embedding bins per file -> UMAP row i belongs to
        # embedding row i.
        for key in ("audio_files", "nr_embeds_per_file"):
            assert (
                loader_umap.metadata_dict["files"][key]
                == loader.metadata_dict["files"][key]
            )

        get_paths = make_set_paths_func(
            TEST_AUDIO_DIR, main_results_dir=results_dir
        )
        paths = get_paths(ALIGNMENT_MODEL)
        labels = create_metadata_labels(
            audio_dir=TEST_AUDIO_DIR,
            model=ALIGNMENT_MODEL,
            paths=paths,
            overwrite=True,
            return_type="dataframe",
            default_label_keys=["audio_file_name"],
            only_embed_annotations=annotations_only,
        )

        # exactly one label row per embedding and per UMAP point
        assert len(labels) == embeds.shape[0] == len(plot_points["x"])

        # the file that was just written (index=False) imports cleanly
        reloaded = create_metadata_labels(
            audio_dir=TEST_AUDIO_DIR,
            model=ALIGNMENT_MODEL,
            paths=paths,
            overwrite=False,
            return_type="dataframe",
        )
        assert not any("Unnamed" in str(c) for c in reloaded.columns)
        assert isinstance(reloaded.index, pd.RangeIndex)
        assert reloaded.index.tolist() == list(range(len(reloaded)))
        pd.testing.assert_frame_equal(reloaded, labels)

        # row i carries the metadata of embedding i: the file name and time
        # bin are identical to the loader's per-file bookkeeping
        metadata = loader.metadata_dict
        segment_s = (
            metadata["segment_length (samples)"] / metadata["sample_rate (Hz)"]
        )
        expected_files = [
            str(f)
            for f, nr_embeds in zip(
                metadata["files"]["audio_files"],
                metadata["files"]["nr_embeds_per_file"],
            )
            for _ in range(nr_embeds)
        ]
        assert labels["audio_file_name"].tolist() == expected_files

        if annotations_only:
            # one embedding per annotated segment: the start of row i is the
            # start of the annotation that produced embedding i, per file in
            # the loader's file order
            annots = pd.read_csv(TEST_AUDIO_DIR / "annotations.csv")
            expected_starts = []
            for file, _nr_embeds in zip(
                metadata["files"]["audio_files"],
                metadata["files"]["nr_embeds_per_file"],
            ):
                file_annots = Loader.filter_df_by_file(
                    TEST_AUDIO_DIR, annots, Path(TEST_AUDIO_DIR) / file
                )
                expected_starts.extend(file_annots.start.unique().tolist())
            assert labels["start"].tolist() == [
                round(s, 4) for s in expected_starts
            ]
        else:
            expected_starts = [
                round(bin_idx * segment_s, 6)
                for nr_embeds in metadata["files"]["nr_embeds_per_file"]
                for bin_idx in range(nr_embeds)
            ]
            assert labels["start"].tolist() == expected_starts
