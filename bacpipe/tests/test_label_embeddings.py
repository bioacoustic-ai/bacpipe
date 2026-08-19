"""
Unit tests for the label-embedding helpers in
``bacpipe.embedding_evaluation.label_embeddings``.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from types import SimpleNamespace

from bacpipe.embedding_evaluation.label_embeddings import (
    assign_global_get_paths_function,
    create_metadata_labels,
    ensure_windoof_path_to_posix,
    fetch_annotation_file,
    filter_annotations,
    filter_df_by_filename,
    get_dt_filename,
    get_ground_truth,
    load_metadata_file,
    make_set_paths_func,
    model_specific_embedding_path,
)


class TestMakeSetPathsFunc:
    def test_creates_directory_structure(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        get_paths = make_set_paths_func(
            audio_dir, main_results_dir=tmp_path / "results"
        )
        paths = get_paths("testmodel")
        for p in [
            paths.main_embeds_path,
            paths.labels_path,
            paths.clust_path,
            paths.probe_path,
            paths.plot_path,
        ]:
            assert p.exists()

    def test_assign_global_get_paths_function(self, tmp_path, monkeypatch):
        import bacpipe.embedding_evaluation.label_embeddings as le

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        # make sure the module-global ``get_paths`` is absent so the
        # function actually assigns it
        monkeypatch.delattr(le, "get_paths", raising=False)
        assign_global_get_paths_function(audio_dir)
        get_paths = le.get_paths
        assert callable(get_paths)
        assert get_paths("some_model").audio_dir == audio_dir

    def test_evaluations_dir_kwarg_overrides_settings(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        get_paths = make_set_paths_func(
            audio_dir,
            main_results_dir=tmp_path / "results",
            evaluations_dir="custom_evaluations",
        )
        paths = get_paths("testmodel")
        assert "custom_evaluations" in str(paths.labels_path)
        assert paths.labels_path.parent.parent.name == "custom_evaluations"

    def test_assign_global_get_paths_function_kwargs(self, tmp_path, monkeypatch):
        import bacpipe.embedding_evaluation.label_embeddings as le

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        monkeypatch.delattr(le, "get_paths", raising=False)
        assign_global_get_paths_function(
            audio_dir,
            main_results_dir=tmp_path / "results",
            evaluations_dir="custom_evaluations",
        )
        paths = le.get_paths("some_model")
        assert "custom_evaluations" in str(paths.labels_path)


class TestModelSpecificEmbeddingPath:
    def _make_embed_dirs(self, tmp_path):
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        (embed_dir / "2024-01-01_00-00___testmodel-birdset").mkdir()
        (embed_dir / "2024-01-02_00-00___testmodel-birdset").mkdir()
        return embed_dir

    def test_returns_most_recent_matching_dir(self, tmp_path):
        embed_dir = self._make_embed_dirs(tmp_path)
        result = model_specific_embedding_path(embed_dir, "testmodel")
        assert result.name == "2024-01-02_00-00___testmodel-birdset"

    def test_raises_when_no_embeddings_found(self, tmp_path):
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        with pytest.raises(ValueError):
            model_specific_embedding_path(embed_dir, "nonexistent")

    def test_filters_by_dim_reduction_model(self, tmp_path):
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        sub = embed_dir / "2024-01-01_00-00___testmodel-birdset-umap"
        sub.mkdir()
        with open(sub / "embedded_data.json", "w") as f:
            f.write('{"x": [1, 2], "y": [3, 4]}')
        result = model_specific_embedding_path(
            embed_dir, "testmodel", dim_reduction_model="umap"
        )
        assert result == sub

    def test_dim_reduction_filter_skips_mismatching_dirs(self, tmp_path):
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        # matches the model but not the dim reduction model in the stem
        sub = embed_dir / "2024-01-01_00-00___testmodel-birdset"
        sub.mkdir()
        with open(sub / "embedded_data.json", "w") as f:
            f.write('{"x": [1, 2], "y": [3, 4]}')
        with pytest.raises(ValueError):
            model_specific_embedding_path(
                embed_dir, "testmodel", dim_reduction_model="tsne"
            )


class TestGetDtFilename:
    def test_standard_birdnet_format(self):
        assert get_dt_filename("CHE_01_20190101_163410.wav") == dt.datetime(
            2019, 1, 1, 16, 34, 10
        )

    def test_compact_underscored_format(self):
        assert get_dt_filename("rec_20210708_080000.wav") == dt.datetime(
            2021, 7, 8, 8, 0, 0
        )

    def test_timezone_suffix_is_ignored(self):
        assert get_dt_filename("rec_20210708_080000+0200.wav") == dt.datetime(
            2021, 7, 8, 8, 0, 0
        )

    def test_falls_back_to_default(self):
        assert get_dt_filename("myrecording.wav") == dt.datetime(
            2000, 10, 10, 0, 0, 0
        )


class TestEnsureWindoofPathToPosix:
    def test_converts_windows_separators(self):
        assert (
            ensure_windoof_path_to_posix("C:\\\\audio\\\\file.wav")
            == "C:/audio/file.wav"
        )

    def test_leaves_posix_path_unchanged(self):
        assert ensure_windoof_path_to_posix("/audio/file.wav") == (
            "/audio/file.wav"
        )


class TestLoadMetadataFile:
    def _write_metadata(self, folder, audio_files, embed_files):
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {
            "audio_dir": "/audio",
            "embed_dir": "/embeds",
            "files": {
                "audio_files": audio_files,
                "embedding_files": embed_files,
            },
        }
        with open(folder / "metadata.yml", "w") as f:
            yaml.dump(metadata, f)

    def test_loads_and_normalizes_paths(self, tmp_path):
        self._write_metadata(
            tmp_path, ["a.wav", "b.wav"], ["a.npy", "b.npy"]
        )
        metadata = load_metadata_file(tmp_path)
        assert metadata["audio_dir"] == "/audio"
        assert len(metadata["files"]["audio_files"]) == 2

    def test_empty_audio_files_raises(self, tmp_path):
        self._write_metadata(tmp_path, [], [])
        with pytest.raises(AssertionError):
            load_metadata_file(tmp_path)


class TestFilterAnnotations:
    def _df(self):
        return pd.DataFrame(
            {
                "audiofilename": ["a.wav", "b.wav", "c.wav"],
                "species": ["tree", "tree", "kestrel"],
            }
        )

    def test_filters_to_classes_with_minimum_occurrences(self):
        annots = self._df()
        filtered = filter_annotations(
            annots, "species", min_label_occurrences=1, bool_filter_labels=True
        )
        # "tree" occurs twice (2 > 1), "kestrel" only once (1 > 1 is False)
        assert len(filtered) == 2
        assert set(filtered.species) == {"tree"}

    def test_no_labels_left_returns_none(self):
        annots = self._df()
        assert (
            filter_annotations(
                annots,
                "species",
                min_label_occurrences=2,
                bool_filter_labels=True,
            )
            is None
        )


class TestFetchAnnotationFile:
    def test_loads_from_annotations_dir(self, tmp_path):
        annots_dir = tmp_path / "annots"
        annots_dir.mkdir()
        csv_path = annots_dir / "annotations.csv"
        pd.DataFrame({"species": ["a"]}).to_csv(csv_path, index=False)
        paths = SimpleNamespace(dataset_path=tmp_path / "dataset")
        df = fetch_annotation_file(annots_dir, "annotations.csv", paths)
        assert list(df.columns) == ["species"]

    def test_falls_back_to_dataset_path(self, tmp_path):
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        pd.DataFrame({"species": ["b"]}).to_csv(
            dataset_path / "annotations.csv", index=False
        )
        paths = SimpleNamespace(dataset_path=dataset_path)
        audio_dir = tmp_path / "empty_audio"
        df = fetch_annotation_file(audio_dir, "annotations.csv", paths)
        assert list(df.columns) == ["species"]

    def test_no_file_anywhere_raises(self, tmp_path):
        paths = SimpleNamespace(dataset_path=tmp_path / "missing_dataset")
        with pytest.raises(FileNotFoundError):
            fetch_annotation_file(
                tmp_path / "empty_audio", "annotations.csv", paths
            )


class TestFilterDfByFilename:
    def test_filters_rows(self):
        annots = pd.DataFrame(
            {
                "audiofilename": ["a.wav", "b.wav", "a.wav"],
                "start": [0, 1, 2],
            }
        )
        filtered = filter_df_by_filename(annots, "a.wav")
        assert len(filtered) == 2
        assert (filtered.audiofilename == "a.wav").all()


class TestGetGroundTruth:
    def test_dataframe_from_file(self, tmp_path):
        csv_path = tmp_path / "gt.csv"
        pd.DataFrame({"species": ["a", "b"]}).to_csv(csv_path, index=False)
        df = get_ground_truth("ignored", file_path=csv_path)
        assert len(df) == 2

    def test_array_from_labels_path(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        get_paths = make_set_paths_func(
            audio_dir, main_results_dir=tmp_path / "results"
        )
        labels_path = get_paths("testmodel").labels_path
        gt = {"a": np.array([1, 0]), "b": np.array([0, 1])}
        np.save(labels_path / "ground_truth.npy", gt, allow_pickle=True)
        loaded = get_ground_truth("testmodel", return_type="array")
        assert set(loaded.keys()) == {"a", "b"}

    def test_none_file_returns_none(self):
        assert get_ground_truth("ignored") is None


class TestCreateMetadataLabels:
    """Tests for ``create_metadata_labels``, which powers the default labels
    used in the ``simple_use_cases`` notebook (cell 15) and by the clustering
    pipeline."""

    TEST_AUDIO_DIR = "bacpipe/tests/test_data"

    def _make_paths(self, tmp_path):
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

    def _fake_metadata(self, audio_file="audio/FewShot/CHE_01_20190101_163410.wav"):
        return {
            "files": {
                "audio_files": [audio_file],
                "nr_embeds_per_file": [2],
            },
            "nr_embeds_total": 2,
            "segment_length (samples)": 48000,
            "sample_rate (Hz)": 48000,
        }

    def test_generates_and_saves_labels(self, tmp_path, monkeypatch):
        import bacpipe.embedding_evaluation.label_embeddings as le

        paths = self._make_paths(tmp_path)
        monkeypatch.setattr(
            le, "get_files_if_no_embeds", lambda *a, **k: ([], 1.0, self._fake_metadata())
        )

        dl = create_metadata_labels(
            audio_dir=self.TEST_AUDIO_DIR,
            model="testmodel",
            paths=paths,
            overwrite=True,
            return_type="dataframe",
            default_label_keys=["time_of_day", "audio_file_name"],
        )
        # two embeddings for the single audio file
        assert len(dl) == 2
        assert "time_of_day" in dl.columns
        assert "audio_file_name" in dl.columns
        assert dl["audio_file_name"].tolist() == [
            "audio/FewShot/CHE_01_20190101_163410.wav",
        ] * 2
        # the per-embedding start/end grid is derived from the segment length
        assert dl["start"].tolist() == [0.0, 1.0]
        assert dl["end"].tolist() == [1.0, 2.0]
        assert (paths.labels_path / "metadata_labels.csv").exists()

    def test_returns_dict_for_return_type_dict(self, tmp_path, monkeypatch):
        import bacpipe.embedding_evaluation.label_embeddings as le

        paths = self._make_paths(tmp_path)
        monkeypatch.setattr(
            le, "get_files_if_no_embeds", lambda *a, **k: ([], 1.0, self._fake_metadata())
        )
        labels = create_metadata_labels(
            audio_dir=self.TEST_AUDIO_DIR,
            model="testmodel",
            paths=paths,
            overwrite=True,
            return_type="dict",
            default_label_keys=["audio_file_name"],
        )
        assert isinstance(labels, dict)
        assert len(labels["audio_file_name"]) == 2

    def test_loads_existing_labels_without_regenerating(self, tmp_path):
        paths = self._make_paths(tmp_path)
        pd.DataFrame(
            {
                "time_of_day": ["12-00-00", "12-00-01"],
                "audio_file_name": ["a.wav", "a.wav"],
            }
        ).to_csv(paths.labels_path / "metadata_labels.csv", index=False)

        dl = create_metadata_labels(
            audio_dir=self.TEST_AUDIO_DIR,
            model="testmodel",
            paths=paths,
            overwrite=False,
            return_type="dataframe",
        )
        assert list(dl.columns) == ["time_of_day", "audio_file_name"]
        assert len(dl) == 2

    def test_kwarg_only_embed_annotations_overrides_settings(
        self, tmp_path, monkeypatch
    ):
        import bacpipe
        import bacpipe.embedding_evaluation.label_embeddings as le

        paths = self._make_paths(tmp_path)
        monkeypatch.setattr(
            le, "get_files_if_no_embeds", lambda *a, **k: ([], 1.0, self._fake_metadata())
        )
        # settings say "annotations only"...
        monkeypatch.setattr(bacpipe.settings, "only_embed_annotations", True)
        # ...but an explicitly passed kwarg must win: a regular time grid is
        # generated instead of annotation based labels
        dl = create_metadata_labels(
            audio_dir=self.TEST_AUDIO_DIR,
            model="testmodel",
            paths=paths,
            overwrite=True,
            return_type="dataframe",
            default_label_keys=["time_of_day", "audio_file_name"],
            only_embed_annotations=False,
        )
        assert dl["start"].tolist() == [0.0, 1.0]
        assert dl["end"].tolist() == [1.0, 2.0]

class TestDeduplicateAnnotationPairs:
    """Row-level deduplication of the annotations. Rows are only dropped
    when they are exact duplicates: same ``(start, end)`` pair, same source
    file and same labels. Several species vocalizing in the same window, or
    the same pair in different files, are never collapsed - each of those
    rows is a distinct annotation."""

    def _df(self):
        return pd.DataFrame(
            {
                "audiofilename": ["a.wav"] * 5,
                "start": [0, 0, 0, 5, 5],
                "end": [5, 5, 10, 10, 10],
                "label:species": ["sp_A", "sp_B", "sp_C", "sp_D", "sp_E"],
            }
        )

    def test_keeps_different_species_sharing_a_pair(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            deduplicate_annotation_pairs,
        )

        deduped = deduplicate_annotation_pairs(self._df())
        # (0,5) labels sp_A and sp_B, (5,10) labels sp_D and sp_E: several
        # species can vocalize at the exact same time, so no row is dropped.
        assert len(deduped) == 5
        assert deduped["start"].tolist() == [0, 0, 0, 5, 5]
        assert deduped["end"].tolist() == [5, 5, 10, 10, 10]
        assert deduped["label:species"].tolist() == [
            "sp_A", "sp_B", "sp_C", "sp_D", "sp_E",
        ]

    def test_drops_only_exact_duplicate_rows(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            deduplicate_annotation_pairs,
        )

        df = pd.DataFrame(
            {
                "audiofilename": ["a.wav"] * 3,
                "start": [0, 0, 5],
                "end": [5, 5, 10],
                "label:species": ["sp_A", "sp_A", "sp_B"],
            }
        )
        deduped = deduplicate_annotation_pairs(df)
        # only the twice-recorded (0,5) sp_A row is an exact duplicate
        assert len(deduped) == 2
        assert deduped["start"].tolist() == [0, 5]
        assert deduped["label:species"].tolist() == ["sp_A", "sp_B"]

    def test_keeps_same_pair_from_different_files(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            deduplicate_annotation_pairs,
        )

        df = pd.DataFrame(
            {
                "audiofilename": ["a.wav", "b.wav"],
                "start": [0, 0],
                "end": [5, 5],
                "label:species": ["sp_A", "sp_A"],
            }
        )
        deduped = deduplicate_annotation_pairs(df)
        # same pair AND same species, but different source files
        assert len(deduped) == 2

    def test_keeps_shared_start_different_end_pairs(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            deduplicate_annotation_pairs,
        )

        df = pd.DataFrame(
            {
                "start": [0, 0, 5],
                "end": [5, 10, 10],
                "label:species": ["a", "b", "c"],
            }
        )
        deduped = deduplicate_annotation_pairs(df)
        assert len(deduped) == 3

    def test_unique_pairs_collapses_species_sharing_a_window(self):
        # The distinct one-row-per-embedded-segment operation that mirrors
        # the audio loader: the count is per unique pair, the species rows
        # themselves are handled separately by the ground truth path.
        from bacpipe.embedding_evaluation.label_embeddings import (
            unique_start_end_annot_pairs,
        )

        deduped = unique_start_end_annot_pairs(self._df())
        assert deduped["start"].tolist() == [0, 0, 5]
        assert deduped["end"].tolist() == [5, 10, 10]
        # first occurrence of each pair wins, input order preserved
        assert deduped["label:species"].tolist() == ["sp_A", "sp_C", "sp_D"]



class TestFitLabelsToEmbeddingTimestamps:
    """In annotated-segment mode the ground truth must hold exactly one row
    per unique ``(start, end)`` pair (i.e. per embedded segment), even when
    the annotations contain duplicate/shared pairs. Rows of different
    species sharing a pair must all survive and mark the shared segment
    (multi-label ground truth)."""

    def _fitted_frame(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            initialize_ground_truth_df,
        )

        return initialize_ground_truth_df(
            pd.DataFrame(
                {"label:species": ["sp_A", "sp_B", "sp_C"]}
            ),
            "species",
        )

    def _annotations(self):
        # (0,5) occurs twice, (5,10) occurs twice; (0,10) shares its start
        return pd.DataFrame(
            {
                "audiofilename": ["a.wav"] * 5,
                "start": [0, 0, 0, 5, 5],
                "end": [5, 5, 10, 10, 10],
                "label:species": ["sp_A", "sp_B", "sp_A", "sp_B", "sp_C"],
            }
        )

    def test_only_embed_annotations_dedupes_pairs(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            fit_labels_to_embedding_timestamps,
        )

        fitted = fit_labels_to_embedding_timestamps(
            self._annotations(),
            self._fitted_frame(),
            num_embeds=5,  # stale metadata count, must be overridden to 3
            segment_s=5.0,
            label_column="species",
            only_embed_annotations=True,
            min_annotation_length=0,
        )
        assert len(fitted) == 3
        assert fitted["start"].tolist() == [0, 0, 5]
        assert fitted["end"].tolist() == [5, 10, 10]

    def test_multi_label_species_sharing_a_window_are_all_kept(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            fit_labels_to_embedding_timestamps,
        )

        df = pd.DataFrame(
            {
                "audiofilename": ["a.wav"] * 4,
                "start": [0, 0, 5, 5],
                "end": [5, 5, 10, 10],
                "label:species": ["sp_A", "sp_B", "sp_B", "sp_C"],
            }
        )
        fitted = fit_labels_to_embedding_timestamps(
            df,
            self._fitted_frame(),
            num_embeds=99,  # stale metadata count, must be overridden to 2
            segment_s=5.0,
            label_column="species",
            only_embed_annotations=True,
            min_annotation_length=0,
        )
        # one row per embedded segment (unique pair)
        assert len(fitted) == 2
        assert fitted["start"].tolist() == [0, 5]
        assert fitted["end"].tolist() == [5, 10]
        # every species vocalizing in a window is marked on its segment:
        # (0,5) holds sp_A + sp_B, (5,10) holds sp_B + sp_C
        assert fitted["sp_A"].tolist() == [1, 0]
        assert fitted["sp_B"].tolist() == [1, 1]
        assert fitted["sp_C"].tolist() == [0, 1]
        assert fitted["simultaneous_labels"].tolist() == [2, 2]

    def test_exact_duplicate_rows_do_not_double_mark(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            fit_labels_to_embedding_timestamps,
        )

        # two identical rows (same pair, same species): one annotation, not
        # two, so the segment is marked once
        df = pd.DataFrame(
            {
                "audiofilename": ["a.wav"] * 2,
                "start": [0, 0],
                "end": [5, 5],
                "label:species": ["sp_A", "sp_A"],
            }
        )
        fitted = fit_labels_to_embedding_timestamps(
            df,
            self._fitted_frame(),
            num_embeds=99,
            segment_s=5.0,
            label_column="species",
            only_embed_annotations=True,
            min_annotation_length=0,
        )
        assert len(fitted) == 1
        assert fitted["sp_A"].tolist() == [1]
        assert fitted["sp_B"].tolist() == [0]
        assert fitted["simultaneous_labels"].tolist() == [1]

    def test_full_mode_uses_grid_ignoring_annotation_count(self):
        from bacpipe.embedding_evaluation.label_embeddings import (
            fit_labels_to_embedding_timestamps,
        )

        fitted = fit_labels_to_embedding_timestamps(
            self._annotations(),
            self._fitted_frame(),
            num_embeds=5,
            segment_s=1.0,
            label_column="species",
            only_embed_annotations=False,
            min_annotation_length=0,
        )
        # full-grid mode keeps one row per embedding bin, not per annotation
        assert len(fitted) == 5
        assert fitted["start"].tolist() == [0, 1, 2, 3, 4]


class TestGetFilesIfNoEmbeds:
    """``get_files_if_no_embeds`` must count the embeddings a run will
    create. In annotated-segment mode that is the number of *unique*
    ``(start, end)`` pairs, matching ``_only_load_annotated_segments``."""

    def _monkeypatch_deps(self, monkeypatch, le, bacpipe, duration=4.0):
        fake_module = SimpleNamespace(
            LENGTH_IN_SAMPLES=48000, SAMPLE_RATE=48000
        )
        monkeypatch.setattr(
            bacpipe, "get_audio_files", lambda audio_dir: [Path("a.wav")]
        )
        monkeypatch.setattr(
            le, "ensure_audio_files",
            lambda found, annotated, audio_dir: found,
        )
        monkeypatch.setattr(le, "import_module", lambda name: fake_module)
        monkeypatch.setattr(le, "get_duration", lambda **kwargs: duration)

    def test_only_embed_annotations_counts_unique_pairs(
        self, tmp_path, monkeypatch
    ):
        import bacpipe
        import bacpipe.embedding_evaluation.label_embeddings as le

        self._monkeypatch_deps(monkeypatch, le, bacpipe)
        label_df = pd.DataFrame(
            {
                "audiofilename": ["a.wav"] * 4,
                "start": [0, 0, 5, 10],
                "end": [5, 5, 10, 15],
                "label:species": ["s1", "s2", "s3", "s4"],
            }
        )

        files, segment_s, metadata = le.get_files_if_no_embeds(
            tmp_path, "testmodel", label_df=label_df,
            only_embed_annotations=True,
        )
        # (0, 5) is duplicated -> 3 unique pairs, not 4 rows
        assert metadata["files"]["nr_embeds_per_file"] == [3]
        assert segment_s == 1.0
        assert files == [Path("a_testmodel")]

    def test_full_mode_uses_duration_grid(self, tmp_path, monkeypatch):
        import bacpipe
        import bacpipe.embedding_evaluation.label_embeddings as le

        self._monkeypatch_deps(monkeypatch, le, bacpipe, duration=4.0)
        label_df = pd.DataFrame(
            {
                "audiofilename": ["a.wav"] * 3,
                "start": [0, 0, 5],
                "end": [5, 5, 10],
                "label:species": ["s1", "s2", "s3"],
            }
        )

        files, segment_s, metadata = le.get_files_if_no_embeds(
            tmp_path, "testmodel", label_df=label_df,
            only_embed_annotations=False,
        )
        # 4 s / 1 s segment length -> 4 embeddings regardless of annotations
        assert metadata["files"]["nr_embeds_per_file"] == [4]

class TestDefaultLabelsOnlyEmbedAnnotations:
    """The per-embedding metadata labels (``time_of_day``,
    ``continuous_timestamp``, ``default_classifier``) index into the
    per-file annotation ``starts`` once per embedding. In annotated-segment
    mode they must use the deduplicated ``(start, end)`` pairs so the
    indexing cannot run past the end of the ``starts`` array when the
    annotations contain duplicate/shared pairs."""

    def _make_default_labels(self, tmp_path, nr_embeds=3):
        from bacpipe.embedding_evaluation.label_embeddings import DefaultLabels

        dl = object.__new__(DefaultLabels)
        dl.only_embed_annotations = True
        dl.paths = SimpleNamespace(audio_dir=tmp_path)
        audio_file = "CHE_01_20190101_163410.wav"
        dl.metadata = {
            "files": {"audio_files": [audio_file]},
            "segment_length (samples)": 48000,
            "sample_rate (Hz)": 48000,
        }
        dl.nr_embeds_per_file = [nr_embeds]
        dl.df = pd.DataFrame(
            {
                "audiofilename": [audio_file] * 4,
                # (0, 5) is a duplicate pair -> 3 unique pairs total
                "start": [0, 0, 5, 10],
                "end": [5, 5, 10, 15],
                "label:species": ["s1", "s2", "s3", "s4"],
            }
        )
        return dl

    def test_time_of_day_indexes_deduplicated_starts(self, tmp_path):
        dl = self._make_default_labels(tmp_path)
        dl.time_of_day()
        # one label per embedded segment (unique pair), starting at 16:34:10
        assert dl.time_of_day_per_embedding == [
            "16-34-10",  # start 0
            "16-34-15",  # start 5
            "16-34-20",  # start 10
        ]

    def test_continuous_timestamp_indexes_deduplicated_starts(self, tmp_path):
        dl = self._make_default_labels(tmp_path)
        dl.continuous_timestamp()
        assert dl.continuous_timestamp_per_embedding == [
            "1900-01-01_16:34:10",  # start 0
            "1900-01-01_16:34:15",  # start 5
            "1900-01-01_16:34:20",  # start 10
        ]

