"""
Unit tests for the experiment manager helpers in
``bacpipe.core.experiment_manager``.
"""

import json

import numpy as np
import pytest
import yaml

from bacpipe import Loader
from bacpipe.core.experiment_manager import (
    replace_default_kwargs_with_user_kwargs,
    return_reduced_dimensions,
    save_logs,
)

TEST_DATA_DIR = "bacpipe/tests/test_data"
TEST_AUDIO_DIR = "bacpipe/tests/test_data/audio"


class TestReplaceDefaultKwargsWithUserKwargs:
    def test_merges_user_kwargs(self):
        merged = replace_default_kwargs_with_user_kwargs(
            device="cpu", custom_option=True
        )
        assert merged["device"] == "cpu"
        assert merged["custom_option"] is True

    def test_removes_specified_keys(self):
        merged = replace_default_kwargs_with_user_kwargs(
            remove_keys=["device"]
        )
        assert "device" not in merged
        assert "custom_option" not in merged


class TestReturnReducedDimensions:
    def test_reads_two_dimensions(self, tmp_path):
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        with open(embed_dir / "embedded_data.json", "w") as f:
            json.dump({"x": [1, 2], "y": [3, 4]}, f)
        assert return_reduced_dimensions(embed_dir) == 2

    def test_reads_three_dimensions(self, tmp_path):
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        with open(embed_dir / "embedded_data.json", "w") as f:
            json.dump({"x": [1, 2], "y": [3, 4], "z": [5, 6]}, f)
        assert return_reduced_dimensions(embed_dir) == 3

    def test_empty_directory_defaults_to_two(self, tmp_path):
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        # without a json file the number of dimensions cannot be read,
        # so the default of 2 is returned
        assert return_reduced_dimensions(embed_dir) == 2


class TestLoader:
    def test_get_audio_files_path_objects(self):
        from pathlib import Path

        files = Loader.get_audio_files(TEST_AUDIO_DIR)
        assert len(files) > 0
        assert all(isinstance(f, Path) for f in files)

    def test_get_audio_files_strings(self):
        files = Loader.get_audio_files(TEST_AUDIO_DIR, return_type="str")
        assert len(files) > 0
        assert all(isinstance(f, str) for f in files)

    def test_init_without_checking_existing(self, tmp_path):
        loader = Loader(
            audio_dir=TEST_AUDIO_DIR,
            model_name="testmodel",
            check_if_combination_exists=False,
            testing=True,
            use_folder_structure=False,
            embed_parent_dir=tmp_path / "embeds",
        )
        assert loader.combination_already_exists is False
        assert len(loader.files) > 0
        assert "model_name" in loader.metadata_dict

    def test_filter_df_by_file(self):
        import pandas as pd

        annots = pd.DataFrame(
            {
                "audiofilename": [
                    "audio/FewShot/CHE_01_20190101_163410.wav",
                    "audio/SomeOther/file.wav",
                ],
                "start": [0, 5],
            }
        )
        from pathlib import Path

        file_path = Path(TEST_DATA_DIR) / "audio/FewShot" / (
            "CHE_01_20190101_163410.wav"
        )
        filtered = Loader.filter_df_by_file(
            TEST_DATA_DIR, annots, file_path
        )
        assert len(filtered) == 1
        assert filtered.iloc[0]["start"] == 0


class TestLoaderEmbeddings:
    """Tests for ``Loader.embeddings()``, used in the example notebooks to
    access already computed embeddings (both as a dict and concatenated
    array)."""

    def _make_loader(self, tmp_path):
        return Loader(
            audio_dir=TEST_AUDIO_DIR,
            model_name="testmodel",
            check_if_combination_exists=False,
            testing=True,
            use_folder_structure=False,
            embed_parent_dir=tmp_path / "embeds",
        )

    def _write_npy_files(self, tmp_path):
        embed_dir = tmp_path / "embeds"
        embed_dir.mkdir(parents=True, exist_ok=True)
        np.save(embed_dir / "file1.npy", np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.save(embed_dir / "file2.npy", np.array([[5.0, 6.0]]))
        return embed_dir

    def test_embeddings_array_concatenates_all_files(self, tmp_path):
        embed_dir = self._write_npy_files(tmp_path)
        loader = self._make_loader(tmp_path)
        loader.embed_dir = embed_dir
        loader.embed_suffix = ".npy"
        loader.dim_reduction_model = False
        loader.files = sorted(embed_dir.glob("*.npy"))

        embeds = loader.embeddings(return_type="array")
        assert embeds.shape == (3, 2)
        np.testing.assert_allclose(embeds[0], [1.0, 2.0])
        np.testing.assert_allclose(embeds[2], [5.0, 6.0])

    def test_embeddings_dict_keys_are_relative_paths(self, tmp_path):
        embed_dir = self._write_npy_files(tmp_path)
        loader = self._make_loader(tmp_path)
        loader.embed_dir = embed_dir
        loader.embed_suffix = ".npy"
        loader.dim_reduction_model = False
        loader.files = sorted(embed_dir.glob("*.npy"))

        d = loader.embeddings(return_type="dict")
        assert set(d.keys()) == {"file1.npy", "file2.npy"}
        np.testing.assert_allclose(d["file2.npy"], [[5.0, 6.0]])

    def test_embeddings_searches_embed_dir_when_files_are_audio(self, tmp_path):
        from pathlib import Path

        embed_dir = self._write_npy_files(tmp_path)
        loader = self._make_loader(tmp_path)
        loader.embed_dir = embed_dir
        loader.embed_suffix = ".npy"
        loader.dim_reduction_model = False
        # the loader still points at the audio files -> the method must
        # discover the npy files inside embed_dir on its own
        audio_file = Path(TEST_AUDIO_DIR) / "FewShot/CHE_01_20190101_163410.wav"
        loader.files = [audio_file]

        embeds = loader.embeddings(return_type="array")
        assert embeds.shape == (3, 2)

    def test_no_embedding_files_returns_none(self, tmp_path):
        from pathlib import Path

        loader = self._make_loader(tmp_path)
        empty_embed_dir = tmp_path / "empty"
        empty_embed_dir.mkdir()
        loader.embed_dir = empty_embed_dir
        loader.embed_suffix = ".npy"
        loader.dim_reduction_model = False
        audio_file = Path(TEST_AUDIO_DIR) / "FewShot/CHE_01_20190101_163410.wav"
        loader.files = [audio_file]

        assert loader.embeddings() is None

    def test_embeddings_for_dim_reduction_loader_finds_primary_npy(self, tmp_path):
        # Simulate the state of a loader right after a dimensionality
        # reduction run: ``embed_dir`` points at the reduced-output directory
        # (containing .json files) while the primary embeddings live in the
        # directory recorded in the metadata. ``embeddings()`` returns the
        # actual .npy embedding files in that case.
        primary_dir = tmp_path / "embeds"
        primary_dir.mkdir(parents=True, exist_ok=True)
        np.save(primary_dir / "file1.npy", np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.save(primary_dir / "file2.npy", np.array([[5.0, 6.0]]))

        umap_dir = tmp_path / "umap"
        umap_dir.mkdir(parents=True, exist_ok=True)
        (umap_dir / "file1.json").write_text(
            '{"x": [1.0, 2.0], "y": [3.0, 4.0]}'
        )

        loader = self._make_loader(tmp_path)
        loader.embed_dir = umap_dir
        loader.embed_suffix = ".npy"  # stale, left over from get_embedding_dir
        loader.dim_reduction_model = "umap"
        loader.files = sorted(umap_dir.glob("*.json"))
        loader.metadata_dict = {"embed_dir": str(primary_dir)}

        embeds = loader.embeddings(return_type="array")
        assert embeds.shape == (3, 2)
        np.testing.assert_allclose(embeds[0], [1.0, 2.0])
        np.testing.assert_allclose(embeds[2], [5.0, 6.0])

        d = loader.embeddings(return_type="dict")
        assert set(d.keys()) == {"file1.npy", "file2.npy"}

    def test_embeddings_for_dim_reduction_loader_loads_metadata_if_missing(
        self, tmp_path
    ):
        # a loader created for an already existing dim-reduction combination
        # never has ``metadata_dict`` set; the primary embed dir must then be
        # read from the metadata.yml stored next to the reduced .json files.
        primary_dir = tmp_path / "embeds"
        primary_dir.mkdir(parents=True, exist_ok=True)
        np.save(primary_dir / "file1.npy", np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {
            "model_name": "testmodel",
            "audio_dir": str(TEST_AUDIO_DIR),
            "embed_dir": str(primary_dir),
            "files": {"audio_files": ["audio/foo.wav"], "nr_embeds_per_file": [2]},
        }
        with open(primary_dir / "metadata.yml", "w") as f:
            yaml.safe_dump(metadata, f)

        umap_dir = tmp_path / "umap"
        umap_dir.mkdir(parents=True, exist_ok=True)
        (umap_dir / "file1.json").write_text('{"x": [1.0, 2.0], "y": [3.0, 4.0]}')
        metadata["embed_dir"] = str(primary_dir)
        with open(umap_dir / "metadata.yml", "w") as f:
            yaml.safe_dump(metadata, f)

        loader = self._make_loader(tmp_path)
        if hasattr(loader, "metadata_dict"):
            del loader.metadata_dict  # existing dim-reduction loaders have none
        loader.embed_dir = umap_dir
        loader.embed_suffix = ".json"
        loader.dim_reduction_model = "umap"
        loader.files = sorted(umap_dir.glob("*.json"))

        embeds = loader.embeddings(return_type="array")
        assert embeds.shape == (2, 2)
        np.testing.assert_allclose(embeds[0], [1.0, 2.0])

    def test_update_files_restores_embed_suffix(self, tmp_path):
        embed_dir = tmp_path / "umap"
        embed_dir.mkdir(parents=True, exist_ok=True)
        (embed_dir / "file1.json").write_text('{"x": [1.0, 2.0], "y": [3.0, 4.0]}')

        loader = self._make_loader(tmp_path)
        loader.embed_dir = embed_dir
        loader.embed_suffix = ".npy"  # stale
        loader.dim_reduction_model = "umap"

        loader.update_files()
        assert loader.embed_suffix == ".json"
        assert loader.files == [embed_dir / "file1.json"]

        loader.dim_reduction_model = False
        loader.embed_suffix = ".json"  # stale
        loader.update_files()
        assert loader.embed_suffix == ".npy"

    def test_get_metadata_dict_resolves_old_relative_embed_dir(self, tmp_path):
        # Older bacpipe versions recorded ``embed_dir`` relative to the
        # working directory of the run (e.g.
        # ``bacpipe/evaluation/embeddings/<name>``). The actual directory is
        # the folder the metadata.yml lives in, so it must still be found.
        primary_name = "2025-03-19_12-18___birdnet-AnuranSet"
        primary_dir = (
            tmp_path / "results" / "AnuranSet" / "embeddings" / primary_name
        )
        primary_dir.mkdir(parents=True)
        np.save(primary_dir / "file1.npy", np.array([[1.0, 2.0]]))

        metadata = {
            "model_name": "birdnet",
            "audio_dir": "../Data/AnuranSet",
            "embed_dir": f"bacpipe/evaluation/embeddings/{primary_name}",
            "files": {"audio_files": ["audio/foo.wav"], "nr_embeds_per_file": [1]},
        }
        with open(primary_dir / "metadata.yml", "w") as f:
            yaml.safe_dump(metadata, f)

        loader = self._make_loader(tmp_path)
        loader._get_metadata_dict(primary_dir)
        assert loader.embed_dir == primary_dir

    def test_resolve_embed_dir_finds_sibling_embeddings_folder(self, tmp_path):
        # Old dim-reduction runs stored their metadata in
        # ``dim_reduced_embeddings/`` while the primary embeddings lived in
        # the sibling ``embeddings/`` folder; the recorded ``embed_dir`` was
        # relative and is no longer resolvable from the current CWD.
        primary_name = "2025-03-19_12-18___birdnet-AnuranSet"
        results = tmp_path / "results"
        primary_dir = results / "AnuranSet" / "embeddings" / primary_name
        primary_dir.mkdir(parents=True)
        reduced_dir = (
            results
            / "AnuranSet"
            / "dim_reduced_embeddings"
            / "2025-04-15_15-53___umap-AnuranSet-birdnet"
        )
        reduced_dir.mkdir(parents=True)

        resolved = Loader._resolve_embed_dir(
            f"bacpipe/evaluation/embeddings/{primary_name}", reduced_dir
        )
        assert resolved == primary_dir
        assert resolved.is_dir()

    def test_embeddings_for_dim_reduction_loader_with_old_relative_embed_dir(
        self, tmp_path
    ):
        # End-to-end reproduction of a dim-reduction loader backed by files
        # generated several months ago: the reduced run has a metadata.yml
        # whose ``embed_dir`` is relative to the old working directory, and
        # the primary .npy files are stored in a sibling ``embeddings``
        # folder. ``embeddings()`` must return the primary embeddings
        # instead of silently returning None.
        primary_name = "2025-03-19_12-18___birdnet-AnuranSet"
        results = tmp_path / "results"
        primary_dir = results / "AnuranSet" / "embeddings" / primary_name
        primary_dir.mkdir(parents=True)
        np.save(primary_dir / "file1.npy", np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.save(primary_dir / "file2.npy", np.array([[5.0, 6.0]]))

        reduced_dir = (
            results
            / "AnuranSet"
            / "dim_reduced_embeddings"
            / "2025-04-15_15-53___umap-AnuranSet-birdnet"
        )
        reduced_dir.mkdir(parents=True)
        (reduced_dir / "AnuranSet_umap.json").write_text(
            '{"x": [1.0, 2.0], "y": [3.0, 4.0]}'
        )
        metadata = {
            "model_name": "birdnet",
            "audio_dir": "../Data/AnuranSet",
            "embed_dir": f"bacpipe/evaluation/embeddings/{primary_name}",
            "files": {"audio_files": ["audio/foo.wav"], "nr_embeds_per_file": [2]},
        }
        with open(reduced_dir / "metadata.yml", "w") as f:
            yaml.safe_dump(metadata, f)

        loader = self._make_loader(tmp_path)
        if hasattr(loader, "metadata_dict"):
            del loader.metadata_dict  # existing dim-reduction loaders have none
        loader.embed_dir = reduced_dir
        loader.embed_suffix = ".json"
        loader.dim_reduction_model = "umap"
        loader.files = sorted(reduced_dir.glob("*.json"))

        embeds = loader.embeddings(return_type="array")
        assert embeds.shape == (3, 2)
        np.testing.assert_allclose(embeds[0], [1.0, 2.0])
        np.testing.assert_allclose(embeds[2], [5.0, 6.0])

        d = loader.embeddings(return_type="dict")
        assert set(d.keys()) == {"file1.npy", "file2.npy"}


class TestLoaderPredictions:
    """Tests for ``Loader.predictions()``, used in the example notebooks to
    read the pretrained classifier outputs back from disk."""

    def _make_loader(self, tmp_path):
        return Loader(
            audio_dir=TEST_AUDIO_DIR,
            model_name="testmodel",
            check_if_combination_exists=False,
            testing=True,
            use_folder_structure=False,
            embed_parent_dir=tmp_path / "embeds",
        )

    def _make_preds_fixture(self, tmp_path):
        from types import SimpleNamespace

        preds_path = tmp_path / "preds"
        out_dir = preds_path / "original_classifier_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "a.json").write_text(
            json.dumps(
                {
                    "head": {"Time bins in this file": 2},
                    "Tree Pipit": {
                        "time_bins_exceeding_threshold": [0],
                        "classifier_predictions": [0.9],
                    },
                }
            )
        )

        loader = self._make_loader(tmp_path)
        loader.paths = SimpleNamespace(preds_path=preds_path)
        loader.metadata_dict = {
            "files": {
                "audio_files": [
                    "audio/FewShot/CHE_01_20190101_163410.wav"
                ],
                "nr_embeds_per_file": [2],
            },
            "segment_length (samples)": 48000,
            "sample_rate (Hz)": 48000,
        }
        from pathlib import Path

        loader.files = [
            Path(TEST_AUDIO_DIR) / "FewShot/CHE_01_20190101_163410.wav"
        ]
        return loader

    def test_predictions_array_returns_matrix_and_label_map(self, tmp_path):
        loader = self._make_preds_fixture(tmp_path)
        arr, keys2idx = loader.predictions(return_type="array")
        # 2 time bins x 1 species
        assert arr.shape == (2, 1)
        assert keys2idx == {"Tree Pipit": 0}
        # predictions are stored as float32
        assert arr[0, 0] == pytest.approx(0.9)
        assert arr[1, 0] == 0.0

    def test_predictions_dataframe_has_metadata_columns(self, tmp_path):
        loader = self._make_preds_fixture(tmp_path)
        df = loader.predictions(return_type="dataframe")
        assert "audiofilename" in df.columns
        assert "start" in df.columns
        assert "end" in df.columns
        assert "Tree Pipit" in df.columns
        assert df.iloc[0]["start"] == 0.0
        assert df.iloc[0]["end"] == 1.0

    def test_predictions_without_saved_outputs_warns(self, tmp_path):
        from types import SimpleNamespace

        loader = self._make_loader(tmp_path)
        loader.paths = SimpleNamespace(preds_path=tmp_path / "missing_preds")
        loader.metadata_dict = {
            "files": {
                "audio_files": ["a.wav"],
                "nr_embeds_per_file": [1],
            },
            "segment_length (samples)": 48000,
            "sample_rate (Hz)": 48000,
        }
        preds, _ = loader.predictions(return_type="array")
        assert preds is None
        assert _ is None


class TestSaveLogs:
    def test_honors_kwarg_override(self, tmp_path, monkeypatch):
        import logging

        from bacpipe import config, settings

        monkeypatch.setattr(config, "audio_dir", "some/default/audio")
        monkeypatch.setattr(
            settings, "main_results_dir", str(tmp_path / "default")
        )

        bacpipe_logger = logging.getLogger("bacpipe")
        handlers_before = list(bacpipe_logger.handlers)
        try:
            save_logs(
                main_results_dir=str(tmp_path / "custom"),
                audio_dir=str(tmp_path / "custom_audio"),
            )
        finally:
            # clean up the file handler added by save_logs
            for handler in list(bacpipe_logger.handlers):
                if handler not in handlers_before:
                    handler.close()
                    bacpipe_logger.removeHandler(handler)

        # the explicitly passed kwargs win over config/settings
        log_dir = tmp_path / "custom" / "custom_audio" / "logs"
        assert log_dir.exists()
        saved = {f.name for f in log_dir.iterdir()}
        assert any(name.endswith(".log") for name in saved)
        assert any(name.startswith("config_") for name in saved)
        assert any(name.startswith("settings_") for name in saved)
