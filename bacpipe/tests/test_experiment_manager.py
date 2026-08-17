"""
Unit tests for the experiment manager helpers in
``bacpipe.core.experiment_manager``.
"""

import json

import pytest

from bacpipe import Loader
from bacpipe.core.experiment_manager import (
    replace_default_kwargs_with_user_kwargs,
    return_reduced_dimensions,
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
