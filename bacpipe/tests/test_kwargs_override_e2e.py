"""
End-to-end checks that explicitly passed kwargs override the
``config.yaml`` / ``settings.yaml`` defaults all the way down the pipeline.

This reproduces the real-world failure mode where a user calls the public API
(``bacpipe.play``, ``bacpipe.generate_embeddings``,
``bacpipe.run_pipeline_for_single_model``, ``bacpipe.model_specific_evaluation``)
with values that differ from the config/settings defaults. Historically some
downstream helpers silently reverted to the defaults (``kwargs.get("X",
settings.X)`` returning ``None`` because ``X`` had been stripped from the
kwargs dict), so the results were written to / read from the wrong directory.

``MelSpectrogramModel`` is a checkpoint-free stand-in for a real feature
extractor (mel-spectrograms are treated as embeddings), so the tests run fast
and do not require a model checkpoint on disk.
"""

import shutil
from pathlib import Path

import librosa
import numpy as np
import torch

import bacpipe
from bacpipe import config, settings
from bacpipe.core.workflows import run_pipeline_for_single_model
from bacpipe.embedding_evaluation.label_embeddings import make_set_paths_func
from bacpipe.model_pipelines.model_utils import ModelBaseClass

TEST_AUDIO_DIR = Path("bacpipe/tests/test_data")


class MelSpectrogramModel(ModelBaseClass):
    """Checkpoint-free feature extractor: mel-spectrograms are treated as
    embeddings. No model checkpoint is required."""

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


def _copy_test_data(dst):
    shutil.copytree(TEST_AUDIO_DIR / "audio", dst / "audio")
    shutil.copy(TEST_AUDIO_DIR / "annotations.csv", dst / "annotations.csv")


class TestKwargsOverrideEndToEnd:
    def _make_env(self, tmp_path, monkeypatch):
        user_audio_dir = tmp_path / "user_audio"
        _copy_test_data(user_audio_dir)
        user_results_dir = tmp_path / "user_results"

        # config/settings defaults deliberately differ from the user's values
        monkeypatch.setattr(config, "audio_dir", str(tmp_path / "config_audio"))
        monkeypatch.setattr(
            settings, "main_results_dir", str(tmp_path / "settings_results")
        )
        assert config.audio_dir != str(user_audio_dir)
        assert settings.main_results_dir != str(user_results_dir)
        return user_audio_dir, user_results_dir

    def test_run_pipeline_for_single_model_uses_user_dirs(
        self, tmp_path, monkeypatch
    ):
        user_audio_dir, user_results_dir = self._make_env(tmp_path, monkeypatch)

        loader = run_pipeline_for_single_model(
            model_name="mel",
            dim_reduction_model="None",
            CustomModel=MelSpectrogramModel,
            audio_dir=str(user_audio_dir),
            main_results_dir=str(user_results_dir),
            device="cpu",
            run_pretrained_classifier=True,
            only_embed_annotations=True,
            check_if_already_processed=False,
        )

        paths = make_set_paths_func(
            str(user_audio_dir), main_results_dir=str(user_results_dir)
        )("mel")

        assert paths.main_embeds_path.exists()
        embeds = loader.embeddings(return_type="array")
        assert len(embeds) > 1
        # nothing written to the config/settings default results dir
        assert not Path(settings.main_results_dir).exists()

    def test_play_with_probing_and_clustering_uses_user_dirs(
        self, tmp_path, monkeypatch
    ):
        user_audio_dir, user_results_dir = self._make_env(tmp_path, monkeypatch)

        bacpipe.play(
            models=["mel"],
            CustomModels=[MelSpectrogramModel],
            audio_dir=str(user_audio_dir),
            main_results_dir=str(user_results_dir),
            device="cpu",
            # ``mel`` has no pretrained classifier; this is the value that used
            # to crash inside ``classifier_should_be_run`` when no prediction
            # directory existed yet.
            run_pretrained_classifier=False,
            only_embed_annotations=True,
            overwrite=True,
            dashboard=False,
            evaluation_task=["probing", "clustering"],
            dim_reduction_model="None",
            # balanced 2-class problem (A vs B call types)
            label_column="call_type",
        )

        paths = make_set_paths_func(
            str(user_audio_dir), main_results_dir=str(user_results_dir)
        )("mel")

        # embeddings, probing and clustering all land under the user's dir
        assert paths.main_embeds_path.exists()
        assert (paths.probe_path / "probe_results_linear.json").exists()
        assert (paths.probe_path / "probe_results_knn.json").exists()
        assert len(list(paths.clust_path.glob("*.json"))) > 0

        # and the config/settings defaults are still untouched
        assert not Path(settings.main_results_dir).exists()
