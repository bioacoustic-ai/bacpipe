"""
End-to-end checks that mirror the code cells of
``bacpipe/examples/basic_examples/using_different_bacpipe_functions.ipynb``.

Two checkpoint-free mel-spectrogram "models" (``MelMockModelA`` and
``MelMockModelB``) stand in for real feature extractors, so the full workflow
-- folder structure, embeddings, model-specific evaluation (probing +
clustering), probing reuse, clustering evaluation and the cross-model
evaluation with overview plots -- can be exercised quickly without
downloading any model checkpoints.

Dimensionality reduction is kept at ``"None"`` here to keep the test fast and
deterministic; the notebook itself additionally demonstrates ``umap`` (which
adds the embedding-space comparison plot ``overview/comp_fig.png``).
"""

import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

import bacpipe
from bacpipe.embedding_evaluation.label_embeddings import make_set_paths_func
from bacpipe.model_pipelines.model_utils import ModelBaseClass

TEST_AUDIO_DIR = Path("bacpipe/tests/test_data")


class _MelMockModelBase(ModelBaseClass):
    """Checkpoint-free feature extractor: mel-spectrograms are treated as
    embeddings. No model checkpoint is required."""

    SAMPLE_RATE = 48_000
    SEGMENT_LENGTH = SAMPLE_RATE  # 1 second windows
    n_mels = 64

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
                    n_mels=self.n_mels,
                    n_fft=1024,
                    hop_length=512,
                )
                for segment in audio
            ]
        )
        return torch.tensor(mels.reshape(len(mels), -1), dtype=torch.float32)


class MelMockModelA(_MelMockModelBase):
    n_mels = 64


class MelMockModelB(_MelMockModelBase):
    n_mels = 96


def _copy_test_data(dst):
    shutil.copytree(TEST_AUDIO_DIR / "audio", dst / "audio")
    shutil.copy(TEST_AUDIO_DIR / "annotations.csv", dst / "annotations.csv")


class TestNotebookWorkflow:
    """Mirrors ``using_different_bacpipe_functions.ipynb`` top to bottom."""

    MODELS = ["mel_mock_a", "mel_mock_b"]
    CUSTOM_MODELS = [MelMockModelA, MelMockModelB]

    def test_full_workflow(self, tmp_path):
        audio_dir = tmp_path / "audio_data"
        _copy_test_data(audio_dir)
        results_dir = tmp_path / "results"

        # --- 2. Folder structure & paths -----------------------------------
        get_paths = make_set_paths_func(
            str(audio_dir), main_results_dir=str(results_dir)
        )
        paths = get_paths(self.MODELS[0])
        assert paths.main_embeds_path.exists()

        # nothing has been evaluated yet
        assert not bacpipe.evaluation_with_settings_already_exists(
            audio_dir=str(audio_dir),
            dim_reduction_model="None",
            models=self.MODELS,
            main_results_dir=str(results_dir),
        )

        # --- 4. Embeddings for multiple models -----------------------------
        loader_dict = bacpipe.run_pipeline_for_models(
            models=self.MODELS,
            audio_dir=str(audio_dir),
            dim_reduction_model="None",
            CustomModels=self.CUSTOM_MODELS,
            main_results_dir=str(results_dir),
            device="cpu",
            run_pretrained_classifier=False,
            only_embed_annotations=True,
            check_if_already_processed=False,
        )
        assert set(loader_dict) == set(self.MODELS)
        for loader in loader_dict.values():
            assert len(loader.embeddings(return_type="array")) > 1

        # --- 5. Custom metadata labels -------------------------------------
        label_maker = bacpipe.MetadataLabelMaker(
            paths=paths,
            model=self.MODELS[0],
            metadata_label_keys=["audio_file_name", "time_of_day"],
        )
        label_maker.generate()
        assert set(label_maker.metadata_label_dict) == {
            "audio_file_name",
            "time_of_day",
        }

        # --- 6. Model-specific evaluation (probing + clustering) -----------
        bacpipe.model_specific_evaluation(
            loader_dict=loader_dict,
            evaluation_task=["probing", "clustering"],
            audio_dir=str(audio_dir),
            main_results_dir=str(results_dir),
            device="cpu",
            run_pretrained_classifier=False,
            only_embed_annotations=True,
            label_column="call_type",
            CustomModels=self.CUSTOM_MODELS,
        )
        assert (paths.probe_path / "probe_results_linear.json").exists()
        assert (paths.probe_path / "probe_results_knn.json").exists()
        assert len(list(paths.clust_path.glob("*.json"))) > 0

        # --- 7. Probing (reuse existing results) ---------------------------
        gt = bacpipe.ground_truth_by_model(
            model=self.MODELS[0],
            audio_dir=str(audio_dir),
            main_results_dir=str(results_dir),
            overwrite=False,
            label_column="call_type",
        )

        probe, label2index, metrics = bacpipe.probing_pipeline(
            model_name=self.MODELS[0],
            ground_truth=gt,
            embeds=loader_dict[self.MODELS[0]].embeddings(return_type="array"),
            paths=paths,
            name="linear",
            overwrite=False,
        )
        assert list(label2index.keys()) == ["A", "B"]
        assert "overall" in metrics

        # --- 8. Clustering (low-level + high-level) ------------------------
        from sklearn.cluster import KMeans

        embeds = loader_dict[self.MODELS[0]].embeddings(return_type="array")
        gt_labels = gt["simultaneous_labels"].values

        clusterings = bacpipe.run_clustering(
            embeds=embeds,
            cluster_configs={"kmeans": KMeans(n_clusters=2, n_init=10)},
            ground_truth=gt_labels,
        )
        assert "kmeans" in clusterings

        metadata = bacpipe.metadata_labels(
            model=self.MODELS[0],
            audio_dir=str(audio_dir),
            main_results_dir=str(results_dir),
            overwrite=False,
            return_type="dict",
        )
        clustering_metrics = bacpipe.eval_clustering(
            clusterings,
            ground_truth=gt_labels,
            embeds=embeds,
            metadata_labels=metadata,
        )
        assert "AMI" in clustering_metrics and "ARI" in clustering_metrics

        silhouette = bacpipe.eval_with_silhouette(
            embeds, ground_truth=gt_labels
        )
        assert "SS" in silhouette

        _, clust_results = bacpipe.clustering_pipeline(
            model_name=self.MODELS[0],
            ground_truth=gt,
            embeds=embeds,
            paths=paths,
            overwrite=False,
        )
        assert clust_results

        # --- 9. Cross-model evaluation -------------------------------------
        bacpipe.cross_model_evaluation(
            audio_dir=str(audio_dir),
            models=self.MODELS,
            evaluation_task=["probing", "clustering"],
            dim_reduction_model="None",
            main_results_dir=str(results_dir),
            device="cpu",
            CustomModels=self.CUSTOM_MODELS,
        )

        overview_dir = paths.plot_path.parent.parent / "overview"
        assert overview_dir.exists()
        pngs = sorted(overview_dir.glob("*.png"))
        assert len(pngs) > 0, f"no overview plots found in {overview_dir}"

    def test_constants_are_exported(self):
        # Notebook section 2: the package-level constants.
        assert isinstance(bacpipe.supported_models, list) and bacpipe.supported_models
        assert isinstance(bacpipe.models_needing_checkpoint, list)
        assert isinstance(bacpipe.TF_MODELS, list) and bacpipe.TF_MODELS
        assert isinstance(bacpipe.EMBEDDING_DIMENSIONS, dict)
        assert len(bacpipe.EMBEDDING_DIMENSIONS) == len(bacpipe.supported_models)
        assert isinstance(bacpipe.NEEDS_CHECKPOINT, list)

    def test_audio_handling(self):
        # Notebook section 5: AudioHandler whole-file + annotated-segment loading.
        model = MelMockModelA(
            model_name="mel_mock_a",
            device="cpu",
            run_pretrained_classifier=False,
        )
        aud = bacpipe.AudioHandler(model=model, audio_dir=str(TEST_AUDIO_DIR))
        files = bacpipe.get_audio_files(str(TEST_AUDIO_DIR))
        assert files

        audio, sr = aud.load_and_resample(files[0])
        assert sr == model.sr
        frames = aud.window_audio(audio)
        assert frames.ndim == 2
        assert frames.shape[1] == model.segment_length

        aud_annotated = bacpipe.AudioHandler(
            model=model,
            audio_dir=str(TEST_AUDIO_DIR),
            only_embed_annotations=True,
        )
        annotated_frames = aud_annotated.only_load_annotated_segments(files[0])
        assert annotated_frames.ndim == 2
        assert annotated_frames.shape[1] == model.segment_length

    def test_embeddings_from_audio(self):
        # Notebook section 6: Embedder + get_embeddings_from_model.
        embed_obj = bacpipe.Embedder(
            model_name="mel_mock_a",
            CustomModel=MelMockModelA,
            audio_dir=str(TEST_AUDIO_DIR),
            device="cpu",
            run_pretrained_classifier=False,
        )
        audio_files = bacpipe.get_audio_files(str(TEST_AUDIO_DIR))
        embeds = embed_obj.get_embeddings_from_model(audio_files[0])
        assert embeds.ndim == 2
        assert embeds.shape[0] > 0
        assert embeds.shape[1] > 0

    def test_get_dt_filename(self):
        # Notebook section 8: filename datetime extraction.
        dt = bacpipe.get_dt_filename("CHE_01_20190101_163410.wav")
        assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (
            2019,
            1,
            1,
            16,
            34,
            10,
        )


class _FakeLoader:
    """Minimal loader stand-in returning precomputed classifier predictions."""

    def __init__(self, preds, label2idx):
        self._preds = preds
        self._label2idx = label2idx

    def predictions(self, return_type="array", **kwargs):
        return self._preds, self._label2idx


class TestNotebookBenchmark:
    """Mirrors notebook section 12 (Benchmarking) without a real model.

    ``benchmark()`` is driven with a stubbed ground-truth table and a stubbed
    loader so the label-alignment + ``classification_report`` logic can be
    checked quickly -- including the single-shared-class edge case that used to
    crash ``classification_report()``.
    """

    @staticmethod
    def _stub_benchmark_deps(monkeypatch, gt, preds, label2idx):
        monkeypatch.setattr(
            bacpipe, "confirm_model_name", lambda model, **kw: "birdnet"
        )
        monkeypatch.setattr(
            bacpipe, "ground_truth_by_model", lambda *a, **kw: gt
        )
        monkeypatch.setattr(
            bacpipe,
            "run_pipeline_for_single_model",
            lambda *a, **kw: _FakeLoader(preds, label2idx),
        )

    def test_benchmark_multiple_shared_classes(self, monkeypatch):
        gt = pd.DataFrame(
            {
                "start": [0.0, 1.0, 2.0, 3.0],
                "end": [1.0, 2.0, 3.0, 4.0],
                "audiofilename": ["a.wav"] * 4,
                "simultaneous_labels": [0, 1, 1, 0],
                "Common Chaffinch": [1, 0, 1, 1],
                "Common Cuckoo": [0, 1, 0, 0],
            }
        )
        label2idx = {"Common Chaffinch": 0, "Common Cuckoo": 1}
        preds = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
        self._stub_benchmark_deps(monkeypatch, gt, preds, label2idx)

        report = bacpipe.benchmark(
            model="birdnet",
            dataset="bacpipe/tests/test_data",
            annotations_file="annotations.csv",
            overwrite=False,
        )
        assert isinstance(report, dict)
        assert set(report["report"]) >= {"Common Chaffinch", "Common Cuckoo"}

    def test_benchmark_single_shared_class(self, monkeypatch):
        # Regression for notebook section 12: a custom model whose classifier
        # only matches one annotated class used to raise
        # "Number of classes, 2, does not match size of target_names, 1".
        gt = pd.DataFrame(
            {
                "start": [0.0, 1.0, 2.0, 3.0],
                "end": [1.0, 2.0, 3.0, 4.0],
                "audiofilename": ["a.wav"] * 4,
                "simultaneous_labels": [0, 1, 1, 0],
                "Eurasian Blackbird": [1, 0, 1, 0],
            }
        )
        label2idx = {"Eurasian Blackbird": 0}
        preds = np.array([[0.9], [0.1], [0.8], [0.2]])
        self._stub_benchmark_deps(monkeypatch, gt, preds, label2idx)

        report = bacpipe.benchmark(
            model="birdnet",
            dataset="bacpipe/tests/test_data",
            annotations_file="annotations.csv",
            overwrite=True,
        )
        assert isinstance(report, dict)
        # The single shared class must be reported with real metrics, not the
        # empty "support 0" row that a naive ``labels=[0]`` produces.
        assert report["report"]["Eurasian Blackbird"]["support"] == 2
        assert report["report"]["Eurasian Blackbird"]["precision"] == 1.0
        assert report["report"]["Eurasian Blackbird"]["recall"] == 1.0

    def test_benchmark_without_predictions_returns_error(self, monkeypatch):
        gt = pd.DataFrame(
            {
                "start": [0.0],
                "end": [1.0],
                "audiofilename": ["a.wav"],
                "simultaneous_labels": [0],
                "Common Chaffinch": [1],
            }
        )
        self._stub_benchmark_deps(monkeypatch, gt, None, None)

        report = bacpipe.benchmark(
            model="birdnet",
            dataset="bacpipe/tests/test_data",
            annotations_file="annotations.csv",
        )
        assert report == {
            "error": (
                "No predictions have been generated, or model does not have "
                "classifier."
            )
        }
