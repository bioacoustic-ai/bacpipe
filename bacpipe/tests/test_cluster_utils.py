"""
Unit tests for the clustering utilities in
``bacpipe.embedding_evaluation.clustering.cluster``.
"""

import json

import numpy as np
import pytest
from types import SimpleNamespace

from sklearn.cluster import KMeans

from bacpipe.embedding_evaluation.clustering.cluster import (
    convert_numpy_types,
    eval_clustering,
    eval_with_silhouette,
    get_clustering_models,
    get_nr_of_clusters,
    run_clustering,
    save_clustering_performance,
)


class TestConvertNumpyTypes:
    def test_int64_converted_to_int(self):
        assert convert_numpy_types(np.int64(5)) == 5
        assert isinstance(convert_numpy_types(np.int64(5)), int)

    def test_float32_converted_to_float(self):
        assert convert_numpy_types(np.float32(2.5)) == 2.5
        assert isinstance(convert_numpy_types(np.float32(2.5)), float)

    def test_ndarray_converted_to_list(self):
        assert convert_numpy_types(np.array([1, 2, 3])) == [1, 2, 3]

    def test_other_types_return_none(self):
        assert convert_numpy_types("not a numpy type") is None
        assert convert_numpy_types(5) is None


class TestSaveClusteringPerformance:
    def _make_paths(self, tmp_path):
        clust_path = tmp_path / "clustering"
        clust_path.mkdir()
        return SimpleNamespace(clust_path=clust_path)

    def test_saves_clusterings_and_metrics(self, tmp_path):
        paths = self._make_paths(tmp_path)
        clusterings = {"kmeans": np.array([0, 1, 0])}
        metrics = {
            "AMI": {"kmeans-ground_truth": 1.0},
            "nr_of_embeddings": np.int64(3),
        }
        save_clustering_performance(
            paths, clusterings, metrics, label_column="species"
        )

        saved = np.load(
            paths.clust_path / "clust_labels.npy", allow_pickle=True
        ).item()
        assert set(saved.keys()) == {"kmeans"}

        with open(paths.clust_path / "clust_results.json") as f:
            results = json.load(f)
        assert results["AMI"]["kmeans-ground_truth"] == 1.0
        # numpy types are converted before serialization
        assert results["nr_of_embeddings"] == 3

    def test_label_column_keys_are_removed(self, tmp_path):
        paths = self._make_paths(tmp_path)
        clusterings = {
            "kmeans": np.array([0, 1]),
            "species": np.array(["a", "b"]),
            "species_no_noise": np.array(["a", "b"]),
        }
        save_clustering_performance(
            paths, clusterings, {}, label_column="species"
        )
        saved = np.load(
            paths.clust_path / "clust_labels.npy", allow_pickle=True
        ).item()
        assert list(saved.keys()) == ["kmeans"]

    def test_no_metrics_means_no_json(self, tmp_path):
        paths = self._make_paths(tmp_path)
        save_clustering_performance(
            paths, {"kmeans": np.array([0])}, {}, label_column="species"
        )
        assert not (paths.clust_path / "clust_results.json").exists()

class TestRunClustering:
    def test_returns_labels_for_each_config(self):
        embeds = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        cluster_configs = {
            "kmeans": KMeans(n_clusters=2, random_state=42, n_init=10)
        }
        clusterings = run_clustering(
            embeds, cluster_configs, label_column=None
        )
        assert set(clusterings.keys()) == {"kmeans"}
        assert len(clusterings["kmeans"]) == len(embeds)

    def test_includes_ground_truth_when_provided(self):
        embeds = np.array([[1.0], [0.0], [1.0], [0.0]])
        ground_truth = np.array(["a", "b", "a", "noise"])
        cluster_configs = {
            "kmeans": KMeans(n_clusters=2, random_state=42, n_init=10)
        }
        clusterings = run_clustering(
            embeds, cluster_configs, label_column="species",
            ground_truth=ground_truth,
        )
        assert "species" in clusterings
        assert "species_no_noise" in clusterings
        assert np.array_equal(clusterings["species"], ground_truth)
        assert len(clusterings["kmeans_no_noise"]) == 3


class TestEvalClustering:
    def test_without_metadata_labels(self):
        clusterings = {"kmeans": np.array([0, 0, 1, 1])}
        ground_truth = np.array([0, 0, 1, 1])
        results = eval_clustering(
            clusterings, ground_truth, label_column="species"
        )
        assert set(results.keys()) == {"AMI", "ARI"}
        assert "kmeans-ground_truth" in results["AMI"]
        assert "kmeans-ground_truth" in results["ARI"]

    def test_with_metadata_labels(self):
        clusterings = {
            "kmeans": np.array([0, 0, 1, 1]),
            "species": np.array([0, 0, 1, 1]),
        }
        ground_truth = np.array([0, 0, 1, 1])
        metadata_labels = {"time_of_day": np.array([0, 1, 0, 1])}
        results = eval_clustering(
            clusterings,
            ground_truth,
            metadata_labels=metadata_labels,
            label_column="species",
        )
        assert "kmeans-time_of_day" in results["AMI"]
        assert "kmeans-time_of_day" in results["ARI"]

    def test_no_noise_branch_filters_embeds(self):
        clusterings = {
            "kmeans": np.array([0, 0, 1, 1]),
            "species_no_noise": np.array(["a", "a", "b", "b"]),
        }
        ground_truth = np.array(["a", "a", "b", "noise"])
        embeds = np.array([[0.0], [0.1], [1.0], [2.0]])
        metadata_labels = {"species": ground_truth.copy()}
        results = eval_clustering(
            clusterings,
            ground_truth,
            embeds=embeds,
            metadata_labels=metadata_labels,
            label_column="species",
        )
        # the no-noise branch filters both the metadata labels and the
        # cluster labels, so the AMI scores stay well-defined
        assert "species_no_noise-species" in results["AMI"]
        assert "kmeans-species" in results["AMI"]


class TestEvalWithSilhouette:
    def test_adds_silhouette_score(self):
        embeds = np.array(
            [[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [0.9, 0.9]]
        )
        labels = np.array([0, 0, 1, 1])
        metrics = eval_with_silhouette(embeds, labels)
        assert "SS" in metrics
        assert 0 <= metrics["SS"] <= 1

    def test_keeps_existing_metrics(self):
        embeds = np.array(
            [[0.0], [0.1], [1.0], [0.9]]
        )
        labels = np.array([0, 0, 1, 1])
        metrics = eval_with_silhouette(
            embeds, labels, metrics={"AMI": 1.0}
        )
        assert metrics["AMI"] == 1.0
        assert "SS" in metrics


class TestGetClusteringModels:
    def test_returns_kmeans_model(self):
        models = get_clustering_models({"kmeans": {"n_clusters": 3}})
        assert isinstance(models["kmeans"], KMeans)
        assert models["kmeans"].n_clusters == 3


class TestGetNrOfClusters:
    def _config(self, n_clusters):
        return {
            "kmeans": {
                "name": "kmeans",
                "params": {"n_clusters": n_clusters},
                "bool": True,
            }
        }

    def test_uses_explicit_n_clusters(self):
        params = get_nr_of_clusters([], self._config(5))
        assert params["kmeans"]["n_clusters"] == 5

    def test_uses_nr_of_classes_from_labels(self):
        labels = ["a", "b", "a", "c"]
        params = get_nr_of_clusters(labels, self._config(None))
        assert params["kmeans"]["n_clusters"] == 3

    def test_defaults_to_42(self):
        params = get_nr_of_clusters([], self._config("None"))
        assert params["kmeans"]["n_clusters"] == 42

    def test_skips_disabled_configs(self):
        configs = {
            "kmeans": self._config(3)["kmeans"],
            "hdbscan": {
                "name": "hdbscan",
                "params": {"min_cluster_size": 10},
                "bool": False,
            },
        }
        params = get_nr_of_clusters([], configs)
        assert list(params.keys()) == ["kmeans"]

