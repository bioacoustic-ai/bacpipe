"""
Unit tests for the visualization helpers in
``bacpipe.embedding_evaluation.visualization``.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    get_arrays_for_spectrogram_text,
    get_boolean_array_for_annotated_embeddings,
    get_single_label_gt_labels,
)
from bacpipe.embedding_evaluation.visualization.visualize_predictions import (
    PredictionsLoader,
)
from bacpipe.embedding_evaluation.visualization.visualize import (
    plot_overview_results,
)


class TestVerifyThreshold:
    def test_empty_string_defaults_to_half(self):
        assert PredictionsLoader.verify_threshold("") == 0.5

    def test_string_threshold_is_parsed(self):
        assert PredictionsLoader.verify_threshold("0.75") == 0.75

    def test_float_threshold_passes_through(self):
        assert PredictionsLoader.verify_threshold(0.2) == 0.2


class TestReorderByMostOccurrance:
    def test_orders_classes_by_decreasing_occurrence(self):
        probs = np.array([[1, 0, 1], [1, 1, 0]])
        label2index = {"a": 0, "b": 1, "c": 2}
        ordered = PredictionsLoader.reorder_by_most_occurrance(
            probs, label2index
        )
        assert list(ordered.keys()) == ["a", "b", "c"]


class TestTransformPresenceIntoHourHeatmap:
    def test_builds_24_hour_by_time_bin_matrix(self):
        presence = np.array([1, 0, 1, 0])
        hours = np.array([0, 1, 0, 1])
        accumulator = np.array(
            [
                [2024, 1, 1],
                [2024, 1, 1],
                [2024, 1, 2],
                [2024, 1, 2],
            ]
        )
        heatmap = PredictionsLoader.transform_presence_into_hour_heatmap(
            presence, hours, accumulator
        )
        assert heatmap.shape == (24, 2)
        assert heatmap[0, 0] == 1
        assert heatmap[1, 0] == 0
        assert heatmap[0, 1] == 1
        assert heatmap[1, 1] == 0
        # hours without any embeddings stay at -1
        assert heatmap[2, 0] == -1


class TestGetSingleLabelGtLabels:
    def test_reduces_multi_label_to_single_label(self):
        df = pd.DataFrame(
            {
                "audiofilename": ["a.wav", "a.wav"],
                "start": [0, 5],
                "end": [5, 10],
                "simultaneous_labels": [1, 1],
                "Tree Pipit": [1, 0],
                "Eurasian Kestrel": [0, 1],
            }
        )
        bool_noise = np.array([False, False, True])
        labels = get_single_label_gt_labels(df, bool_noise)
        assert labels[0] == "Tree Pipit"
        assert labels[1] == "Eurasian Kestrel"
        assert labels[2] == "noise"


class TestGetBooleanArrayForAnnotatedEmbeddings:
    def _metadata_labels(self):
        # one row per embedding segment (matches the model time grid)
        return pd.DataFrame(
            {
                "audio_file_name": ["a.wav", "a.wav", "b.wav"],
                "start": [0.0, 3.0, 0.0],
                "end": [3.0, 6.0, 3.0],
            }
        )

    def _ground_truth(self):
        return pd.DataFrame(
            {
                "audiofilename": ["a.wav", "b.wav"],
                "start": [0.0, 0.0],
                "end": [3.0, 3.0],
                "simultaneous_labels": [1, 2],
                "sp_a": [1, 1],
                "sp_b": [0, 1],
            }
        )

    def _patch_metadata_labels(self, monkeypatch):
        import bacpipe.embedding_evaluation.label_embeddings as le

        monkeypatch.setattr(
            le, "create_metadata_labels", lambda **kwargs: self._metadata_labels()
        )

    def test_marks_unannotated_embeddings_as_noise(self, monkeypatch):
        self._patch_metadata_labels(monkeypatch)
        is_noise = get_boolean_array_for_annotated_embeddings(
            self._ground_truth(), "birdnet"
        )
        # a.wav@0 and b.wav@0 are annotated; a.wav@3 has no annotation
        assert is_noise.tolist() == [False, True, False]

    def test_returns_boolean_array(self, monkeypatch):
        self._patch_metadata_labels(monkeypatch)
        is_noise = get_boolean_array_for_annotated_embeddings(
            self._ground_truth(), "birdnet"
        )
        assert isinstance(is_noise, np.ndarray)
        assert is_noise.dtype == bool

    def test_all_segments_annotated(self, monkeypatch):
        self._patch_metadata_labels(monkeypatch)
        gt = pd.DataFrame(
            {
                "audiofilename": ["a.wav", "a.wav", "b.wav"],
                "start": [0.0, 3.0, 0.0],
                "end": [3.0, 6.0, 3.0],
                "simultaneous_labels": [1, 1, 1],
                "sp_a": [1, 1, 1],
            }
        )
        is_noise = get_boolean_array_for_annotated_embeddings(gt, "birdnet")
        assert is_noise.tolist() == [False, False, False]


class TestGetArraysForSpectrogramText:
    def test_filters_labels_by_settings_and_data_dict(self):
        labels = {
            "label": ["a", "b"],
            "time_of_day": ["morning", "evening"],
            "audio_file_name": ["a.wav", "b.wav"],
            "kmeans": [0, 1],
            "custom_attr": ["x", "y"],
        }
        data_dict = {"custom_attr": np.array(["x", "y"])}
        embeds = {"metadata": {"model_name": "x", "embed_dir": "/tmp"}}
        out = get_arrays_for_spectrogram_text(
            labels, "label", data_dict, embeds
        )
        assert out == {}

    def test_keeps_custom_arrays(self):
        labels = {
            "label": ["a", "b"],
            "custom_attr": ["x", "y"],
        }
        data_dict = {}
        embeds = {"metadata": {"model_name": "x", "embed_dir": "/tmp"}}
        out = get_arrays_for_spectrogram_text(
            labels, "label", data_dict, embeds
        )
        assert out == {"custom_attr": ["x", "y"]}


class TestPlotOverviewResults:
    """Regression tests for the probing overview bar plot.

    The overview plot used by the dashboard must read the per-model
    ``probe_results_*.json`` files directly (instead of the aggregated
    ``overview/probing_results.json``, which can be missing or stale) and
    must show the same metrics as the per-model probing plots.
    """

    @staticmethod
    def _make_probe_results(tmp_path, model_names, configs=("linear", "knn")):
        """Write probe_results_<config>.json files and return a path_func."""
        import json

        probe_dirs = {}
        for m_idx, model in enumerate(model_names):
            probe_dir = tmp_path / model / "probing"
            probe_dir.mkdir(parents=True)
            probe_dirs[model] = probe_dir
            for i, config in enumerate(configs):
                results = {
                    "overall": {
                        "macro_accuracy": 0.5 + m_idx / 10 + i / 10,
                        "micro_accuracy": 0.7 + m_idx / 10 + i / 10,
                        "auc": 0.8 + m_idx / 10 + i / 10,
                        "macro_f1": 0.4 + m_idx / 10 + i / 10,
                        "micro_f1": 0.7 + m_idx / 10 + i / 10,
                    },
                    "per_class_accuracy": {"a": 0.6, "b": 0.6},
                }
                with open(
                    probe_dir / f"probe_results_{config}.json", "w"
                ) as f:
                    json.dump(results, f)

        def path_func(model_name):
            return SimpleNamespace(probe_path=probe_dirs[model_name])

        return path_func

    @staticmethod
    def _bar_heights_by_model(ax):
        """Group bar heights by model index. Models are sorted by the first
        metric (macro_accuracy) in descending order, and each model's bars
        are centered around the corresponding x-tick position."""
        ticks = ax.get_xticks()
        heights = {i: [] for i in range(len(ticks))}
        for p in ax.patches:
            center = p.get_x() + p.get_width() / 2
            model_idx = int(np.argmin(np.abs(ticks - center)))
            heights[model_idx].append(round(float(p.get_height()), 3))
        return heights

    def test_loads_per_model_results_without_aggregate_file(self, tmp_path):
        """The dashboard path must work without overview/probing_results.json."""
        models = ["model_a", "model_b"]
        path_func = self._make_probe_results(tmp_path, models)

        fig = plot_overview_results(
            plot_path=None,
            task_name="linear",
            model_list=models,
            metrics=None,
            path_func=path_func,
            return_fig=True,
        )
        ax = fig.axes[0]
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        # micro-averaged metrics are dropped so the overview matches the
        # per-model probing plots shown in the dashboard
        assert legend == ["macro_accuracy", "auc", "macro_f1"]
        # model_b (macro 0.6) is sorted before model_a (macro 0.5)
        heights = self._bar_heights_by_model(ax)
        assert heights == {0: [0.6, 0.9, 0.5], 1: [0.5, 0.8, 0.4]}

    def test_knn_task_selects_knn_results(self, tmp_path):
        """The 'knn' classification type must pick the knn probe results."""
        models = ["model_a", "model_b"]
        path_func = self._make_probe_results(tmp_path, models)

        fig = plot_overview_results(
            plot_path=None,
            task_name="knn",
            model_list=models,
            metrics=None,
            path_func=path_func,
            return_fig=True,
        )
        ax = fig.axes[0]
        heights = self._bar_heights_by_model(ax)
        # model_b knn (macro 0.7) sorted before model_a knn (macro 0.6)
        assert heights == {0: [0.7, 1.0, 0.6], 1: [0.6, 0.9, 0.5]}

