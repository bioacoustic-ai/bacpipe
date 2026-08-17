"""
Unit tests for the visualization helpers in
``bacpipe.embedding_evaluation.visualization``.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    get_arrays_for_spectrogram_text,
    get_boolean_array_for_annotated_embeddings,
    get_single_label_gt_labels,
    return_rows_cols,
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


from pathlib import Path

import pytest

import bacpipe
from bacpipe import settings
from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    plot_embeddings_px,
    set_legend,
)
from bacpipe.embedding_evaluation.visualization import visualize_predictions


def _make_embeds(n=40):
    """Build a minimal embeddings dict for ``plot_embeddings_px``."""
    rng = np.random.default_rng(0)
    return {
        "x": rng.normal(size=n).tolist(),
        "y": rng.normal(size=n).tolist(),
        "z": None,
        "timestamp": np.arange(n).tolist(),
        "durations": [1.0] * n,
        "index": list(range(n)),
        "metadata": {
            "audio_files": [f"file_{i % 4}.wav" for i in range(n)],
            "segment_length (samples)": [32000] * n,
            "sample_rate (Hz)": [32000] * n,
            "model_name": "birdnet",
            "embed_dir": "/tmp/does/not/matter",
        },
    }


class TestPlotEmbeddingsPxDiscreteVsContinuous:
    """The plotly embedding plot must use a discrete legend (and no colorbar)
    whenever the number of categories is below ``settings.max_nr_categories``,
    even when the labels are numeric (e.g. integer kmeans cluster ids)."""

    def test_integer_cluster_labels_use_discrete_legend(self):
        labels = {"kmeans": np.array([0, 1, 2, 3] * 10, dtype=np.int32)}
        fig = plot_embeddings_px(_make_embeds(), labels, label_by="kmeans")
        layout = fig.layout.to_plotly_json()
        assert "coloraxis" not in layout
        assert layout.get("legend")
        # one trace per cluster -> categorical legend entries
        assert len(fig.data) == 4

    def test_float_cluster_labels_use_discrete_legend(self):
        labels = {"kmeans": np.array([0.0, 1.0, 2.0, 3.0] * 10)}
        fig = plot_embeddings_px(_make_embeds(), labels, label_by="kmeans")
        layout = fig.layout.to_plotly_json()
        assert "coloraxis" not in layout
        assert len(fig.data) == 4

    def test_string_labels_keep_discrete_legend(self):
        labels = {"label": np.array(["a", "b", "c", "d"] * 10)}
        fig = plot_embeddings_px(_make_embeds(), labels, label_by="label")
        layout = fig.layout.to_plotly_json()
        assert "coloraxis" not in layout
        assert len(fig.data) == 4

    def test_high_cardinality_labels_keep_colorbar(self, monkeypatch):
        # More categories than the threshold must keep the gradient colorbar.
        monkeypatch.setattr(settings, "max_nr_categories", 5)
        labels = {"file": np.array([f"f{i % 40}" for i in range(40)])}
        fig = plot_embeddings_px(_make_embeds(), labels, label_by="file")
        layout = fig.layout.to_plotly_json()
        assert "coloraxis" in layout


class TestPredictionsLoaderCacheConsistency:
    """A failed load must not poison the PredictionsLoader cache. Otherwise
    the single model predictions tab crashes with a KeyError when switching
    between classifier types/models after a failed probe run."""

    class _FakeVisLoader:
        def __init__(self):
            n = 6
            self.embeds = {
                "birdnet": {
                    "x": np.arange(n).tolist(),
                    "y": np.arange(n).tolist(),
                    "timestamp": np.arange(n).tolist(),
                    "metadata": {
                        "audio_files": [
                            "20240101060000.wav",
                            "20240102060000.wav",
                        ],
                        "nr_embeds_per_file": [3, 3],
                    },
                }
            }

    class _FakePanelSelection:
        def __init__(self):
            self.options = []
            self.value = None

    @staticmethod
    def _fake_load_classification(model, threshold):
        binary_presence = np.zeros((6, 2), dtype=np.int8)
        binary_presence[:3, 0] = 1
        binary_presence[3:, 1] = 1
        return binary_presence, {"class_a": 0, "class_b": 1}

    @staticmethod
    def _fake_prepare_probe(model, probe_path):
        return object(), {"probe_a": 0, "probe_b": 1}

    @staticmethod
    def _fake_run_probe_success(
        model, probe, threshold, return_binary_presence=True, callbacks=None
    ):
        binary_presence = np.zeros((6, 2), dtype=np.int8)
        binary_presence[:3, 0] = 1
        binary_presence[3:, 1] = 1
        return binary_presence

    @staticmethod
    def _fake_run_probe_failure(*args, **kwargs):
        raise RuntimeError("simulated probe inference failure")

    def _make_loader(self, tmp_path, run_probe):
        (tmp_path / "probing").mkdir(parents=True)
        (tmp_path / "probing" / "linear_probe.pt").touch()

        def path_func(model_name):
            return SimpleNamespace(
                probe_path=tmp_path / "probing",
                preds_path=tmp_path / "predictions",
            )

        loader = PredictionsLoader(
            vis_loader=self._FakeVisLoader(),
            path_func=path_func,
            models=["birdnet"],
            panel_selection=self._FakePanelSelection(),
            progress_bar=SimpleNamespace(value=0),
            loading_pane=SimpleNamespace(value="", name=""),
        )
        loader.load_classification = self._fake_load_classification
        visualize_predictions.prepare_probe_inference = (
            self._fake_prepare_probe
        )
        visualize_predictions.run_probe_inference = run_probe
        return loader

    def test_integrated_load_adds_overall_and_options(self, tmp_path):
        loader = self._make_loader(tmp_path, self._fake_run_probe_success)
        loader.get_data("birdnet", 0.5, clfier_type="Integrated")
        assert loader.binary_presence.shape[1] == 3  # 2 classes + overall
        assert "overall" in loader.class_dict
        assert "overall" in loader.panel_selection.options

    def test_failed_linear_run_clears_cache(self, tmp_path):
        loader = self._make_loader(tmp_path, self._fake_run_probe_success)
        loader.get_data("birdnet", 0.5, clfier_type="Integrated")
        assert loader.binary_presence is not None

        # The probe inference fails -> no stale state may be left behind.
        visualize_predictions.run_probe_inference = (
            self._fake_run_probe_failure
        )
        with pytest.raises(RuntimeError, match="simulated"):
            loader.get_data("birdnet", 0.5, clfier_type="Linear")
        assert loader.binary_presence is None
        assert loader.class_dict is None

        # A repeated Linear request must retry (not hit a stale cache).
        with pytest.raises(RuntimeError, match="simulated"):
            loader.get_data("birdnet", 0.5, clfier_type="Linear")

        # Switching back to the integrated classifier still works.
        loader.load_classification = self._fake_load_classification
        loader.get_data("birdnet", 0.5, clfier_type="Integrated")
        assert "overall" in loader.class_dict
        assert loader.binary_presence.shape[1] == 3

    def test_accumulate_data_falls_back_to_overall(self, tmp_path):
        loader = self._make_loader(tmp_path, self._fake_run_probe_success)
        loader.get_data("birdnet", 0.5, clfier_type="Integrated")
        # A species that is not part of the current classifier outputs must
        # not crash the heatmap; it falls back to the overall presence.
        accumulated = loader.accumulate_data("not_a_species", "day")
        assert accumulated.shape[0] == 24


class TestSetLegendDashboardStaysInBounds:
    """The dashboard (comparison) legend must stay inside the figure
    boundaries even when there are many labels (e.g. many species in the
    all-models comparison plot). Regression test: the legend used to be drawn
    outside the canvas, extending beyond the figure boundaries."""

    @pytest.mark.parametrize("num_labels", [7, 30, 60, 100])
    def test_legend_within_figure_bounds(self, num_labels):
        fig = plt.figure(figsize=(11, 5), dpi=100)
        ax = fig.subplots()
        handles, labels = [], []
        rng = np.random.default_rng(0)
        for i in range(num_labels):
            p = ax.scatter(rng.random(10), rng.random(10), label=f"species_{i}")
            handles.append(p)
            labels.append(f"species_{i}")

        fig, ax = set_legend(
            handles, labels, fig, ax, bool_plot_centroids=False, dashboard=True
        )
        fig.canvas.draw()
        fig_bounds = fig.get_window_extent()
        legend_bounds = fig.legends[0].get_window_extent()
        # Allow a 1px tolerance for rounding at the figure edges.
        assert legend_bounds.x0 >= fig_bounds.x0
        assert legend_bounds.x1 <= fig_bounds.x1 + 1
        assert legend_bounds.y0 >= fig_bounds.y0 - 1
        assert legend_bounds.y1 <= fig_bounds.y1 + 1
        plt.close(fig)


class TestReturnRowsColsExactFitGrid:
    """The comparison plot grid must not leave empty slots for the common
    model counts, so the individual plots stay as large as possible (no dead
    band in the figure). Regression test: up to three models always used a
    1x3 grid, leaving a third of the width empty when comparing two models."""

    @pytest.mark.parametrize(
        "num_models, expected",
        [
            (1, (1, 1)),
            (2, (1, 2)),
            (3, (1, 3)),
            (4, (2, 2)),
            (5, (2, 3)),
            (6, (2, 3)),
            (7, (3, 3)),
            (12, (3, 4)),
            (16, (4, 4)),
            (20, (4, 5)),
            (21, (5, 5)),
            (25, (5, 5)),
        ],
    )
    def test_grid_has_no_empty_slots(self, num_models, expected):
        rows, cols = return_rows_cols(num_models)
        assert (rows, cols) == expected
        # The grid must always be able to hold all models.
        assert rows * cols >= num_models

