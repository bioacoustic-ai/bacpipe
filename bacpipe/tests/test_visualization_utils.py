"""
Unit tests for the visualization helpers in
``bacpipe.embedding_evaluation.visualization``.
"""

from types import SimpleNamespace

import json
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    align_annotations_df_with_embeddings,
    generate_strings_for_spectrogram_text,
    get_arrays_for_spectrogram_text,
    get_boolean_array_for_annotated_embeddings,
    get_labels_for_plot,
    get_single_label_gt_labels,
    plot_embeddings_px,
    return_rows_cols,
)
from bacpipe.embedding_evaluation.visualization.visualize_spectrograms import (
    SpectrogramPlot,
    timestamps_match,
)
from bacpipe.embedding_evaluation.visualization.visualize_predictions import (
    PredictionsLoader,
    plot_per_class_results,
)
from bacpipe.embedding_evaluation.visualization.visualize import (
    plot_overview_results,
)
from bacpipe.embedding_evaluation.visualization.dashboard_utils import (
    _apply_view_ranges,
    _capture_view_ranges,
    _friendly_export_error,
    _static_export_figure,
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

    def test_large_counts_do_not_overflow(self):
        # Sums larger than 127 must not wrap to negative values: the heatmap
        # renders any negative cell as an empty (NaN) cell, which used to make
        # the "overall" view show gaps even though individual classes had data.
        presence = np.ones(200, dtype=np.int8)
        hours = np.array([5] * 200)
        accumulator = [(2024, 1, 1)] * 200
        heatmap = PredictionsLoader.transform_presence_into_hour_heatmap(
            presence, hours, accumulator
        )
        assert heatmap[5, 0] == 200


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
            le, "metadata_labels", lambda **kwargs: self._metadata_labels()
        )

    def test_marks_unannotated_embeddings_as_noise(self, monkeypatch):
        self._patch_metadata_labels(monkeypatch)
        is_noise = get_boolean_array_for_annotated_embeddings(
            self._ground_truth(), "insect459"
        )
        # a.wav@0 and b.wav@0 are annotated; a.wav@3 has no annotation
        assert is_noise.tolist() == [False, True, False]

    def test_returns_boolean_array(self, monkeypatch):
        self._patch_metadata_labels(monkeypatch)
        is_noise = get_boolean_array_for_annotated_embeddings(
            self._ground_truth(), "insect459"
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
        is_noise = get_boolean_array_for_annotated_embeddings(gt, "insect459")
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
        assert out == {"kmeans": [0, 1]}

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
            "model_name": "insect459",
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


class TestPlotPerClassResults:
    """The cross-model comparison plot must align each model's per-class
    values with the x-axis labels and stay consistent with the per-model
    probe plots (which sort classes by accuracy)."""

    def test_labels_match_values_and_use_accuracy_order(
        self, tmp_path, monkeypatch
    ):
        # Prevent the figure from being closed so we can inspect it.
        monkeypatch.setattr(plt, "close", lambda *a, **k: None)

        results = {
            "model_a": {
                # insertion order differs from alphabetical AND from
                # accuracy order on purpose, to catch misalignment
                "per_class_accuracy": {"cat": 0.9, "ant": 0.3, "bat": 0.6},
                "overall": {"macro_accuracy": 0.6},
            },
            "model_b": {
                "per_class_accuracy": {"bat": 0.7, "ant": 0.4, "cat": 0.8},
                "overall": {"macro_accuracy": 0.7},
            },
        }
        plot_per_class_results(
            tmp_path, "linear probing", ["model_a", "model_b"], results
        )

        ax = plt.gca()
        # model_b has the highest macro_accuracy -> reference model. Its
        # classes sorted by accuracy descending are cat, bat, ant.
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "cat",
            "bat",
            "ant",
        ]

        # The first scatter is the reference model (sorted first). Its points
        # must sit at x = 0, 1, 2 with the values for cat, bat, ant.
        offsets = ax.collections[0].get_offsets()
        np.testing.assert_allclose(offsets[:, 0], [0, 1, 2])
        np.testing.assert_allclose(offsets[:, 1], [0.8, 0.7, 0.4])

        # The figure was written to the plot path.
        assert len(list(tmp_path.glob("*.png"))) == 1

    def test_loads_results_from_disk_and_returns_figure(
        self, tmp_path
    ):
        # The dashboard calls with results=None, path_func=... and
        # return_fig=True; the results must then be read from the per-model
        # ``probe_results_*.json`` files instead of saving to disk.
        import json

        from types import SimpleNamespace

        models = ["model_a", "model_b"]
        probe_dirs = {}
        for m_idx, model in enumerate(models):
            probe_dir = tmp_path / model / "probing"
            probe_dir.mkdir(parents=True)
            probe_dirs[model] = probe_dir
            results = {
                "overall": {"macro_accuracy": 0.5 + m_idx / 10},
                "per_class_accuracy": {
                    "cat": 0.9,
                    "ant": 0.3 + m_idx / 10,
                    "bat": 0.6,
                },
            }
            with open(probe_dir / "probe_results_linear.json", "w") as f:
                json.dump(results, f)

        def path_func(model_name):
            return SimpleNamespace(probe_path=probe_dirs[model_name])

        fig = plot_per_class_results(
            None,
            "linear",
            models,
            results=None,
            path_func=path_func,
            return_fig=True,
        )

        assert type(fig).__name__ == "Figure"
        ax = fig.axes[0]
        # model_b has the higher macro_accuracy -> reference model; classes
        # sorted by accuracy descending are cat, bat, ant.
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "cat",
            "bat",
            "ant",
        ]

    def test_return_fig_caps_width_for_dashboard(self):
        # The dashboard requests ``return_fig=True`` so the figure is embedded
        # in a Panel accordion. For many classes the default width formula
        # (num_classes * 0.5) used to produce huge figures that extended
        # beyond the accordion; the dashboard figure must stay at a fixed,
        # modest width.
        results = {
            "model_a": {
                "per_class_accuracy": {
                    f"class_{i}": 0.9 for i in range(60)
                },
                "overall": {"macro_accuracy": 0.5},
            },
        }
        fig = plot_per_class_results(
            None, "linear", ["model_a"], results=results, return_fig=True
        )
        assert fig.get_size_inches()[0] == 12
        plt.close(fig)


class TestPredictionsLoaderCacheConsistency:
    """A failed load must not poison the PredictionsLoader cache. Otherwise
    the single model predictions tab crashes with a KeyError when switching
    between classifier types/models after a failed probe run."""

    class _FakeVisLoader:
        def __init__(self):
            n = 6
            self.embeds = {
                "insect459": {
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
        model, probe, threshold, return_binary_presence=True, callbacks=None,
        **kwargs,
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
                audio_dir=str(tmp_path / "audio"),
                main_results_dir=str(tmp_path / "results"),
            )

        loader = PredictionsLoader(
            vis_loader=self._FakeVisLoader(),
            path_func=path_func,
            models=["insect459"],
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
        loader.get_data("insect459", 0.5, clfier_type="Integrated")
        assert loader.binary_presence.shape[1] == 3  # 2 classes + overall
        assert "overall" in loader.class_dict
        assert "overall" in loader.panel_selection.options

    def test_failed_linear_run_clears_cache(self, tmp_path):
        loader = self._make_loader(tmp_path, self._fake_run_probe_success)
        loader.get_data("insect459", 0.5, clfier_type="Integrated")
        assert loader.binary_presence is not None

        # The probe inference fails -> no stale state may be left behind.
        visualize_predictions.run_probe_inference = (
            self._fake_run_probe_failure
        )
        with pytest.raises(RuntimeError, match="simulated"):
            loader.get_data("insect459", 0.5, clfier_type="Linear")
        assert loader.binary_presence is None
        assert loader.class_dict is None

        # A repeated Linear request must retry (not hit a stale cache).
        with pytest.raises(RuntimeError, match="simulated"):
            loader.get_data("insect459", 0.5, clfier_type="Linear")

        # Switching back to the integrated classifier still works.
        loader.load_classification = self._fake_load_classification
        loader.get_data("insect459", 0.5, clfier_type="Integrated")
        assert "overall" in loader.class_dict
        assert loader.binary_presence.shape[1] == 3

    def test_accumulate_data_falls_back_to_overall(self, tmp_path):
        loader = self._make_loader(tmp_path, self._fake_run_probe_success)
        loader.get_data("insect459", 0.5, clfier_type="Integrated")
        # A species that is not part of the current classifier outputs must
        # not crash the heatmap; it falls back to the overall presence.
        accumulated = loader.accumulate_data("not_a_species", "day")
        assert accumulated.shape[0] == 24

    def test_linear_probe_forwards_audio_and_results_dir(self, tmp_path):
        captured = {}

        def capturing_run_probe(model, probe, threshold, **kwargs):
            captured.update(kwargs)
            return self._fake_run_probe_success(model, probe, threshold)

        loader = self._make_loader(tmp_path, capturing_run_probe)
        loader.get_data("insect459", 0.5, clfier_type="Linear")

        # ``DashBoard.__init__`` consumes ``audio_dir``/``main_results_dir`` as
        # named parameters, so they are absent from ``self.kwargs``. The
        # predictions loader must re-supply them from the path helper;
        # otherwise ``run_probe_inference`` silently falls back to the
        # config/settings defaults and loads embeddings from the wrong place.
        assert captured["audio_dir"] == str(tmp_path / "audio")
        assert captured["main_results_dir"] == str(tmp_path / "results")


class TestPlotWidgetResponsive:
    """``DashBoardHelper.plot_widget`` must make ``return_fig=True`` plots
    fill the available accordion width instead of rendering at their natural
    (potentially very wide) size."""

    def test_return_fig_panel_stretches_to_accordion_width(self):
        import panel as pn

        from bacpipe.embedding_evaluation.visualization.dashboard_utils import (
            DashBoardHelper,
        )

        helper = object.__new__(DashBoardHelper)
        slider = pn.widgets.IntSlider(value=1, start=1, end=10)

        def fake_plot(x, return_fig=False):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            return fig

        panel = helper.plot_widget(fake_plot, x=slider, return_fig=True)
        assert panel.sizing_mode == "stretch_width"


class TestTimestampsMatch:
    def test_identical_timestamps_match(self):
        assert timestamps_match(0.0, 0.0)

    def test_tiny_float_rounding_differences_match(self):
        # the plot rounds start times to 4 decimals, the metadata file stores
        # the raw values -> small differences must not trigger a warning
        assert timestamps_match(0.6891, 0.6890625)

    def test_subsecond_shift_is_detected(self):
        # an int() based comparison would treat 0.5 and 0.8 as equal, but they
        # refer to different audio segments
        assert not timestamps_match(0.5, 0.8)

    def test_non_numeric_input_does_not_raise(self):
        assert not timestamps_match(None, 0.0)
        assert not timestamps_match("not-a-number", 0.0)


class TestCheckTimestampOfClickDataAgainstMetadata:
    """Tests for the spectrogram safety check that verifies the clicked
    point's timestamp against the metadata labels."""

    class _FakeVisLoader:
        def __init__(self):
            self.embeds = {}

        def get_data(self, model, label_by):
            self.embeds[model] = {
                "metadata": {
                    "sample_rate (Hz)": 48000,
                    "segment_length (samples)": 48000,
                }
            }

    class _FakeModelSelect:
        options = ["insect459"]

    def _make_spec_plot(self, tmp_path):
        def path_func(model_name):
            return SimpleNamespace(
                labels_path=tmp_path / model_name / "labels"
            )

        return SpectrogramPlot(
            audio_dir=tmp_path,
            loader=self._FakeVisLoader(),
            model_name=self._FakeModelSelect(),
            panel_static_text=None,
            paths=path_func,
        )

    def _write_csv(self, spec, tmp_path, starts):
        labels_path = tmp_path / "insect459" / "labels"
        labels_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"start": starts}).to_csv(
            labels_path / "metadata_labels.csv", index=False
        )

    def test_matching_timestamp_logs_no_warning(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        self._write_csv(spec, tmp_path, [0.0, 1.0])
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "insect459", 0, 0.0
            )
        assert not any("do not match" in r.message for r in caplog.records)

    def test_mismatching_timestamp_logs_warning(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        self._write_csv(spec, tmp_path, [0.0, 1.0])
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "insect459", 0, 5.0
            )
        assert any("do not match" in r.message for r in caplog.records)

    def test_missing_metadata_file_does_not_raise(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "insect459", 0, 0.0
            )
        assert any("No metadata_labels file" in r.message for r in caplog.records)

    def test_parquet_fallback(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        labels_path = tmp_path / "insect459" / "labels"
        labels_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"start": [0.0, 1.0]}).to_parquet(
            labels_path / "metadata_labels.parquet"
        )
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "insect459", 1, 1.0
            )
        assert not any("do not match" in r.message for r in caplog.records)




    def test_idx_out_of_range_does_not_raise(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        self._write_csv(spec, tmp_path, [0.0, 1.0])
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "insect459", 99, 0.0
            )
        assert any("Could not find a metadata label" in r.message for r in caplog.records)

    def test_none_model_name_does_not_raise(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                None, 0, 0.0
            )
        assert len(caplog.records) == 0

    def test_remove_noise_disables_check(self, tmp_path, caplog):
        """Noise-filtered embeddings remap the click indices, so the
        timestamp check must be skipped entirely when remove_noise is set."""
        spec = self._make_spec_plot(tmp_path)
        self._write_csv(spec, tmp_path, [0.0, 1.0])

        class _FakeWidget:
            value = True

        # both a plain bool and a widget-like object should disable the check
        for remove_noise in (True, _FakeWidget()):
            spec.remove_noise_widget = remove_noise
            caplog.clear()
            with caplog.at_level("WARNING", logger="bacpipe"):
                spec.check_timestamp_of_click_data_against_metadata(
                    "insect459", 0, 5.0
                )
            # timestamps deliberately mismatch, yet no warning is logged
            assert len(caplog.records) == 0

    def test_metadata_starts_are_cached(self, tmp_path, caplog, monkeypatch):
        spec = self._make_spec_plot(tmp_path)
        self._write_csv(spec, tmp_path, [0.0, 1.0])

        real_read_csv = pd.read_csv
        calls = []

        def counting_read_csv(*args, **kwargs):
            calls.append(args)
            return real_read_csv(*args, **kwargs)

        monkeypatch.setattr("pandas.read_csv", counting_read_csv)
        spec.check_timestamp_of_click_data_against_metadata("insect459", 0, 0.0)
        spec.check_timestamp_of_click_data_against_metadata("insect459", 1, 1.0)
        # the metadata file is only read once per model, not once per click
        assert len(calls) == 1


class TestPlotEmbeddingsPxCustomData:
    """Regression test for the spectrogram click data contract: the embedding
    plot must attach exactly the 8 customdata columns that
    ``SpectrogramPlot.update_spectrogram`` unpacks."""

    def _embeds(self):
        return {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "timestamp": [0.0, 1.0],
            "index": [0, 1],
            "metadata": {
                "model_name": "insect459",
                "audio_files": ["a.wav", "a.wav"],
                "segment_length (samples)": 48000,
                "sample_rate (Hz)": 48000,
            },
        }

    def _labels(self):
        return {"time_of_day": ["12-00-00", "12-00-01"]}

    def test_customdata_contains_eight_columns_in_click_order(self):
        fig = plot_embeddings_px(
            self._embeds(), self._labels(), label_by="time_of_day"
        )
        # categorical labels are split into one trace per label, so collect
        # the customdata of all traces
        rows = []
        for trace in fig.data:
            rows.extend(np.asarray(trace.customdata, dtype=object).tolist())
        customdata = np.asarray(rows, dtype=object)
        assert customdata.shape == (2, 8)
        # column 7 is the model name, column 6 the numeric label id
        assert set(customdata[:, 7]) == {"insect459"}
        assert set(customdata[:, 6]) == {0, 1}

def _aligned_embeds(n=6):
    """Embeddings whose values all encode the index of the embedding."""
    return {
        "x": [float(i) for i in range(n)],
        "y": [float(i) * 10 for i in range(n)],
        "z": None,
        "timestamp": [float(i) for i in range(n)],
        "durations": [1.0] * n,
        "index": list(range(n)),
        "metadata": {
            "audio_files": [f"file_{i}.wav" for i in range(n)],
            "segment_length (samples)": 48000,
            "sample_rate (Hz)": 48000,
            "model_name": "insect459",
            "embed_dir": "/tmp/does/not/matter",
        },
    }


# deliberately unsorted so that sorting the plot dataframe reorders the rows
_SPECIES = ["zebra", "aardvark", "zebra", "manatee", "aardvark", "manatee"]


class TestGenerateStringsForSpectrogramText:
    """The json strings shown with the spectrogram must hold the label values
    of *their own* embedding.

    Regressions:
    1. String label values were dropped (``convert_numpy_types`` returned
       ``None`` for anything that was not a numpy type), so the spectrogram
       text showed ``null`` instead of e.g. the species.
    2. Label arrays with a different length than the embeddings were zipped
       together with the correct ones, which truncated the strings to the
       shortest array and shifted the labels of every point.
    """

    def _labels(self, n=6):
        return {
            "time_of_day": [f"12-00-0{i}" for i in range(n)],
            "call_type": [f"call_{i}" for i in range(n)],
        }

    def test_string_label_values_are_kept(self):
        embeds = _aligned_embeds()
        strings = generate_strings_for_spectrogram_text(
            self._labels(), "time_of_day", {"x": embeds["x"]}, embeds
        )
        assert len(strings) == len(embeds["x"])
        parsed = [json.loads(s) for s in strings]
        assert [p["call_type"] for p in parsed] == [
            f"call_{i}" for i in range(6)
        ]

    def test_numpy_label_values_are_serializable(self):
        embeds = _aligned_embeds()
        labels = self._labels()
        labels["kmeans"] = np.arange(6, dtype=np.int32)
        strings = generate_strings_for_spectrogram_text(
            labels, "time_of_day", {"x": embeds["x"]}, embeds
        )
        assert [json.loads(s)["kmeans"] for s in strings] == list(range(6))

    def test_one_string_per_embedding_without_additional_labels(self):
        embeds = _aligned_embeds(n=3)
        labels = {"time_of_day": ["12-00-00", "12-00-01", "12-00-02"]}
        strings = generate_strings_for_spectrogram_text(
            labels, "time_of_day", {"x": embeds["x"]}, embeds
        )
        # one (empty) json object per embedding, so that the strings can be
        # attached to the plot dataframe without a length mismatch
        assert strings == [json.dumps({})] * 3

    def test_label_arrays_of_wrong_length_are_dropped(self, caplog):
        embeds = _aligned_embeds()
        labels = self._labels()
        labels["broken"] = ["only", "two"]
        with caplog.at_level("WARNING", logger="bacpipe"):
            strings = generate_strings_for_spectrogram_text(
                labels, "time_of_day", {"x": embeds["x"]}, embeds
            )
        assert len(strings) == 6
        parsed = [json.loads(s) for s in strings]
        # the correctly sized array keeps all of its values ...
        assert [p["call_type"] for p in parsed] == [
            f"call_{i}" for i in range(6)
        ]
        # ... and the mismatched one is omitted instead of truncating the rest
        assert all("broken" not in p for p in parsed)
        assert "broken" in caplog.text

    def test_extra_label_arrays_are_added(self):
        embeds = _aligned_embeds()
        strings = generate_strings_for_spectrogram_text(
            self._labels(),
            "time_of_day",
            {"x": embeds["x"]},
            embeds,
            extra_label_arrays={"annotator": [f"ann_{i}" for i in range(6)]},
        )
        assert [json.loads(s)["annotator"] for s in strings] == [
            f"ann_{i}" for i in range(6)
        ]

    def test_extra_label_arrays_of_wrong_length_are_dropped(self):
        embeds = _aligned_embeds()
        strings = generate_strings_for_spectrogram_text(
            self._labels(),
            "time_of_day",
            {"x": embeds["x"]},
            embeds,
            extra_label_arrays={"annotator": ["ann_0"]},
        )
        assert len(strings) == 6
        assert all("annotator" not in json.loads(s) for s in strings)


class TestPlotEmbeddingsPxAlignment:
    """Every scatter point must carry its own coordinates, label, hover data
    and custom data.

    The plot dataframe is sorted by label to get an alphabetically ordered
    legend. All per-point information therefore has to live in that dataframe
    (so that entire rows are moved), instead of being read from the unsorted
    embedding arrays afterwards. Otherwise a clicked point shows the
    spectrogram and the labels of a different embedding.
    """

    def _labels(self, n=6):
        return {
            "species": list(_SPECIES),
            "call_type": [f"call_{i}" for i in range(n)],
        }

    def test_customdata_and_coordinates_stay_aligned(self):
        embeds = _aligned_embeds()
        fig = plot_embeddings_px(embeds, self._labels(), label_by="species")
        seen = []
        for trace in fig.data:
            customdata = np.asarray(trace.customdata, dtype=object)
            assert len(customdata) == len(trace.x)
            for x, y, row in zip(trace.x, trace.y, customdata):
                idx = int(round(x))
                seen.append(idx)
                assert y == pytest.approx(idx * 10)
                # audiofilename, start, end and idx of the clicked point
                assert row[0] == f"file_{idx}.wav"
                assert row[1] == pytest.approx(idx)
                assert row[2] == pytest.approx(idx + 1)
                assert row[3] == idx
                # label of the point, of its trace and of the spectrogram text
                assert row[4] == _SPECIES[idx]
                assert trace.name == _SPECIES[idx]
                assert json.loads(row[5])["call_type"] == f"call_{idx}"
        # no point was lost or duplicated by the sorting
        assert sorted(seen) == list(range(6))

    def test_traces_are_sorted_by_label(self):
        fig = plot_embeddings_px(
            _aligned_embeds(), self._labels(), label_by="species"
        )
        assert [trace.name for trace in fig.data] == sorted(set(_SPECIES))

    def test_hovertemplate_reads_the_click_data_columns(self):
        fig = plot_embeddings_px(
            _aligned_embeds(), self._labels(), label_by="species"
        )
        # the hover text must read the same customdata columns that
        # SpectrogramPlot.update_spectrogram unpacks
        for column, position in [
            ("audiofilename", 0),
            ("start", 1),
            ("end", 2),
            ("label", 4),
        ]:
            assert (
                f"{column}=%{{customdata[{position}]}}"
                in fig.data[0].hovertemplate
            )

    def test_continuous_colorbar_plot_stays_aligned(self, monkeypatch):
        # high cardinality labels are plotted as a single trace with a
        # colorbar, the values must still belong to their own point
        monkeypatch.setattr(settings, "max_nr_categories", 2)
        embeds = _aligned_embeds()
        labels = {"file": [f"file_{i}.wav" for i in range(6)]}
        fig = plot_embeddings_px(embeds, labels, label_by="file")
        trace = fig.data[0]
        customdata = np.asarray(trace.customdata, dtype=object)
        for x, row in zip(trace.x, customdata):
            idx = int(round(x))
            assert row[0] == f"file_{idx}.wav"
            assert row[4] == f"file_{idx}.wav"


class TestAlignAnnotationsDfWithEmbeddings:
    """A user provided ``annotations_df`` must never break the alignment of
    the plot: its values are matched to the embedded segments instead of being
    attached by position, and the columns needed for plotting always come from
    the embeddings.
    """

    def _plot_df(self, n=6):
        return pd.DataFrame(
            {
                "x": [float(i) for i in range(n)],
                "y": [float(i) for i in range(n)],
                "label": list(_SPECIES[:n]),
                "audiofilename": [f"file_{i}.wav" for i in range(n)],
                "start": [float(i) for i in range(n)],
                "end": [float(i) + 1 for i in range(n)],
                "idx": list(range(n)),
            }
        )

    def _embeds(self, model="insect459"):
        embeds = _aligned_embeds()
        embeds["metadata"]["model_name"] = model
        return embeds

    def test_values_are_matched_by_audiofilename_and_start(self):
        df = self._plot_df()
        # shuffled annotations: attaching them by position would misalign them
        order = [3, 0, 5, 1, 4, 2]
        annots = pd.DataFrame(
            {
                "audiofilename": [f"file_{i}.wav" for i in order],
                "start": [float(i) for i in order],
                "annotator": [f"ann_{i}" for i in order],
            }
        )
        aligned = align_annotations_df_with_embeddings(
            df, annots, self._embeds()
        )
        assert aligned["annotator"] == [f"ann_{i}" for i in range(6)]

    def test_plot_columns_are_not_overwritten(self):
        df = self._plot_df()
        annots = pd.DataFrame(
            {
                "audiofilename": [f"file_{i}.wav" for i in range(6)],
                "start": [float(i) for i in range(6)],
                # the annotations hold different values for the plot columns
                "end": [99.0] * 6,
                "label": ["wrong"] * 6,
                "annotator": [f"ann_{i}" for i in range(6)],
            }
        )
        aligned = align_annotations_df_with_embeddings(
            df, annots, self._embeds()
        )
        assert list(aligned.keys()) == ["annotator"]

    def test_only_the_rows_of_the_current_model_are_used(self):
        df = self._plot_df()
        annots = pd.DataFrame(
            {
                "model": ["insect459"] * 6 + ["other_model"] * 6,
                "audiofilename": [f"file_{i}.wav" for i in range(6)] * 2,
                "start": [float(i) for i in range(6)] * 2,
                "annotator": [f"ann_{i}" for i in range(6)] + ["other"] * 6,
            }
        )
        aligned = align_annotations_df_with_embeddings(
            df, annots, self._embeds()
        )
        assert aligned["annotator"] == [f"ann_{i}" for i in range(6)]

    def test_annotations_of_another_model_only_warn(self, caplog):
        df = self._plot_df()
        annots = pd.DataFrame(
            {
                "model": ["other_model"] * 6,
                "audiofilename": [f"file_{i}.wav" for i in range(6)],
                "start": [float(i) for i in range(6)],
                "annotator": [f"ann_{i}" for i in range(6)],
            }
        )
        with caplog.at_level("WARNING", logger="bacpipe"):
            aligned = align_annotations_df_with_embeddings(
                df, annots, self._embeds()
            )
        assert aligned == {}
        assert "no rows for" in caplog.text

    def test_simultaneous_annotations_do_not_duplicate_embeddings(self):
        df = self._plot_df()
        # two annotations for the same segment (multi-label annotations)
        annots = pd.DataFrame(
            {
                "audiofilename": [f"file_{i}.wav" for i in range(6)] * 2,
                "start": [float(i) for i in range(6)] * 2,
                "annotator": [f"ann_{i}" for i in range(6)] * 2,
            }
        )
        aligned = align_annotations_df_with_embeddings(
            df, annots, self._embeds()
        )
        assert len(aligned["annotator"]) == len(df)

    def test_partially_annotated_embeddings_are_filled_up(self):
        df = self._plot_df()
        annots = pd.DataFrame(
            {
                "audiofilename": ["file_2.wav"],
                "start": [2.0],
                "annotator": ["ann_2"],
            }
        )
        aligned = align_annotations_df_with_embeddings(
            df, annots, self._embeds()
        )
        assert len(aligned["annotator"]) == len(df)
        assert aligned["annotator"][2] == "ann_2"
        assert pd.isna(aligned["annotator"][0])

    def test_unmatched_annotations_only_warn(self, caplog):
        df = self._plot_df()
        annots = pd.DataFrame(
            {
                "audiofilename": ["not_embedded.wav"],
                "start": [123.0],
                "annotator": ["ann"],
            }
        )
        with caplog.at_level("WARNING", logger="bacpipe"):
            aligned = align_annotations_df_with_embeddings(
                df, annots, self._embeds()
            )
        assert len(aligned["annotator"]) == len(df)
        assert "None of the rows" in caplog.text

    def _folder_structure_df(self, sep, n=6):
        """Plot dataframe of a dataset with a folder structure.

        The file names of the embeddings come from ``metadata.yml``, which
        stores the path relative to the audio directory with the separators
        of the operating system the embeddings were created on.
        """
        df = self._plot_df(n=n)
        df["audiofilename"] = [
            f"audio{sep}FewShot{sep}file_{i}.wav" for i in range(n)
        ]
        return df

    def _folder_structure_annots(self, sep, n=6):
        return pd.DataFrame(
            {
                "audiofilename": [
                    f"audio{sep}FewShot{sep}file_{i}.wav" for i in range(n)
                ],
                "start": [float(i) for i in range(n)],
                "annotator": [f"ann_{i}" for i in range(n)],
            }
        )

    def test_windows_embeddings_match_posix_annotations(self):
        # embeddings created on windows: metadata.yml holds backslashes while
        # the annotations file uses forward slashes
        aligned = align_annotations_df_with_embeddings(
            self._folder_structure_df("\\"),
            self._folder_structure_annots("/"),
            self._embeds(),
        )
        assert aligned["annotator"] == [f"ann_{i}" for i in range(6)]

    def test_posix_embeddings_match_windows_annotations(self):
        # the other way around: the annotations were written on windows while
        # the embeddings were created on linux
        aligned = align_annotations_df_with_embeddings(
            self._folder_structure_df("/"),
            self._folder_structure_annots("\\"),
            self._embeds(),
        )
        assert aligned["annotator"] == [f"ann_{i}" for i in range(6)]

    def test_path_objects_as_filenames_are_matched(self):
        # a user can build the file name column from Path objects, which
        # carry the separators of their own operating system
        annots = self._folder_structure_annots("/")
        annots["audiofilename"] = [
            Path(name) for name in annots.audiofilename
        ]
        aligned = align_annotations_df_with_embeddings(
            self._folder_structure_df("/"), annots, self._embeds()
        )
        assert aligned["annotator"] == [f"ann_{i}" for i in range(6)]

    def test_positional_fallback_requires_one_row_per_embedding(self, caplog):
        df = self._plot_df()
        annots = pd.DataFrame({"annotator": ["ann_0", "ann_1"]})
        with caplog.at_level("WARNING", logger="bacpipe"):
            aligned = align_annotations_df_with_embeddings(
                df, annots, self._embeds()
            )
        assert aligned == {}
        assert "2 rows while 6" in caplog.text

    def test_positional_fallback_for_matching_row_count(self):
        df = self._plot_df()
        annots = pd.DataFrame({"annotator": [f"ann_{i}" for i in range(6)]})
        aligned = align_annotations_df_with_embeddings(
            df, annots, self._embeds()
        )
        assert aligned["annotator"] == [f"ann_{i}" for i in range(6)]

    def test_none_annotations_df_returns_no_columns(self):
        # ``annotations_df`` is an optional kwarg, so None is the normal case
        # and must not be treated as a dataframe (``None.columns``).
        aligned = align_annotations_df_with_embeddings(
            self._plot_df(), None, self._embeds()
        )
        assert aligned == {}

    def test_non_dataframe_annotations_df_only_warns(self, caplog):
        # e.g. a path to the annotations file instead of the dataframe: warn
        # and plot without the additional columns instead of crashing
        with caplog.at_level("WARNING", logger="bacpipe"):
            aligned = align_annotations_df_with_embeddings(
                self._plot_df(), "annotations.csv", self._embeds()
            )
        assert aligned == {}
        assert "has to be a pandas.DataFrame" in caplog.text



class TestPlotEmbeddingsPxWithAnnotationsDf:
    """``annotations_df`` is a user kwarg that is forwarded to the embedding
    plot. Its columns are displayed with the spectrogram of a clicked point
    and must not change the click data contract (8 customdata columns).
    """

    def _annotations(self, n=6):
        order = list(reversed(range(n)))
        return pd.DataFrame(
            {
                "audiofilename": [f"file_{i}.wav" for i in order],
                "start": [float(i) for i in order],
                "end": [float(i) + 1 for i in order],
                "annotator": [f"ann_{i}" for i in order],
            }
        )

    def _labels(self):
        return {"species": list(_SPECIES)}

    def _customdata(self, fig):
        rows = []
        for trace in fig.data:
            rows.extend(np.asarray(trace.customdata, dtype=object).tolist())
        return np.asarray(rows, dtype=object)

    def test_windows_filenames_are_matched_to_posix_annotations(self):
        # mirrors the dashboard on windows: metadata.yml stores the paths
        # relative to the audio_dir with backslashes while the annotations
        # csv of the user holds forward slashes
        embeds = _aligned_embeds()
        embeds["metadata"]["audio_files"] = [
            f"audio\\FewShot\\file_{i}.wav" for i in range(6)
        ]
        annots = self._annotations()
        annots["audiofilename"] = [
            "audio/" + name.replace("file_", "FewShot/file_")
            for name in annots.audiofilename
        ]
        fig = plot_embeddings_px(
            embeds,
            self._labels(),
            label_by="species",
            annotations_df=annots,
        )
        annotators = [
            json.loads(row[5])["annotator"] for row in self._customdata(fig)
        ]
        assert len(annotators) == 6
        assert all(name.startswith("ann_") for name in annotators)

    def test_click_data_keeps_eight_columns(self):
        fig = plot_embeddings_px(
            _aligned_embeds(),
            self._labels(),
            label_by="species",
            annotations_df=self._annotations(),
        )
        assert self._customdata(fig).shape == (6, 8)

    def test_annotation_values_belong_to_their_own_point(self):
        fig = plot_embeddings_px(
            _aligned_embeds(),
            self._labels(),
            label_by="species",
            annotations_df=self._annotations(),
        )
        for trace in fig.data:
            customdata = np.asarray(trace.customdata, dtype=object)
            for x, row in zip(trace.x, customdata):
                idx = int(round(x))
                assert json.loads(row[5])["annotator"] == f"ann_{idx}"

    def test_unusable_annotations_do_not_break_the_plot(self, caplog):
        # no audiofilename/start columns and a different number of rows
        annots = pd.DataFrame({"annotator": ["ann_0", "ann_1"]})
        with caplog.at_level("WARNING", logger="bacpipe"):
            fig = plot_embeddings_px(
                _aligned_embeds(),
                self._labels(),
                label_by="species",
                annotations_df=annots,
            )
        assert self._customdata(fig).shape == (6, 8)
        assert "Continuing without" in caplog.text


class TestDashboardExportHelpers:
    """The Save Figure button must re-apply the zoomed view and export a
    non-webgl figure so kaleido reliably renders colorbars/legends."""

    def test_capture_view_ranges_list_form(self):
        ranges = _capture_view_ranges(
            {"xaxis.range": [-1, 2], "yaxis.range": [3, 4]}
        )
        assert ranges == {"xaxis": (-1, 2), "yaxis": (3, 4)}

    def test_capture_view_ranges_indexed_form(self):
        ranges = _capture_view_ranges(
            {
                "xaxis.range[0]": -1,
                "xaxis.range[1]": 2,
                "yaxis.range[0]": 3,
                "yaxis.range[1]": 4,
            }
        )
        assert ranges == {"xaxis": (-1, 2), "yaxis": (3, 4)}

    def test_capture_view_ranges_empty_when_no_ranges(self):
        assert _capture_view_ranges(None) == {}
        assert _capture_view_ranges({"xaxis.autorange": True}) == {}

    def test_apply_view_ranges_disables_autorange(self):
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
        _apply_view_ranges(fig, {"xaxis": (-1, 2), "yaxis": (3, 4)})
        assert list(fig.layout.xaxis.range) == [-1, 2]
        assert list(fig.layout.yaxis.range) == [3, 4]
        assert fig.layout.xaxis.autorange is False
        assert fig.layout.yaxis.autorange is False

    def test_static_export_converts_scattergl_to_scatter(self):
        import plotly.express as px

        fig = px.scatter(
            x=[0, 1, 2],
            y=[0, 1, 2],
            color=[0, 1, 2],
            render_mode="webgl",
        )
        assert {t.type for t in fig.data} == {"scattergl"}

        converted = _static_export_figure(fig)
        assert {t.type for t in converted.data} == {"scatter"}
        # the colorbar/coloraxis survives the round-trip
        assert converted.layout.coloraxis is not None
        # the original figure is left untouched
        assert {t.type for t in fig.data} == {"scattergl"}


class TestFriendlyExportError:
    """The Save Figure button must surface friendly, actionable messages instead
    of raw kaleido/plotly exceptions when a browser or kaleido is unavailable."""

    def test_returns_hint_for_kaleido_chrome_not_found(self):
        from kaleido.errors import ChromeNotFoundError

        message = _friendly_export_error(ChromeNotFoundError())
        assert message is not None
        assert "Chrome" in message and "Edge" in message

    def test_returns_hint_for_plotly_chrome_runtime_error(self):
        # plotly wraps ChromeNotFoundError into a RuntimeError with this text
        message = _friendly_export_error(
            RuntimeError("Kaleido requires Google Chrome to be installed.")
        )
        assert message is not None
        assert "kaleido_get_chrome" in message

    def test_returns_hint_for_browser_launch_failures(self):
        from kaleido.errors import BrowserFailedError, ChromeNotFoundError

        errors = [ChromeNotFoundError(), BrowserFailedError()]
        try:
            from choreographer.errors import BrowserDepsError

            # BrowserDepsError is a subclass of BrowserFailedError
            errors.append(BrowserDepsError())
        except ImportError:  # pragma: no cover - choreographer is a kaleido dep
            pass

        for exc in errors:
            assert _friendly_export_error(exc) is not None

    def test_returns_hint_for_missing_kaleido_package(self):
        # plotly raises ValueError (or a raw ModuleNotFoundError) when kaleido
        # is not installed
        assert (
            _friendly_export_error(
                ValueError(
                    'Image export using the "kaleido" engine requires the '
                    "Kaleido package"
                )
            )
            is not None
        )
        assert (
            _friendly_export_error(ModuleNotFoundError("No module named 'kaleido'"))
            is not None
        )

    def test_returns_none_for_unrelated_errors(self):
        assert _friendly_export_error(ValueError("boom")) is None
        assert _friendly_export_error(TypeError("boom")) is None

    def test_does_not_crash_when_kaleido_unimportable(self, monkeypatch):
        import sys

        # simulate an environment where the kaleido package cannot be imported
        monkeypatch.setitem(sys.modules, "kaleido", None)
        monkeypatch.setitem(sys.modules, "kaleido.errors", None)

        # the message-based fallback must still fire without importing kaleido
        message = _friendly_export_error(
            RuntimeError("Kaleido requires Google Chrome to be installed.")
        )
        assert message is not None
        assert "Chrome" in message


class TestDashboardEmbeddingPanelKwargs:
    """``DashBoard.embedding_panel`` forwards user kwargs to the plot functions
    without colliding with the explicit ``dashboard``/``dashboard_idx`` flags.

    Regression: ``bacpipe.play`` merges ``config.yaml``/``settings.yaml`` into
    the dashboard kwargs, and ``config.yaml`` contains a ``dashboard`` key.
    A naive ``**self.kwargs`` splat next to ``dashboard=True`` raised
    "TypeError: got multiple values for keyword argument 'dashboard'".
    """

    def test_dashboard_kwarg_does_not_collide_with_explicit_flag(self):
        from bacpipe.embedding_evaluation.visualization.dashboard import (
            DashBoard,
        )

        dash = object.__new__(DashBoard)
        # mirrors a real ``bacpipe.play`` run where the merged config/settings
        # dict (including the ``dashboard`` flag) lands in the dashboard kwargs
        dash.kwargs = {
            "dashboard": True,
            "models": ["model_a"],
            "overwrite": False,
            "already_computed": False,
        }
        dash.interactive_embedding_plot = False
        dash.vis_loader = object()
        dash.model_select = {0: "model_a"}
        dash.label_select = {0: "time_of_day"}
        dash.metadata_label_keys = ["time_of_day"]
        dash.noise_select = {}
        dash.ground_truth = None
        dash.dim_reduction_model = "umap"
        dash.embed_save_button = {0: None}
        dash.embed_notification = {0: None}

        captured = {}

        def fake_init_plot(p_type, plot_func, widget_idx, **kwargs):
            captured.update(kwargs)
            captured["plot_func"] = plot_func
            return "plot"

        dash.init_plot = fake_init_plot

        # must not raise "got multiple values for keyword argument 'dashboard'"
        dash.embedding_panel(0)

        assert captured["dashboard"] is True
        assert captured["dashboard_idx"] == 0
        assert captured["model_name"] == "model_a"
        assert captured["label_by"] == "time_of_day"
        # ``metadata_label_keys`` is a named ``DashBoard.__init__`` parameter and
        # therefore absent from ``self.kwargs``; it must still reach the plot.
        assert captured["metadata_label_keys"] == ["time_of_day"]
        # user kwargs are still forwarded, just without the colliding keys
        assert captured["overwrite"] is False
        assert captured["models"] == ["model_a"]



class TestDashboardInitClustConfigs:
    """``DashBoard.__init__`` must not read ``self.kwargs`` before it is set.

    Regression: the clustering label block used ``self.kwargs`` while
    ``self.kwargs = kwargs`` is only assigned at the end of ``__init__``.
    Building the dashboard after clustering had run (the normal
    ``bacpipe.play`` flow) therefore raised
    ``AttributeError: 'DashBoard' object has no attribute 'kwargs'``.
    """

    def test_clustered_results_do_not_raise_attribute_error(
        self, tmp_path, monkeypatch
    ):
        import bacpipe.embedding_evaluation.label_embeddings as le
        from bacpipe.embedding_evaluation.visualization.dashboard import (
            DashBoard,
        )

        clust_dir = tmp_path / "clustering"
        clust_dir.mkdir()
        (clust_dir / "model_a_kmeans.npy").write_bytes(b"")
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        def fake_paths(model_name):
            return SimpleNamespace(
                preds_path=tmp_path / "predictions",
                clust_path=clust_dir,
                labels_path=labels_dir,
                plot_path=tmp_path / "plots",
            )

        # ``le.get_paths`` is a module global that only exists once
        # ``make_set_paths_func`` has been called, so tolerate a missing attr.
        monkeypatch.setattr(le, "get_paths", fake_paths, raising=False)
        monkeypatch.setattr(
            le, "make_set_paths_func", lambda *a, **k: fake_paths
        )

        dash = DashBoard(
            model_names=["model_a"],
            audio_dir=str(tmp_path),
            main_results_dir=tmp_path,
            metadata_label_keys=["label"],
            evaluation_task="linear",
            dim_reduction_model=None,
            dim_reduc_parent_dir="dim_reduced",
            clust_configs={"kmeans": {"name": "kmeans", "bool": True}},
        )

        assert "kmeans" in dash.label_by



class TestDashboardExplicitArgsWinOverDefaults:
    """Values passed to ``DashBoard.__init__`` must win over the defaults.

    ``replace_default_kwargs_with_user_kwargs`` re-inserts the
    ``config.yaml``/``settings.yaml`` defaults into ``kwargs``, while the
    explicitly passed values are bound to the named parameters. Reading the
    ``kwargs`` value first therefore silently replaced e.g. a custom
    ``main_results_dir`` by the default ``"bacpipe_results"``, so the
    dashboard looked for the embeddings in the wrong directory and raised
    "No embeddings found for model ...".
    """

    def _dashboard(self, tmp_path, monkeypatch, **kwargs):
        import bacpipe.embedding_evaluation.label_embeddings as le
        from bacpipe.embedding_evaluation.visualization.dashboard import (
            DashBoard,
        )

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        clust_dir = tmp_path / "clustering"
        clust_dir.mkdir()

        def fake_paths(model_name):
            return SimpleNamespace(
                preds_path=tmp_path / "predictions",
                clust_path=clust_dir,
                labels_path=labels_dir,
                plot_path=tmp_path / "plots",
            )

        captured = {}

        def fake_make_set_paths_func(
            audio_dir, main_results_dir=None, *args, **kw
        ):
            captured["main_results_dir"] = main_results_dir
            return fake_paths

        monkeypatch.setattr(le, "get_paths", fake_paths, raising=False)
        monkeypatch.setattr(
            le, "make_set_paths_func", fake_make_set_paths_func
        )

        dash = DashBoard(
            model_names=["model_a"],
            audio_dir=str(tmp_path),
            **kwargs,
        )
        return dash, captured

    def test_passed_main_results_dir_is_not_replaced_by_default(
        self, tmp_path, monkeypatch
    ):
        results_dir = str(tmp_path / "my_results")
        dash, captured = self._dashboard(
            tmp_path,
            monkeypatch,
            main_results_dir=results_dir,
            metadata_label_keys=["time_of_day"],
            evaluation_task=["clustering"],
            dim_reduction_model="pca",
            dim_reduc_parent_dir="my_dim_reduced",
        )

        assert dash.main_results_dir == results_dir
        # the paths of the whole dashboard are built from it
        assert captured["main_results_dir"] == results_dir
        assert dash.dim_reduction_model == "pca"
        assert dash.dim_reduc_parent_dir == "my_dim_reduced"
        assert dash.evaluation_task == ["clustering"]
        assert dash.metadata_label_keys == ["time_of_day"]
        # the resolved keys must not linger in the kwargs that are splatted
        # into the plot functions next to the explicit arguments
        for key in (
            "main_results_dir",
            "dim_reduction_model",
            "dim_reduc_parent_dir",
            "evaluation_task",
            "metadata_label_keys",
        ):
            assert key not in dash.kwargs

    def test_omitted_arguments_fall_back_to_the_defaults(
        self, tmp_path, monkeypatch
    ):
        # nothing passed -> the config.yaml/settings.yaml defaults are used
        dash, captured = self._dashboard(tmp_path, monkeypatch)

        assert dash.main_results_dir == bacpipe.settings.main_results_dir
        assert captured["main_results_dir"] == (
            bacpipe.settings.main_results_dir
        )
        assert dash.dim_reduction_model == bacpipe.config.dim_reduction_model


class TestVisualizeUsingDashboardCustomModels:
    """``visualize_using_dashboard`` has to accept the plural ``CustomModels``.

    Regression: the kwarg was forwarded verbatim to ``confirm_model_name``,
    which only understands the singular ``CustomModel``. Serving the dashboard
    for a custom model therefore raised
    "NameError: The provided model_name='my_model' is not included in the
    supported_models".
    """

    def _patch_dashboard(self, monkeypatch):
        """Replace the dashboard and the server with recorders."""
        import panel as pn
        from bacpipe.embedding_evaluation.visualization import dashboard as db

        captured = {}

        class FakeDashBoard:
            def __init__(self, model_names, **kwargs):
                captured["models"] = model_names
                self.app = "app"

            def build_layout(self):
                captured["built"] = True

        monkeypatch.setattr(db, "DashBoard", FakeDashBoard)
        monkeypatch.setattr(
            pn.template.BootstrapTemplate,
            "show",
            lambda self, **kwargs: captured.setdefault("shown", True),
        )
        return captured

    def test_custom_models_list_is_accepted(self, monkeypatch):
        captured = self._patch_dashboard(monkeypatch)

        class MyModel:
            pass

        bacpipe.visualize_using_dashboard(
            models=["insect459", "my_model"],
            CustomModels=[None, MyModel],
            audio_dir="bacpipe/tests/test_data",
        )

        assert captured["models"] == ["insect459", "my_model"]
        assert captured["built"] and captured["shown"]

    def test_unknown_model_without_custom_class_still_raises(
        self, monkeypatch
    ):
        # the name check must stay in place for the models bacpipe ships
        self._patch_dashboard(monkeypatch)

        class MyModel:
            pass

        with pytest.raises(NameError):
            bacpipe.visualize_using_dashboard(
                models=["not_a_real_model", "my_model"],
                CustomModels=[None, MyModel],
                audio_dir="bacpipe/tests/test_data",
            )

    def test_custom_models_length_must_match_models(self, monkeypatch):
        self._patch_dashboard(monkeypatch)

        class MyModel:
            pass

        with pytest.raises(AssertionError):
            bacpipe.visualize_using_dashboard(
                models=["insect459", "my_model"],
                CustomModels=[MyModel],
                audio_dir="bacpipe/tests/test_data",
            )


def _write_ground_truth_csv(path, rows):
    """Write a minimal ground truth csv file (one row per annotation)."""
    pd.DataFrame(
        {
            "audiofilename": [r[0] for r in rows],
            "start": [r[1] for r in rows],
            "end": [r[1] + 3.0 for r in rows],
            "simultaneous_labels": [1] * len(rows),
            "sp_a": [int(r[2] == "sp_a") for r in rows],
            "sp_b": [int(r[2] == "sp_b") for r in rows],
        }
    ).to_csv(path, index=False)


class TestGetLabelsForPlotGroundTruthMode:
    """Ground truth files are cached per ``only_embed_annotations`` mode, so
    both ``ground_truth_species.csv`` and
    ``ground_truth_species_only_annotated.csv`` can exist side by side.

    Regressions:
    1. Both files were loaded, although only the file of the active mode has
       one row per embedded segment. The labels of the other mode did not
       align with the embeddings.
    2. The ``_only_annotated`` suffix ended up in the label name, so the
       dashboard offered a ``species_only_annotated`` entry instead of
       ``species``.
    """

    # metadata labels of the two modes: all segments vs. annotated only
    _ALL_SEGMENTS = [("a.wav", 0.0), ("a.wav", 3.0), ("b.wav", 0.0)]
    _ANNOTATED = [("a.wav", 0.0), ("b.wav", 0.0)]

    def _labels_dir(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        _write_ground_truth_csv(
            labels_dir / "ground_truth_species.csv",
            [("a.wav", 0.0, "sp_a"), ("b.wav", 0.0, "sp_b")],
        )
        _write_ground_truth_csv(
            labels_dir / "ground_truth_species_only_annotated.csv",
            [("a.wav", 0.0, "sp_a"), ("b.wav", 0.0, "sp_b")],
        )
        return labels_dir

    def _patch(self, tmp_path, monkeypatch, segments):
        import bacpipe.embedding_evaluation.label_embeddings as le

        labels_dir = self._labels_dir(tmp_path)
        clust_dir = tmp_path / "clustering"
        clust_dir.mkdir()

        def fake_paths(model_name):
            return SimpleNamespace(
                labels_path=labels_dir, clust_path=clust_dir
            )

        monkeypatch.setattr(le, "get_paths", fake_paths, raising=False)
        monkeypatch.setattr(
            le,
            "_get_metadata_labels",
            lambda *a, **k: {
                "audio_file_name": [f for f, _ in segments],
                "time_of_day": ["12-00-00"] * len(segments),
            },
        )
        monkeypatch.setattr(
            le,
            "metadata_labels",
            lambda **k: pd.DataFrame(
                {
                    "audio_file_name": [f for f, _ in segments],
                    "start": [s for _, s in segments],
                    "end": [s + 3.0 for _, s in segments],
                }
            ),
        )

    def test_full_mode_uses_the_unsuffixed_file(self, tmp_path, monkeypatch):
        self._patch(tmp_path, monkeypatch, self._ALL_SEGMENTS)
        labels, bool_noise = get_labels_for_plot(
            model_name="insect459", only_embed_annotations=False
        )
        assert "species" in labels
        # the file of the other mode must not add a second label key
        assert "species_only_annotated" not in labels
        # one label per embedded segment, unannotated segments are noise
        assert list(labels["species"]) == ["sp_a", "noise", "sp_b"]
        assert bool_noise.tolist() == [False, True, False]

    def test_annotated_mode_uses_the_suffixed_file(
        self, tmp_path, monkeypatch
    ):
        self._patch(tmp_path, monkeypatch, self._ANNOTATED)
        labels, bool_noise = get_labels_for_plot(
            model_name="insect459", only_embed_annotations=True
        )
        assert "species" in labels
        assert "species_only_annotated" not in labels
        # only the annotated segments were embedded -> no noise label
        assert list(labels["species"]) == ["sp_a", "sp_b"]
        assert bool_noise.tolist() == [False, False]



class TestGetGroundTruthLabelNames:
    """The label dropdown of the dashboard is built from the ground truth
    files of *all* selected models and of the active
    ``only_embed_annotations`` mode.

    Regressions:
    1. Only the ground truth files of the first model were checked, so ground
       truth labels of the other models were missing in the dropdown.
    2. The file names were used as label names, which added a
       ``species_only_annotated`` entry that no model can resolve.
    """

    def _labels_dir(self, tmp_path, name, files):
        labels_dir = tmp_path / name / "labels"
        labels_dir.mkdir(parents=True)
        for file_name, rows in files.items():
            _write_ground_truth_csv(labels_dir / file_name, rows)
        return labels_dir

    def _patch_paths(self, monkeypatch, dirs):
        import bacpipe.embedding_evaluation.label_embeddings as le

        monkeypatch.setattr(
            le,
            "get_paths",
            lambda model_name: SimpleNamespace(labels_path=dirs[model_name]),
            raising=False,
        )

    def test_all_models_contribute_their_labels(self, tmp_path, monkeypatch):
        from bacpipe.embedding_evaluation.visualization.dashboard import (
            get_ground_truth_label_names,
        )

        rows = [("a.wav", 0.0, "sp_a")]
        dirs = {
            "model_a": self._labels_dir(
                tmp_path, "model_a", {"ground_truth_species.csv": rows}
            ),
            "model_b": self._labels_dir(
                tmp_path, "model_b", {"ground_truth_call_type.csv": rows}
            ),
        }
        self._patch_paths(monkeypatch, dirs)
        names = get_ground_truth_label_names(["model_a", "model_b"])
        assert sorted(set(names)) == ["call_type", "species"]

    def test_only_the_files_of_the_active_mode_are_used(
        self, tmp_path, monkeypatch
    ):
        from bacpipe.embedding_evaluation.visualization.dashboard import (
            get_ground_truth_label_names,
        )

        rows = [("a.wav", 0.0, "sp_a")]
        dirs = {
            "model_a": self._labels_dir(
                tmp_path,
                "model_a",
                {
                    "ground_truth_species.csv": rows,
                    "ground_truth_species_only_annotated.csv": rows,
                },
            )
        }
        self._patch_paths(monkeypatch, dirs)

        # both modes must yield the same label name, exactly once
        assert get_ground_truth_label_names(["model_a"]) == ["species"]
        assert get_ground_truth_label_names(
            ["model_a"], only_embed_annotations=True
        ) == ["species"]

    def test_label_names_with_underscores_stay_intact(
        self, tmp_path, monkeypatch
    ):
        from bacpipe.embedding_evaluation.visualization.dashboard import (
            get_ground_truth_label_names,
        )

        rows = [("a.wav", 0.0, "sp_a")]
        dirs = {
            "model_a": self._labels_dir(
                tmp_path,
                "model_a",
                {"ground_truth_call_type_only_annotated.csv": rows},
            )
        }
        self._patch_paths(monkeypatch, dirs)
        names = get_ground_truth_label_names(
            ["model_a"], only_embed_annotations=True
        )
        assert names == ["call_type"]

    def test_files_of_other_types_are_ignored(self, tmp_path, monkeypatch):
        from bacpipe.embedding_evaluation.visualization.dashboard import (
            get_ground_truth_label_names,
        )

        labels_dir = self._labels_dir(
            tmp_path, "model_a", {"ground_truth_species.csv": [("a.wav", 0.0, "sp_a")]}
        )
        (labels_dir / "ground_truth_species.txt").write_text("not a label")
        self._patch_paths(monkeypatch, {"model_a": labels_dir})
        assert get_ground_truth_label_names(["model_a"]) == ["species"]

