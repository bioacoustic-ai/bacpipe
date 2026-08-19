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
    plot_embeddings_px,
    return_rows_cols,
)
from bacpipe.embedding_evaluation.visualization.visualize_spectrograms import (
    SpectrogramPlot,
    timestamps_match,
)
from bacpipe.embedding_evaluation.visualization.visualize_predictions import (
    PredictionsLoader,
)
from bacpipe.embedding_evaluation.visualization.visualize import (
    plot_overview_results,
)
from bacpipe.embedding_evaluation.visualization.dashboard_utils import (
    _friendly_export_error,
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
        options = ["birdnet"]

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
        labels_path = tmp_path / "birdnet" / "labels"
        labels_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"start": starts}).to_csv(
            labels_path / "metadata_labels.csv", index=False
        )

    def test_matching_timestamp_logs_no_warning(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        self._write_csv(spec, tmp_path, [0.0, 1.0])
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "birdnet", 0, 0.0
            )
        assert not any("do not match" in r.message for r in caplog.records)

    def test_mismatching_timestamp_logs_warning(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        self._write_csv(spec, tmp_path, [0.0, 1.0])
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "birdnet", 0, 5.0
            )
        assert any("do not match" in r.message for r in caplog.records)

    def test_missing_metadata_file_does_not_raise(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "birdnet", 0, 0.0
            )
        assert any("No metadata_labels file" in r.message for r in caplog.records)

    def test_parquet_fallback(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        labels_path = tmp_path / "birdnet" / "labels"
        labels_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"start": [0.0, 1.0]}).to_parquet(
            labels_path / "metadata_labels.parquet"
        )
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "birdnet", 1, 1.0
            )
        assert not any("do not match" in r.message for r in caplog.records)




    def test_idx_out_of_range_does_not_raise(self, tmp_path, caplog):
        spec = self._make_spec_plot(tmp_path)
        self._write_csv(spec, tmp_path, [0.0, 1.0])
        with caplog.at_level("WARNING", logger="bacpipe"):
            spec.check_timestamp_of_click_data_against_metadata(
                "birdnet", 99, 0.0
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
                    "birdnet", 0, 5.0
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
        spec.check_timestamp_of_click_data_against_metadata("birdnet", 0, 0.0)
        spec.check_timestamp_of_click_data_against_metadata("birdnet", 1, 1.0)
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
                "model_name": "birdnet",
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
        assert set(customdata[:, 7]) == {"birdnet"}
        assert set(customdata[:, 6]) == {0, 1}

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
            default_label_keys=["label"],
            evaluation_task="linear",
            dim_reduction_model=None,
            dim_reduc_parent_dir="dim_reduced",
            clust_configs={"kmeans": {"name": "kmeans", "bool": True}},
        )

        assert "kmeans" in dash.label_by

