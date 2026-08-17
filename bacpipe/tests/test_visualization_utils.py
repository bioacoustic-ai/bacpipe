"""
Unit tests for the visualization helpers in
``bacpipe.embedding_evaluation.visualization``.
"""

import numpy as np
import pandas as pd

from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    get_arrays_for_spectrogram_text,
    get_single_label_gt_labels,
)
from bacpipe.embedding_evaluation.visualization.visualize_predictions import (
    PredictionsLoader,
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
