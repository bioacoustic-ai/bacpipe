"""
Unit tests for the probing evaluation metrics in
``bacpipe.embedding_evaluation.probing.evaluate_probe``.
"""

import numpy as np
import pytest

from bacpipe.embedding_evaluation.probing.evaluate_probe import (
    accuracy_per_class,
    auc,
    compute_task_metrics,
    macro_accuracy,
    macro_f1,
    micro_accuracy,
    micro_f1,
)


class TestAccuracyPerClass:
    def test_returns_accuracy_per_label(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        label2index = {"a": 0, "b": 1}
        items_per_class = {"a": 2, "b": 2}
        acc = accuracy_per_class(
            y_true, y_pred, label2index, items_per_class
        )
        assert acc == {"a": 0.5, "b": 1.0}


class TestMacroAccuracy:
    def test_balanced_accuracy(self):
        assert macro_accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75

    def test_perfect_prediction(self):
        assert macro_accuracy([0, 1], [0, 1]) == 1.0


class TestMicroAccuracy:
    def test_fraction_of_correct(self):
        assert micro_accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75


class TestAuc:
    def test_binary_problem_uses_second_column(self):
        y_true = [0, 0, 1, 1]
        probs = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
        )
        assert auc(y_true, probs) == 1.0

    def test_multiclass_problem(self):
        y_true = [0, 1, 2]
        probs = np.array(
            [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]]
        )
        assert 0 <= auc(y_true, probs) <= 1

    def test_single_class_returns_nan(self):
        y_true = [0, 0, 0]
        probs = np.array([[0.9], [0.8], [0.7]])
        assert np.isnan(auc(y_true, probs))


class TestMacroF1:
    def test_averaged_f1(self):
        assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(
            0.7333, abs=1e-3
        )


class TestMicroF1:
    def test_global_f1(self):
        assert micro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.75)


class TestComputeTaskMetrics:
    def test_returns_all_metrics(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        probs = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]
        )
        label2index = {"a": 0, "b": 1}
        results = compute_task_metrics(
            y_pred, y_true, probs, label2index
        )
        assert set(results.keys()) == {
            "overall",
            "items_per_class",
            "per_class_accuracy",
        }
        assert set(results["overall"].keys()) == {
            "macro_accuracy",
            "micro_accuracy",
            "auc",
            "macro_f1",
            "micro_f1",
        }
        assert results["items_per_class"] == {"a": 2, "b": 2}
        assert results["per_class_accuracy"] == {"a": 1.0, "b": 1.0}

    def test_single_class_overall_drops_auc(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        probs = np.array([[0.9], [0.8], [0.7]])
        label2index = {"a": 0}
        results = compute_task_metrics(
            y_pred, y_true, probs, label2index
        )
        assert "auc" not in results["overall"]
