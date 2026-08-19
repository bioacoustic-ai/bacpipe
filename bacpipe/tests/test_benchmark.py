"""
Unit tests for the label-matching helpers in
``bacpipe.embedding_evaluation.benchmark``.
"""

import numpy as np
import pandas as pd
import pytest

from bacpipe.embedding_evaluation.benchmark import (
    associate_ground_truth_and_prediction_labels,
    associate_labels_regardless_of_puctuation,
    associate_labels_to_eBird_Codes,
    clean_string,
    normalize_name,
)


class TestCleanString:
    def test_lowercases_and_removes_punctuation(self):
        assert clean_string("Red-Shouldered Hawk") == "redshoulderedhawk"

    def test_removes_spaces(self):
        assert clean_string("Common Nighthawk") == "commonnighthawk"


class TestNormalizeName:
    def test_removes_all_non_alphanumerics(self):
        assert normalize_name("Eastern/Western Warbling Vireo") == (
            "easternwesternwarblingvireo"
        )

    def test_standardizes_grey_to_gray(self):
        assert normalize_name("Grey/Gray Hawk") == "graygrayhawk"


class TestAssociateLabelsToEBirdCodes:
    def test_converts_species_code_to_common_name(self):
        # object dtype so the assigned common name is not truncated
        gt_species_cols = np.array(["ostric2"], dtype=object)
        gt_without_metadata = pd.DataFrame({"ostric2": [1, 0]})
        cols, df = associate_labels_to_eBird_Codes(
            gt_species_cols, gt_without_metadata
        )
        assert cols[0] == "Common Ostrich"
        assert "Common Ostrich" in df.columns
        assert "ostric2" not in df.columns

    def test_unknown_code_is_left_unchanged(self):
        gt_species_cols = np.array(["Bird X"])
        gt_without_metadata = pd.DataFrame({"Bird X": [1, 0]})
        cols, df = associate_labels_to_eBird_Codes(
            gt_species_cols, gt_without_metadata
        )
        assert cols[0] == "Bird X"
        assert "Bird X" in df.columns


class TestAssociateLabelsRegardlessOfPuctuation:
    def test_renames_similar_columns(self):
        label2idx = {
            "Red-Shouldered Hawk": 0,
            "Common Nighthawk": 1,
        }
        gt_df = pd.DataFrame(
            {
                "Red Shouldered Hawk": [1, 0],
                "Common Nighthawk": [0, 1],
            }
        )
        found = []
        not_found = ["Red Shouldered Hawk", "Something Else"]
        gt_out = associate_labels_regardless_of_puctuation(
            label2idx, gt_df, found, not_found
        )
        assert "Red-Shouldered Hawk" in gt_out.columns
        assert "Red Shouldered Hawk" not in gt_out.columns
        assert "Red Shouldered Hawk" in found
        assert "Something Else" not in found
        assert "Common Nighthawk" in gt_out.columns

    def test_empty_similar_columns(self):
        label2idx = {"Red-Shouldered Hawk": 0}
        gt_df = pd.DataFrame({"Bird X": [1, 0]})
        found = []
        not_found = ["Red-Shouldered Hawk"]
        gt_out = associate_labels_regardless_of_puctuation(
            label2idx, gt_df, found, not_found
        )
        assert "Bird X" in gt_out.columns
        assert not_found == ["Red-Shouldered Hawk"]


class TestAssociateGroundTruthAndPredictionLabels:
    def test_exact_matches_are_aligned(self):
        gt_species_cols = np.array(
            ["Red-Shouldered Hawk", "Common Nighthawk"]
        )
        label2idx = {
            "Red-Shouldered Hawk": 0,
            "Common Nighthawk": 1,
        }
        gt_without_metadata = pd.DataFrame(
            {
                "Red-Shouldered Hawk": [1, 0],
                "Common Nighthawk": [0, 1],
            }
        )
        result = associate_ground_truth_and_prediction_labels(
            gt_species_cols, label2idx, gt_without_metadata
        )
        gt_aligned, shared_labels, shared_indices, not_found = result
        assert list(gt_aligned.columns) == list(label2idx.keys())
        assert shared_labels == ["Red-Shouldered Hawk", "Common Nighthawk"]
        assert shared_indices == [0, 1]
        assert not_found == []

    def test_punctuation_mismatch_is_resolved(self):
        gt_species_cols = np.array(["Red Shouldered Hawk"])
        label2idx = {"Red-Shouldered Hawk": 0}
        gt_without_metadata = pd.DataFrame({"Red Shouldered Hawk": [1, 0]})
        gt_aligned, shared_labels, shared_indices, not_found = (
            associate_ground_truth_and_prediction_labels(
                gt_species_cols, label2idx, gt_without_metadata
            )
        )
        assert shared_labels == ["Red-Shouldered Hawk"]
        assert shared_indices == [0]
        assert not_found == []

    def test_no_matches_returns_error_dict(self):
        gt_species_cols = np.array(["Bird X"])
        label2idx = {"Bird Y": 0}
        gt_without_metadata = pd.DataFrame({"Bird X": [1, 0]})
        result = associate_ground_truth_and_prediction_labels(
            gt_species_cols, label2idx, gt_without_metadata
        )
        assert result == {
            "error": "No ground truth classes have been found in the predictions."
        }

