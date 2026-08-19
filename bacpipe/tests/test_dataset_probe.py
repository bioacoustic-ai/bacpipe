"""
Unit tests for the probing dataset helpers in
``bacpipe.embedding_evaluation.probing.dataset_probe``.
"""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from bacpipe.embedding_evaluation.probing.dataset_probe import (
    ProbeDatasetLoader,
    generate_annotations_for_probing_task,
    probe_dataset_loader,
)


def make_class_df():
    return pd.DataFrame(
        {
            "predefined_set": ["train", "train", "train"],
            "label": ["a", "b", "a"],
            "index": [0, 1, 2],
        },
        index=[0, 1, 2],
    )


def make_embeds(n=3, dim=4):
    rng = np.random.RandomState(0)
    return rng.rand(n, dim)


class TestProbeDatasetLoader:
    def test_len_filters_by_set_name(self):
        df = make_class_df()
        embeds = make_embeds()
        dataset = ProbeDatasetLoader(
            class_df=df,
            embeds=embeds,
            label2index={"a": 0, "b": 1},
            set_name="train",
        )
        assert len(dataset) == 3

    def test_getitem_returns_embedding_and_label(self):
        df = make_class_df()
        embeds = make_embeds()
        dataset = ProbeDatasetLoader(
            class_df=df,
            embeds=embeds,
            label2index={"a": 0, "b": 1},
            set_name="train",
        )
        X, y = dataset[0]
        assert X.shape == (embeds.shape[1],)
        assert y in (0, 1)


class TestProbeDatasetLoaderFunction:
    def test_returns_dataloader_with_batches(self):
        df = make_class_df()
        embeds = make_embeds()
        loader = probe_dataset_loader(
            "train", df, embeds, {"a": 0, "b": 1}, batch_size=2
        )
        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape[1] == embeds.shape[1]


class TestGenerateAnnotationsForProbingTask:
    def _ground_truth(self):
        return pd.DataFrame(
            {
                "audiofilename": ["a.wav"] * 6,
                "start": [0, 1, 2, 3, 4, 5],
                "end": [1, 2, 3, 4, 5, 6],
                "simultaneous_labels": [1, 1, 1, 1, 1, 1],
                "sp_a": [1, 1, 1, 0, 0, 0],
                "sp_b": [0, 0, 0, 1, 1, 1],
            }
        )

    def test_generates_and_saves_annotations(self, tmp_path):
        labels_path = tmp_path / "labels"
        labels_path.mkdir()
        paths = SimpleNamespace(labels_path=labels_path)
        df = generate_annotations_for_probing_task(
            self._ground_truth(),
            paths,
            label_column="species",
            train_ratio=0.5,
            test_ratio=0.25,
        )
        assert set(df.label.unique()) == {"sp_a", "sp_b"}
        assert set(df.predefined_set.unique()) <= {"train", "test", "val"}
        assert (labels_path / "probing_dataframe.csv").exists()

    def test_loads_existing_file(self, tmp_path):
        labels_path = tmp_path / "labels"
        labels_path.mkdir()
        paths = SimpleNamespace(labels_path=labels_path)
        csv_path = labels_path / "probing_dataframe.csv"
        first = generate_annotations_for_probing_task(
            self._ground_truth(),
            paths,
            label_column="species",
            train_ratio=0.5,
            test_ratio=0.25,
        )
        loaded = generate_annotations_for_probing_task(
            self._ground_truth(),
            paths,
            label_column="species",
            dataset_csv_path=csv_path,
            train_ratio=0.5,
            test_ratio=0.25,
        )
        assert loaded.equals(first)
