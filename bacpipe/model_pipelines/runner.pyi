"""Type stub for :mod:`bacpipe.model_pipelines.runner`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code
with Pylance) for the public ``bacpipe.Embedder`` API and the
``Classifier`` it uses for models with a pretrained classifier head. The
docstrings are read from the implementation in ``runner.py``, so they are
rendered on hover without being duplicated here.

Both classes receive a large number of configuration options through
``**kwargs`` that are sourced from ``bacpipe.config`` and
``bacpipe.settings`` at runtime. Declaring the most frequently used ones
explicitly here makes them discoverable while typing, without changing
any runtime behaviour.

This file has no effect on the runtime import of the package.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch

from bacpipe.core.audio_processor import AudioHandler
from bacpipe.core.experiment_manager import Loader


class Embedder(AudioHandler):
    model_name: str
    loader: Optional[Loader]
    dim_reduction_model: str | bool
    nr_parallel_workers: Optional[int]
    classifier: Classifier

    def __init__(
        self,
        model_name: str,
        loader: Optional[Loader] = None,
        CustomModel: Any = None,
        dim_reduction_model: str | bool = False,
        audio_dir: Optional[str | Path] = None,
        *,
        device: Optional[str] = None,
        only_embed_annotations: Optional[bool] = None,
        annotations_filename: Optional[str] = None,
        annotations_df: Optional[pd.DataFrame] = None,
        nr_parallel_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> None: ...

    def init_dataloader(self, audio: Any) -> Any: ...

    def batch_inference(
        self,
        batched_samples: Any,
        callback: Optional[Callable[[float], Any]] = None,
    ) -> torch.Tensor | np.ndarray: ...

    def get_embeddings_for_audio(self, sample: Any) -> np.ndarray: ...

    def get_reduced_dimensionality_embeddings(
        self, embeds: np.ndarray
    ) -> np.ndarray: ...

    def run_dimensionality_reduction_pipeline(self) -> None: ...

    def generate_embeddings_from_audio_array(
        self, array_of_audios: np.ndarray | torch.Tensor
    ) -> list[np.ndarray]: ...

    def run_inference_pipeline_using_multithreading(self) -> None: ...

    def run_inference_pipeline_sequentially(self) -> None: ...

    def get_embeddings_from_model(
        self, sample: str | Path | np.ndarray
    ) -> np.ndarray: ...


class Classifier:
    model: Any
    model_name: str
    classifier_threshold: float
    save_raven_tables: bool
    max_labels_per_timestamp: Optional[int]
    paths: SimpleNamespace
    predictions: torch.Tensor
    df: pd.DataFrame
    cumulative_annotations: pd.DataFrame

    def __init__(
        self,
        model: Any,
        model_name: str,
        audio_dir: str | Path,
        main_results_dir: str | Path,
        classifier_threshold: float,
        use_folder_structure: bool = True,
        save_raven_tables: bool = False,
        **kwargs: Any,
    ) -> None: ...

    @staticmethod
    def filter_top_k_classifications(
        probabilities: np.ndarray | torch.Tensor,
        class_names: list[str] | np.ndarray,
        threshold: float,
        max_labels_per_timestamp: Optional[int] = None,
    ) -> dict[str, Any]: ...

    @staticmethod
    def make_classification_dict(
        probabilities: np.ndarray | torch.Tensor,
        classes: list[str] | np.ndarray,
        threshold: float,
        max_labels_per_timestamp: Optional[int] = None,
    ) -> dict[str, Any]: ...

    def classify(self, embeddings: np.ndarray | torch.Tensor) -> None: ...

    def save_annotation_table(
        self, loader_obj: Loader, **kwargs: Any
    ) -> None: ...

    def save_classifier_outputs(
        self, fileloader_obj: Loader, file: str | Path
    ) -> None: ...

    def save_Raven_table(
        self, file: str | Path, relative_parent_path: str | Path
    ) -> None: ...

    def run_default_classifier(self, loader: Loader) -> None: ...

    # settings and annotation bookkeeping are set as attributes dynamically
    def __getattr__(self, name: str) -> Any: ...


def __getattr__(name: str) -> Any: ...
