"""Type stub for :mod:`bacpipe.embedding_evaluation.clustering.cluster`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code
with Pylance) for the public ``bacpipe.clustering_pipeline``,
``bacpipe.run_clustering``, ``bacpipe.eval_clustering`` and
``bacpipe.eval_with_silhouette`` entry points. The docstrings are read
from the implementation in ``cluster.py``, so they are rendered on hover
without being duplicated here.

``clustering_pipeline`` accepts further configuration options through
``**kwargs`` that are sourced from ``bacpipe.config`` and
``bacpipe.settings`` at runtime. Declaring the most frequently used ones
explicitly here makes them discoverable while typing, without changing
any runtime behaviour.

This file has no effect on the runtime import of the package.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np


def convert_numpy_types(obj: Any) -> Any: ...


def save_clustering_performance(
    paths: SimpleNamespace,
    clusterings: dict[str, Any],
    metrics: dict[str, Any],
    label_column: Optional[str],
) -> None: ...


def run_clustering(
    embeds: np.ndarray,
    cluster_configs: dict[str, Any],
    label_column: Optional[str] = None,
    ground_truth: Any = ...,
) -> dict[str, Any]: ...


def eval_clustering(
    clusterings: dict[str, Any],
    ground_truth: Any = ...,
    embeds: Optional[np.ndarray] = None,
    metadata_labels: Optional[dict[str, Any]] = None,
    label_column: Optional[str] = None,
    *,
    evaluate_with_silhouette: Optional[bool] = None,
    **kwargs: Any,
) -> dict[str, Any]: ...


def eval_with_silhouette(
    embeds: np.ndarray,
    ground_truth: Any,
    metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]: ...


def get_clustering_models(clust_params: dict[str, Any]) -> dict[str, Any]: ...


def get_nr_of_clusters(
    labels: Any, clust_configs: dict[str, Any], **kwargs: Any
) -> dict[str, Any]: ...


def clustering_pipeline(
    model_name: str,
    ground_truth: dict[str, Any],
    embeds: np.ndarray,
    paths: Optional[SimpleNamespace] = None,
    overwrite: bool = True,
    label_column: str = ...,
    *,
    audio_dir: Optional[str | Path] = None,
    main_results_dir: Optional[str | Path] = None,
    evaluate_with_silhouette: Optional[bool] = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]: ...


def __getattr__(name: str) -> Any: ...
