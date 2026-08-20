"""Type stub for :mod:`bacpipe.core.workflows`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code with
Pylance) for the public ``bacpipe.*`` pipeline entry points such as
``bacpipe.play`` and ``bacpipe.run_pipeline_for_single_model``.

Many of these functions accept a large number of configuration options through
``**kwargs`` that are sourced from ``bacpipe.config`` and ``bacpipe.settings``
at runtime. Declaring the most frequently used ones explicitly here makes
them discoverable while typing, without changing any runtime behaviour.

This file has no effect on the runtime import of the package.
"""

from pathlib import Path
from typing import Any, Optional

from bacpipe.core.experiment_manager import Loader

# Re-exported in the runtime module for internal use.
from bacpipe.embedding_evaluation.visualization.visualize import (
    visualise_results_across_models,
)
from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    EmbedAndLabelLoader,
    plot_comparison,
    plot_embeddings,
)


def play(
    bool_save_logs: bool = False,
    *,
    models: Optional[list[str]] = None,
    audio_dir: Optional[str | Path] = None,
    dim_reduction_model: Optional[str] = None,
    check_if_already_processed: Optional[bool] = None,
    check_if_already_dim_reduced: Optional[bool] = None,
    only_embed_annotations: Optional[bool] = None,
    annotations_filename: Optional[str] = None,
    overwrite: Optional[bool] = None,
    device: Optional[str] = None,
    use_folder_structure: Optional[bool] = None,
    already_computed: bool = False,
    dashboard: bool = False,
    **kwargs: Any,
) -> None: ...


def ensure_models_exist(
    model_base_path: Optional[str | Path] = None,
    model_names: Optional[list[str]] = None,
    repo_id: str = "vskode/bacpipe_models",
    CustomModel: Any = None,
    CustomModels: Any = None,
) -> Path: ...


def confirm_model_name(model_name: str, **kwargs: Any) -> str: ...


def get_model_names(
    models: Optional[list[str]],
    audio_dir: str | Path,
    main_results_dir: str | Path,
    embed_parent_dir: str | Path,
    already_computed: bool = False,
    **kwargs: Any,
) -> list[str]: ...


def evaluation_with_settings_already_exists(
    audio_dir: str | Path,
    dim_reduction_model: Optional[str],
    models: list[str],
    testing: bool = False,
    **kwargs: Any,
) -> bool: ...


def run_pipeline_for_models(
    models: list[str],
    audio_dir: str | Path,
    dim_reduction_model: Optional[str],
    check_if_already_processed: Optional[bool] = None,
    check_if_already_dim_reduced: Optional[bool] = None,
    **kwargs: Any,
) -> dict[str, Loader]: ...


def model_specific_evaluation(
    loader_dict: dict[str, Loader],
    evaluation_task: str | list[str],
    probe_configs: Optional[list[dict[str, Any]]] = None,
    dim_reduction_model: Any = False,
    **kwargs: Any,
) -> None: ...


def cross_model_evaluation(
    audio_dir: str | Path,
    evaluation_task: str | list[str],
    models: list[str],
    dim_reduction_model: Optional[str] = None,
    **kwargs: Any,
) -> None: ...


def run_pipeline_for_single_model(
    model_name: str,
    audio_dir: str | Path,
    dim_reduction_model: str = "None",
    check_if_already_processed: bool = False,
    check_if_already_dim_reduced: bool = True,
    testing: bool = False,
    **kwargs: Any,
) -> Loader: ...


def generate_embeddings(
    model_name: str,
    audio_dir: str | Path,
    avoid_pipelined_gpu_inference: bool = False,
    check_if_already_processed: Optional[bool] = None,
    check_if_already_dim_reduced: Optional[bool] = None,
    **kwargs: Any,
) -> Loader: ...


def visualize_using_dashboard(
    models: list[str],
    dashboard_port: int = 5006,
    dashboard_address: str = "localhost",
    dashboard_websocket_origin: Any = False,
    **kwargs: Any,
) -> None: ...
