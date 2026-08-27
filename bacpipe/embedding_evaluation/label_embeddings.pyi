"""Type stub for :mod:`bacpipe.embedding_evaluation.label_embeddings`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code
with Pylance) for the public ``bacpipe.MetadataLabelMaker``,
``bacpipe.metadata_labels``, ``bacpipe.ground_truth_by_model``,
``bacpipe.make_set_paths_func`` and ``bacpipe.get_dt_filename`` entry
points, as well as for the label handling helpers they build on. The
docstrings are read from the implementation in ``label_embeddings.py``, so
they are rendered on hover without being duplicated here.

Several of these functions accept configuration options through
``**kwargs`` that are sourced from ``bacpipe.config`` and
``bacpipe.settings`` at runtime. Declaring the most frequently used ones
explicitly here makes them discoverable while typing, without changing
any runtime behaviour.

This file has no effect on the runtime import of the package.
"""

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd

ONLY_ANNOTATED_SUFFIX: str


def deduplicate_annotation_pairs(df: pd.DataFrame) -> pd.DataFrame: ...


def unique_start_end_annot_pairs(df: pd.DataFrame) -> pd.DataFrame: ...


class MetadataLabelMaker:
    paths: SimpleNamespace
    model: str
    metadata_label_keys: list[str]
    only_embed_annotations: bool
    df: pd.DataFrame
    metadata: dict[str, Any]
    nr_embeds_per_file: list[int]
    nr_embeds_total: int
    metadata_label_dict: dict[str, Any]

    def __init__(
        self,
        paths: SimpleNamespace,
        model: str,
        metadata_label_keys: list[str],
        *,
        only_embed_annotations: Optional[bool] = None,
        annotations_df: Optional[pd.DataFrame] = None,
        annotations_filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None: ...

    def generate(self) -> None: ...

    def get_datetimes(self) -> None: ...

    def time_of_day(self) -> None: ...

    def week_of_year(self) -> None: ...

    def day_of_year(self) -> None: ...

    def continuous_timestamp(self) -> None: ...

    def parent_directory(self) -> None: ...

    def audio_file_name(self) -> None: ...

    def default_classifier(self) -> None: ...

    def fill_remaining_labels(self, df: pd.DataFrame) -> pd.DataFrame: ...

    # one attribute per metadata label is created dynamically
    def __getattr__(self, name: str) -> Any: ...


def make_set_paths_func(
    audio_dir: str | Path,
    main_results_dir: Optional[str | Path] = None,
    dim_reduc_parent_dir: str | Path = "dim_reduced_embeddings",
    testing: bool = False,
    **kwargs: Any,
) -> Callable[..., SimpleNamespace]: ...


def get_dim_reduc_path_func(
    model_name: str, dim_reduction_model: str = "umap", **kwargs: Any
) -> Path: ...


def ensure_windoof_path_to_posix(path: str | Path) -> str: ...


def load_metadata_file(folder: str | Path) -> dict[str, Any]: ...


def get_ground_truth(
    model_name: str,
    file_path: Optional[str | Path] = None,
    return_type: Literal["dataframe", "array"] = "dataframe",
) -> pd.DataFrame | dict[str, Any]: ...


def strip_only_annotated_suffix(name: str) -> str: ...


def select_ground_truth_files_for_mode(
    ground_truth_files: list[str | Path],
    only_embed_annotations: bool = False,
) -> list[Path]: ...


def get_dt_filename(file: str | Path) -> dt.datetime: ...


def model_specific_embedding_path(
    path: str | Path,
    model: str,
    dim_reduction_model: Optional[str] = None,
    **kwargs: Any,
) -> Path: ...


def metadata_labels(
    audio_dir: Optional[str | Path] = None,
    model: Optional[str] = None,
    paths: Optional[SimpleNamespace] = None,
    overwrite: bool = True,
    return_type: Literal["dataframe", "dict"] = "dataframe",
    *,
    metadata_label_keys: Optional[list[str]] = None,
    only_embed_annotations: Optional[bool] = None,
    annotations_df: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> pd.DataFrame | dict[str, Any]: ...


def fetch_annotation_file(
    audio_dir: str | Path,
    annotations_filename: str,
    paths: Optional[SimpleNamespace],
) -> pd.DataFrame: ...


def filter_annotations(
    label_df: pd.DataFrame,
    main_label_column: Optional[str],
    min_label_occurrences: int,
    bool_filter_labels: bool,
) -> Optional[pd.DataFrame]: ...


def load_labels_and_build_dict(
    paths: Optional[SimpleNamespace],
    annotations_filename: str,
    audio_dir: str | Path,
    audio_files: list[str | Path] = ...,
    bool_filter_labels: bool = True,
    min_label_occurrences: int = 150,
    main_label_column: Optional[str] = None,
    testing: bool = False,
    **kwargs: Any,
) -> pd.DataFrame: ...


def fit_labels_to_embedding_timestamps(
    df: pd.DataFrame,
    df_fitted_gt: pd.DataFrame,
    num_embeds: int,
    segment_s: float,
    label_column: Optional[str] = None,
    only_embed_annotations: bool = False,
    **kwargs: Any,
) -> pd.DataFrame: ...


def build_ground_truth_labels_by_file(
    ind: int,
    model: str,
    num_embeds: int,
    segment_s: float,
    metadata: dict[str, Any],
    all_labels: pd.DataFrame,
    label_df: Optional[pd.DataFrame] = None,
    label_column: Optional[str] = None,
    filename_array: Optional[np.ndarray] = None,
    only_embed_annotations: bool = False,
    **kwargs: Any,
) -> pd.DataFrame: ...


def filter_df_by_filename(
    df_to_filter: pd.DataFrame,
    file_name: str,
    filename_array: Optional[np.ndarray] = None,
    file_name_column: str = "audiofilename",
    model: Optional[str] = None,
) -> pd.DataFrame: ...


def create_Raven_annotation_table(
    df: pd.DataFrame, label_column: str, high_freq: int = 1000
) -> pd.DataFrame: ...


def ensure_file_names_match(
    metadata: dict[str, Any], ind: int, file: str | Path, model: str
) -> None: ...


def initialize_ground_truth_df(
    label_df: pd.DataFrame, label_column: str
) -> pd.DataFrame: ...


def get_filename_array(
    label_df: pd.DataFrame, label_column: str
) -> np.ndarray: ...


def collect_ground_truth_labels(
    files: list[str | Path],
    model: str,
    segment_s: float,
    metadata: dict[str, Any],
    label_df: pd.DataFrame,
    label_column: str,
    **kwargs: Any,
) -> pd.DataFrame: ...


def assign_global_get_paths_function(
    audio_dir: str | Path, **kwargs: Any
) -> None: ...


def ground_truth_by_model(
    model: str,
    audio_dir: str | Path,
    label_df: Optional[pd.DataFrame] = None,
    label_column: str = "label:species",
    paths: Optional[SimpleNamespace] = None,
    annotations_filename: str = "annotations.csv",
    only_embed_annotations: bool = False,
    overwrite: bool = True,
    bool_filter_labels: bool = False,
    *,
    min_annotation_length: Optional[float] = None,
    min_label_occurrences: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame | dict[str, Any]: ...


def ensure_audio_files(
    found_audio_files: list[str | Path],
    annotated_audio_files: list[str],
    audio_dir: str | Path,
) -> list[str]: ...


def get_files_if_no_embeds(
    audio_dir: str | Path,
    model: str,
    label_df: Optional[pd.DataFrame] = None,
    only_embed_annotations: bool = False,
) -> tuple[list[str], float, dict[str, Any]]: ...


def __getattr__(name: str) -> Any: ...
