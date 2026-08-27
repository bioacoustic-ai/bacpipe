"""Type stub for :mod:`bacpipe.core.experiment_manager`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code
with Pylance) for the public ``bacpipe.Loader`` API and for
``bacpipe.get_audio_files``, which is ``Loader.get_audio_files``. The
docstrings are read from the implementation in ``experiment_manager.py``,
so they are rendered on hover without being duplicated here.

``Loader`` receives a large number of configuration options through
``**kwargs`` that are sourced from ``bacpipe.config`` and
``bacpipe.settings`` at runtime, and sets the corresponding attributes
dynamically. Declaring the most frequently used ones explicitly here
makes them discoverable while typing, without changing any runtime
behaviour.

This file has no effect on the runtime import of the package.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Optional, overload

import numpy as np
import pandas as pd


def save_logs(**kwargs: Any) -> None: ...


class Loader:
    audio_dir: Path
    model_name: Optional[str]
    dim_reduction_model: str | bool
    use_folder_structure: bool
    testing: bool
    check_if_combination_exists: bool
    combination_already_exists: bool
    continue_incomplete_run: bool
    embed_suffix: str
    embed_dir: Path
    embed_parent_dir: Path
    dim_reduc_embed_dir: Path | bool
    files: list[Path]
    annot_files: list[Path]
    metadata_dict: dict[str, Any]
    paths: SimpleNamespace

    def __init__(
        self,
        audio_dir: str | Path,
        model_name: Optional[str] = None,
        check_if_combination_exists: bool = True,
        dim_reduction_model: str | bool = False,
        use_folder_structure: bool = False,
        testing: bool = False,
        *,
        main_results_dir: Optional[str | Path] = None,
        embed_parent_dir: Optional[str | Path] = None,
        only_embed_annotations: Optional[bool] = None,
        annotations_filename: Optional[str] = None,
        overwrite: Optional[bool] = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    @staticmethod
    def get_audio_files(
        audio_dir: str | Path,
        audio_suffixes: list[str] = ...,
        return_type: Literal["pathlib.Path"] = ...,
    ) -> list[Path]: ...
    @overload
    @staticmethod
    def get_audio_files(
        audio_dir: str | Path,
        audio_suffixes: list[str] = ...,
        return_type: Literal["str"] = ...,
    ) -> list[str]: ...

    def get_embedding_dir(self) -> Path: ...

    def read_embedding_file(self, file: str | Path) -> np.ndarray: ...

    @staticmethod
    def filter_df_by_file(
        audio_dir: str | Path,
        annots: pd.DataFrame,
        file_path: str | Path,
        sort_by_start: bool = True,
    ) -> pd.DataFrame: ...

    def embeddings(
        self, return_type: Literal["dict", "array"] = "dict"
    ) -> dict[str, np.ndarray] | np.ndarray: ...

    def get_preds_array(
        self,
        return_type: Literal["dict", "array", "dataframe"] = "dict",
        preds_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[int, str]] | pd.DataFrame: ...

    def get_annotations_parquet(
        self, preds_path: Optional[str | Path] = None, **kwargs: Any
    ) -> pd.DataFrame: ...

    def predictions(
        self,
        return_type: Literal["dict", "array", "dataframe"] = "dict",
        parent_dir: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[int, str]] | pd.DataFrame: ...

    def write_metadata_file(self) -> None: ...

    def update_files(self) -> None: ...

    def save_embedding_file(
        self, file: str | Path, embeds: np.ndarray
    ) -> None: ...

    def classifier_should_be_run(
        self,
        run_pretrained_classifier: bool = False,
        testing: bool = False,
        paths: Optional[SimpleNamespace] = None,
        **kwargs: Any,
    ) -> bool: ...

    def get_generated_annotation_files(
        self, preds_path: Optional[str | Path] = None
    ) -> tuple[list[str], Path]: ...

    # paths and settings are set as attributes dynamically
    def __getattr__(self, name: str) -> Any: ...


def replace_default_kwargs_with_user_kwargs(
    remove_keys: Optional[list[str]] = None, **kwargs: Any
) -> dict[str, Any]: ...


def return_reduced_dimensions(directory: str | Path) -> int: ...


def __getattr__(name: str) -> Any: ...
