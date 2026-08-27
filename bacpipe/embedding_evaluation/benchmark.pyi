"""Type stub for :mod:`bacpipe.embedding_evaluation.benchmark`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code
with Pylance) for the public ``bacpipe.benchmark`` entry point and the
label matching helpers it uses. The docstrings are read from the
implementation in ``benchmark.py``, so they are rendered on hover without
being duplicated here.

``benchmark`` accepts further configuration options through ``**kwargs``
that are sourced from ``bacpipe.config`` and ``bacpipe.settings`` at
runtime. Declaring the most frequently used ones explicitly here makes
them discoverable while typing, without changing any runtime behaviour.

This file has no effect on the runtime import of the package.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def clean_string(s: str) -> str: ...


def normalize_name(s: str) -> str: ...


def associate_labels_to_eBird_Codes(
    gt_species_cols: np.ndarray, gt_without_metadata: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]: ...


def associate_labels_regardless_of_puctuation(
    label2idx: dict[str, int],
    gt_without_metadata: pd.DataFrame,
    found: list[str],
    not_found: list[str],
) -> pd.DataFrame: ...


def associate_labels_regardless_of_spelling_and_substrings(
    label2idx: dict[str, int],
    found: list[str],
    gt_without_metadata: pd.DataFrame,
    not_found: list[str],
) -> pd.DataFrame: ...


def associate_ground_truth_and_prediction_labels(
    gt_species_cols: np.ndarray,
    label2idx: dict[str, int],
    gt_without_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[int], list[str]]: ...


def benchmark(
    model: str,
    dataset: str | Path,
    annotations_file: Optional[str] = None,
    CustomModel: Any = None,
    check_if_already_processed: bool = True,
    min_annotation_length: float = 0.01,
    overwrite: bool = True,
    *,
    classifier_threshold: Optional[float] = None,
    main_results_dir: Optional[str | Path] = None,
    use_folder_structure: Optional[bool] = None,
    device: Optional[str] = None,
    testing: bool = False,
    **kwargs: Any,
) -> dict[str, Any]: ...


def __getattr__(name: str) -> Any: ...
