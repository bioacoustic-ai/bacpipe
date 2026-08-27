"""Type stub for :mod:`bacpipe.embedding_evaluation.probing.probe`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code
with Pylance) for the public ``bacpipe.probing_pipeline`` entry point. The
docstrings are read from the implementation in ``probe.py``, so they are
rendered on hover without being duplicated here.

``probing_pipeline`` accepts further configuration options through
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
import pandas as pd


def embeds_array_where_single_label(
    embeds: np.ndarray,
    ground_truth: dict[str, Any],
    bool_noise: bool,
    df: pd.DataFrame,
    **kwargs: Any,
) -> tuple[pd.DataFrame, np.ndarray]: ...


def probing_pipeline(
    model_name: str,
    ground_truth: pd.DataFrame | dict[str, Any],
    embeds: np.ndarray,
    paths: Optional[SimpleNamespace] = None,
    name: str = "linear",
    overwrite: bool = True,
    label_column: str = ...,
    dataset_csv_path: str = "probing_dataframe.csv",
    *,
    audio_dir: Optional[str | Path] = None,
    main_results_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, int], dict[str, Any]]: ...


def __getattr__(name: str) -> Any: ...
