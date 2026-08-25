"""Type stub for :mod:`bacpipe.embedding_evaluation.probing.inference_probe`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code
with Pylance) for the public ``bacpipe.prepare_probe_inference`` and
``bacpipe.run_probe_inference`` entry points. The docstrings are read from
the implementation in ``inference_probe.py``, so they are rendered on
hover without being duplicated here.

Both functions accept further configuration options through ``**kwargs``
that are sourced from ``bacpipe.config`` and ``bacpipe.settings`` at
runtime. Declaring the most frequently used ones explicitly here makes
them discoverable while typing, without changing any runtime behaviour.

This file has no effect on the runtime import of the package.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch


def prepare_probe_inference(
    model: str,
    probe_path: str | Path = "",
    *,
    audio_dir: Optional[str | Path] = None,
    main_results_dir: Optional[str | Path] = None,
    dim_reduc_parent_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, int]]: ...


def run_probe_inference(
    model: str,
    linear_probe: Any,
    threshold: float = 0.5,
    embeds: Optional[np.ndarray | torch.Tensor] = None,
    return_binary_presence: bool = True,
    callbacks: Optional[Callable[..., Any]] = None,
    device: str = "cpu",
    *,
    audio_dir: Optional[str | Path] = None,
    **kwargs: Any,
) -> np.ndarray: ...


def __getattr__(name: str) -> Any: ...
