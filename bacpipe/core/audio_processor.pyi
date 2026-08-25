"""Type stub for :mod:`bacpipe.core.audio_processor`.

This stub exists only to improve editor autosuggestions (e.g. in VS Code
with Pylance) for the public ``bacpipe.AudioHandler`` API. The docstrings
are read from the implementation in ``audio_processor.py``, so they are
rendered on hover without being duplicated here.

``AudioHandler`` forwards a number of options through ``**kwargs`` that
are sourced from ``bacpipe.config`` and ``bacpipe.settings`` at runtime.
Declaring the most frequently used ones explicitly here makes them
discoverable while typing, without changing any runtime behaviour.

This file has no effect on the runtime import of the package.
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch


class _ModelStub:
    name: str
    model_name: str
    sr: int
    segment_length: int
    only_embed_annotations: bool
    device: str

    def __init__(
        self,
        name: str,
        sr: int,
        segment_length: int,
        only_embed_annotations: bool = False,
    ) -> None: ...


def _get_model_constants(model_name: str) -> tuple[str, int, int]: ...


class AudioHandler:
    # ``model`` is either a feature extractor model object or, if the
    # model was passed by name, a ``_ModelStub`` until the model itself
    # is needed.
    model: Any
    model_name: str
    audio_dir: str | Path
    padding: str
    bool_change_speed: bool
    new_speed: Optional[float]
    kwargs: dict[str, Any]
    file_length: dict[str, float]
    preprocessed_shape: tuple[int, ...]

    def __init__(
        self,
        model: Any,
        audio_dir: str | Path,
        padding: str = "constant",
        bool_change_speed: bool = False,
        new_speed: Optional[float] = None,
        *,
        only_embed_annotations: Optional[bool] = None,
        annotations_filename: Optional[str] = None,
        annotations_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> None: ...

    def prepare_audio(self, sample: str | Path) -> torch.Tensor: ...

    def get_file_length(self, path: Path) -> None: ...

    def load_and_resample(
        self, path: str | Path
    ) -> tuple[torch.Tensor, int]: ...

    def only_load_annotated_segments(
        self,
        file_path: str | Path,
        annotations_filename: str = "annotations.csv",
        annotations_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> torch.Tensor: ...

    def window_audio(self, audio: torch.Tensor) -> torch.Tensor: ...

    # attributes of the loaded model and the kwargs are set dynamically
    def __getattr__(self, name: str) -> Any: ...


def __getattr__(name: str) -> Any: ...
