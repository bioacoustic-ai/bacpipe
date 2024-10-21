# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define the globals that can be used in configuration files."""
from typing import Any

from bacpipe.perch_chirp.chirp import audio_utils
from bacpipe.perch_chirp.chirp import config_utils
from bacpipe.perch_chirp.chirp.eval import callbacks
from bacpipe.perch_chirp.chirp.eval import eval_lib
from bacpipe.perch_chirp.chirp.models import conformer
from bacpipe.perch_chirp.chirp.models import efficientnet
from bacpipe.perch_chirp.chirp.models import efficientnet_v2
from bacpipe.perch_chirp.chirp.models import frontend
from bacpipe.perch_chirp.chirp.models import handcrafted_features
from bacpipe.perch_chirp.chirp.models import hubert
from bacpipe.perch_chirp.chirp.models import layers
from bacpipe.perch_chirp.chirp.models import quantizers
from bacpipe.perch_chirp.chirp.models import soundstream_unet
from bacpipe.perch_chirp.chirp.models import taxonomy_model
from bacpipe.perch_chirp.chirp.preprocessing import pipeline
from bacpipe.perch_chirp.chirp.train import train_utils
from flax import linen as nn
import optax


def get_globals() -> dict[str, Any]:
  return {
      "audio_utils": audio_utils,
      "callbacks": callbacks,
      "config_utils": config_utils,
      "conformer": conformer,
      "efficientnet": efficientnet,
      "efficientnet_v2": efficientnet_v2,
      "eval_lib": eval_lib,
      "hubert": hubert,
      "quantizers": quantizers,
      "frontend": frontend,
      "layers": layers,
      "nn": nn,
      "optax": optax,
      "pipeline": pipeline,
      "handcrafted_features": handcrafted_features,
      "soundstream_unet": soundstream_unet,
      "taxonomy_model": taxonomy_model,
      "train_utils": train_utils,
  }
