import importlib.resources as pkg_resources

import torch
import yaml

import bacpipe
from .beats import BeatsModel
from ..utils import ModelBaseClass


SAMPLE_RATE = 16_000
LENGTH_IN_SAMPLES = int(5 * SAMPLE_RATE)

BEATS_PRETRAINED_PATH_SSL = "naturebeats/BEATs_iter3_plus_AS2M.pt"
BEATS_PRETRAINED_PATH_NATURELM = "naturebeats/naturebeats.pt"


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)

        self.beats = BeatsModel(
            checkpoint_path=self.model_base_path / BEATS_PRETRAINED_PATH_SSL
        )
        beats_ckpt_naturelm = torch.load(
            self.model_base_path / BEATS_PRETRAINED_PATH_NATURELM,
            map_location=self.device,
            weights_only=True,
        )

        if "predictor.weight" in beats_ckpt_naturelm.keys():
            beats_ckpt_naturelm.pop("predictor.weight")
        if "predictor.bias" in beats_ckpt_naturelm.keys():
            beats_ckpt_naturelm.pop("predictor.bias")

        self.beats.model.load_state_dict(beats_ckpt_naturelm, strict=True)
        self.beats.model.eval()
        self.beats.model.to(self.device)

    def preprocess(self, audio):
        audio = torch.clamp(audio, -1.0, 1.0)
        return self.beats.process_audio_beats(audio)

    def __call__(self, x):
        return self.beats.get_embeddings(x)
