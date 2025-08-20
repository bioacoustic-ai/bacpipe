import torch
import yaml

from .beats import BeatsModel
from ..utils import ModelBaseClass

SAMPLE_RATE = 16_000
LENGTH_IN_SAMPLES = int(5 * SAMPLE_RATE)
with open("bacpipe/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

DEVICE = settings["device"]

BEATS_PRETRAINED_PATH_SSL = (
    "bacpipe/model_checkpoints/naturelm_audio/BEATs_iter3_plus_AS2M.pt"
)
BEATS_PRETRAINED_PATH_NATURELM = (
    "bacpipe/model_checkpoints/naturelm_audio/naturebeats.pt"
)


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES)

        self.beats = BeatsModel(checkpoint_path=BEATS_PRETRAINED_PATH_SSL)
        beats_ckpt_naturelm = torch.load(
            BEATS_PRETRAINED_PATH_NATURELM, map_location=DEVICE, weights_only=True
        )

        if "predictor.weight" in beats_ckpt_naturelm.keys():
            beats_ckpt_naturelm.pop("predictor.weight")
        if "predictor.bias" in beats_ckpt_naturelm.keys():
            beats_ckpt_naturelm.pop("predictor.bias")

        self.beats.model.load_state_dict(beats_ckpt_naturelm, strict=True)
        self.beats.model.eval()

    def preprocess(self, audio):
        audio = torch.clamp(audio, -1.0, 1.0)
        return self.beats.process_audio_beats(audio)

    def __call__(self, x):
        return self.beats.get_embeddings(x)
