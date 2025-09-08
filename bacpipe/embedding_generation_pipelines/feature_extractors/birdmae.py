import importlib.resources as pkg_resources
import torch
import yaml


import bacpipe
from transformers import AutoFeatureExtractor, AutoModel
from ..utils import ModelBaseClass

with pkg_resources.open_text(bacpipe, "settings.yaml") as f:
    settings = yaml.load(f, Loader=yaml.CLoader)

DEVICE = settings["device"]

SAMPLE_RATE = 32_000
LENGTH_IN_SAMPLES = 160_000


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES)

        self.audio_processor = AutoFeatureExtractor.from_pretrained(
            "DBD-research-group/Bird-MAE-Base", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "DBD-research-group/Bird-MAE-Huge",
            trust_remote_code=True,
            dtype="auto",
        )
        self.model.eval()
        self.model.to(DEVICE)

    def preprocess(self, audio):
        processed_audio = self.audio_processor(audio).unsqueeze(1)
        return processed_audio.to(DEVICE)

    @torch.inference_mode()
    def __call__(self, input):
        return self.model(input).last_hidden_state
