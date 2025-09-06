import importlib.resources as pkg_resources
import torch
import numpy as np
import yaml

import bacpipe
from transformers import ClapModel, ClapProcessor

from ..utils import ModelBaseClass

with pkg_resources.open_text(bacpipe, "settings.yaml") as f:
    settings = yaml.load(f, Loader=yaml.CLoader)

DEVICE = settings["device"]


SAMPLE_RATE = 48_000
LENGTH_IN_SAMPLES = 480_000

BATCH_SIZE = 16


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES)

        self.preprocessor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
        self.model = ClapModel.from_pretrained("davidrrobinson/BioLingual")

    def preprocess(self, audio):
        audio_input = []
        for frame in audio:
            features = self.preprocessor(
                audios=frame.cpu(), return_tensor="pt", sampling_rate=SAMPLE_RATE
            )
            audio_input.append(features["input_features"])
        audio_input = np.array(audio_input)
        audio_input = torch.from_numpy(audio_input)
        return audio_input.squeeze(1)

    def __call__(self, input):
        if DEVICE == "cuda":
            return self.model.get_audio_features(input.cuda())
        else:
            return self.model.get_audio_features(input)
