import torch
from transformers import pipeline
from ..utils import ModelBaseClass
import numpy as np
import yaml


with open("bacpipe/settings.yaml", "rb") as f:
    settings = yaml.load(f, Loader=yaml.CLoader)

DEVICE = settings["device"]


SAMPLE_RATE = 48_000
LENGTH_IN_SAMPLES = 480_000

BATCH_SIZE = 16


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES)
        self.audio_classifier = pipeline(
            task="zero-shot-audio-classification", model="davidrrobinson/BioLingual"
        )
        self.model = self.audio_classifier.model.get_audio_features

    def preprocess(self, audio):
        audio_input = []
        for frame in audio:
            features = self.audio_classifier.feature_extractor(
                frame.cpu(), sampling_rate=SAMPLE_RATE
            )
            audio_input.append(features["input_features"])
        audio_input = np.array(audio_input)
        audio_input = torch.from_numpy(audio_input)
        return audio_input.squeeze(1)

    @torch.inference_mode()
    def __call__(self, input):
        if DEVICE == 'cuda':
            return self.model(input.cuda())
        else:
            return self.model(input)
