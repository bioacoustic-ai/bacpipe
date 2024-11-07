import torch
from transformers import pipeline
import librosa as lb
from .utils import ModelBaseClass
import numpy as np
from tqdm import tqdm

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
        features = self.audio_classifier.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE
        )
        aud_input = features["input_features"]
        aud_input = torch.tensor(aud_input)
        return aud_input

    @torch.inference_mode()
    def __call__(self, input):
        embeds = []
        for batch in tqdm(input.split(BATCH_SIZE)):
            embeds.append(self.model(batch))
        embeds = torch.cat(embeds)
        return np.array(embeds)
