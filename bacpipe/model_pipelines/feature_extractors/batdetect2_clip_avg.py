from functools import partial

import numpy as np
import torch

from ..model_utils import ModelBaseClass

SAMPLE_RATE = 256_000
DEFAULT_SEGMENT_DURATION = 1
NUM_FEATURES = 32
NUM_CLASSES = 17


class Model(ModelBaseClass):
    def __init__(
        self,
        segment_duration=DEFAULT_SEGMENT_DURATION,
        **kwargs,
    ):
        super().__init__(
            sr=SAMPLE_RATE,
            segment_length=int(segment_duration * SAMPLE_RATE),
            **kwargs,
        )

        from batdetect2 import api

        self.config = api.get_config()
        self.model, _ = api.load_model(device=self.device)  # type: ignore

        self.generate_spectrogram = partial(
            api.generate_spectrogram,
            config=self.config,
            samp_rate=SAMPLE_RATE,
            device=self.device,
        )
        self.classes = self.config["class_names"]

    def preprocess(self, audio):
        segments = audio.numpy()
        # NOTE: Need to pre-process each segment separately
        spectrograms = torch.stack(
            [self.generate_spectrogram(segment) for segment in segments]
            )
        if len(spectrograms.shape) > 4:
            spectrograms = spectrograms.squeeze(1)
        return spectrograms

    @torch.no_grad()
    def __call__(self, x):
        self.output = self.model(x)

        features = self.output.features.mean(dim=(-2, -1))

        return features

    def classifier_predictions(self, embeddings):
        # NOTE: Last element is the background class
        class_scores = self.output.pred_class.amax(dim=(-2, -1))[:, :-1]
        return class_scores
