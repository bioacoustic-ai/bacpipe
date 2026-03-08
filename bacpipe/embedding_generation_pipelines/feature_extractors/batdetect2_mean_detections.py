from functools import partial

import numpy as np
import torch

from ..utils import ModelBaseClass

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
        spectrograms = [self.generate_spectrogram(segment) for segment in segments]
        return torch.from_numpy(np.concatenate(spectrograms, axis=0))

    @torch.no_grad()
    def __call__(self, x, return_class_results=False):
        output = self.model(x)

        features = output.features.mean(dim=(-2, -1))

        # NOTE: Last element is the background class
        class_scores = output.pred_class.amax(dim=(-2, -1))[:, :-1]

        if return_class_results:
            return features, class_scores

        return features

    def classifier_predictions(self, inference_results):
        # NOTE: This method is left unimplemented. Since 'inference_results'
        # are averaged across the whole audio clip to map to the single-feature
        # interface, running a classifier on these aggregated features won't
        # produce the intended results.
        raise NotImplementedError(
            "Classifier predictions are invalid for averaged features."
        )
