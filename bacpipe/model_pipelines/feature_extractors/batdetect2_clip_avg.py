from functools import partial

import numpy as np
import torch

from ..model_utils import ModelBaseClass

SAMPLE_RATE = 256_000
LENGTH_IN_SAMPLES = 256_000
DEFAULT_SEGMENT_DURATION = 1
NUM_FEATURES = 32
NUM_CLASSES = 17


class Model(ModelBaseClass):
    def __init__(
        self,
        segment_duration=DEFAULT_SEGMENT_DURATION,
        **kwargs,
    ):
        """
        Initialize the BatDetect2 clip-averaged model.

        Parameters
        ----------
        segment_duration : float
            duration of each audio segment in seconds
        **kwargs
            additional keyword arguments passed to the base class
        """
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
        """
        Generate a spectrogram for each audio segment.

        Parameters
        ----------
        audio : torch.Tensor
            audio samples to be preprocessed

        Returns
        -------
        torch.Tensor
            stacked spectrograms
        """
        if audio.device.type == "cuda":
            segments = audio.cpu().numpy()
        else:
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
        """
        Run the model on the input spectrograms.

        Parameters
        ----------
        x : torch.Tensor
            preprocessed spectrograms

        Returns
        -------
        torch.Tensor
            clip-averaged features
        """
        self.output = self.model(x)

        features = self.output.features.mean(dim=(-2, -1))

        return features

    def classifier_predictions(self, embeddings):
        """
        Return the class scores stored during the last call.

        Parameters
        ----------
        embeddings : torch.Tensor
            embeddings from the last model call (unused)

        Returns
        -------
        torch.Tensor
            max class scores across detections (background class removed)
        """
        # NOTE: Last element is the background class
        class_scores = self.output.pred_class.amax(dim=(-2, -1))[:, :-1]
        return class_scores
