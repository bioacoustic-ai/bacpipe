import torch
import numpy as np
from transformers import ClapModel, ClapProcessor

from ..model_utils import ModelBaseClass

SAMPLE_RATE = 48_000
LENGTH_IN_SAMPLES = 480_000

BATCH_SIZE = 16


class Model(ModelBaseClass):
    """
    BioLingual feature extractor (contrastive language-audio pretraining).
    """

    def __init__(self, **kwargs):
        """
        Initialize the BioLingual model.
        """
        super().__init__(
            sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
        )

        self.preprocessor = ClapProcessor.from_pretrained(
            "davidrrobinson/BioLingual",
        )
        self.model = ClapModel.from_pretrained(
            "davidrrobinson/BioLingual",
        )

        self.model.to(self.device)

    def preprocess(self, audio):
        """
        Preprocess the audio frames into input features for the model.

        Parameters
        ----------
        audio : torch.Tensor
            audio frames to be preprocessed

        Returns
        -------
        torch.Tensor
            input features for the model
        """
        audio_input = []
        for frame in audio:
            features = self.preprocessor(
                audios=frame.cpu(),
                return_tensor="pt",
                sampling_rate=SAMPLE_RATE,
            )
            audio_input.append(features["input_features"])
        audio_input = np.array(audio_input)
        audio_input = torch.from_numpy(audio_input)
        return audio_input.squeeze(1)

    def __call__(self, input):
        """
        Get the audio features from the model for the input.

        Parameters
        ----------
        input : torch.Tensor
            preprocessed input features

        Returns
        -------
        torch.Tensor
            audio features
        """
        return self.model.get_audio_features(input)
