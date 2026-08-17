import importlib.resources as pkg_resources

import torch
import yaml

import bacpipe
from .beats import BeatsModel
from ..model_utils import ModelBaseClass

SAMPLE_RATE = 16_000
LENGTH_IN_SAMPLES = int(5 * SAMPLE_RATE)

BEATS_PRETRAINED_PATH_FT = (
    "beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"
)
BEATS_PRETRAINED_PATH_NATURELM = "naturebeats/naturebeats.pt"


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        """
        Initialize the NatureBeats model.
        """
        super().__init__(
            sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
        )

        self.beats = BeatsModel(
            checkpoint_path=self.model_base_path / BEATS_PRETRAINED_PATH_FT
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
        
        self.model = self.beats.model
        self.model.preprocess_audio = self.beats.process_audio_beats
        self.model.return_embeddings = self.beats.get_embeddings

    def preprocess(self, audio):
        """
        Clamp the audio samples and convert them with the BEATs preprocessing.

        Parameters
        ----------
        audio : torch.Tensor
            audio samples to be preprocessed

        Returns
        -------
        torch.Tensor
            preprocessed audio
        """
        audio = torch.clamp(audio, -1.0, 1.0)
        return self.beats.process_audio_beats(audio)

    def __call__(self, x):
        """
        Get the BEATs embeddings for the input.

        Parameters
        ----------
        x : torch.Tensor
            preprocessed audio

        Returns
        -------
        torch.Tensor
            BEATs embeddings
        """
        return self.beats.get_embeddings(x)
