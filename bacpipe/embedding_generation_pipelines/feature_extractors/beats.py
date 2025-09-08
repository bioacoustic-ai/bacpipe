import torch

from bacpipe.model_specific_utils.naturebeats.BEATs import BEATs, BEATsConfig
from ..utils import ModelBaseClass


SAMPLE_RATE = 16_000
LENGTH_IN_SAMPLES = int(5 * SAMPLE_RATE)


BEATS_PRETRAINED_PATH_FT = "naturebeats/BEATs_iter1_finetuned_on_AS2M_cpt1.pt"


class BeatsModel:
    def __init__(self, checkpoint_path):

        # load the fine-tuned checkpoints
        checkpoint = torch.load(checkpoint_path)

        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.avg_pooling = True

        # disable classifier
        self.model.predictor = None

        self.process_audio_beats = self.model.preprocess

    def get_embeddings(self, spectrogram_input):
        """
        Taken from the BEATS forward call. Adapted to work based on the spectrogram input
        to enable visualization of spectrograms for model result interpretation.

        Parameters
        ----------
        spectrogram_input : torch.Tensor
            batched spectrograms from self.model.preprocess

        Returns
        -------
        torch.Tensor
            batched embeddings
        """
        spectrogram_input = spectrogram_input.unsqueeze(1)
        features = self.model.patch_embedding(spectrogram_input)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.model.layer_norm(features)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        x = self.model.dropout_input(features)

        x, _ = self.model.encoder(
            x,
            padding_mask=None,
        )

        if self.avg_pooling:
            x = x.mean(dim=1)
        return x


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)

        self.model = BeatsModel(
            checkpoint_path=self.model_base_path / BEATS_PRETRAINED_PATH_FT
        )
        self.model.model.eval()
        self.model.model.to(self.device)

    def preprocess(self, audio):
        return self.model.process_audio_beats(audio)

    def __call__(self, x):
        return self.model.get_embeddings(x)
