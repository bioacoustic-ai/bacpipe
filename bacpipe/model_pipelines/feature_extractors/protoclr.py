from torchaudio import transforms as T
import torch
from bacpipe.model_pipelines.model_specific_utils.protoclr.cvt import cvt13
from ..model_utils import ModelBaseClass
import yaml

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(SAMPLE_RATE * 6)
BATCH_SIZE = 8


# Mel Spectrogram
NMELS = 128  # number of mels
NFFT = 1024  # size of FFT
HOPLEN = 320  # hop between STFT windows
FMAX = 8000  # fmax
FMIN = 50  # fmin


class Normalization(torch.nn.Module):
    def __init__(self):
        """
        Initialize the normalization module.
        """
        super().__init__()
        self.batch_size = BATCH_SIZE

    def forward(self, x):
        """
        Normalize the input to the range [0, 1].

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            normalized tensor
        """
        return (x - x.min()) / (x.max() - x.min())


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        """
        Initialize the ProtoCLR feature extractor model.
        """
        super().__init__(
            sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
        )
        self.batch_size = 2

        self.mel = (
            T.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=NFFT,
                hop_length=HOPLEN,
                f_min=FMIN,
                f_max=FMAX,
                n_mels=NMELS,
            )
            .to(self.device)
            .eval()
        )
        self.power_to_db = T.AmplitudeToDB().eval()
        self.norm = Normalization().eval()

        self.model = cvt13()
        state_dict = torch.load(
            self.model_base_path / "protoclr/protoclr.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, audio):
        """
        Convert the audio samples to a normalized mel spectrogram.

        Parameters
        ----------
        audio : torch.Tensor
            audio samples to be preprocessed

        Returns
        -------
        torch.Tensor
            normalized mel spectrogram
        """
        audio = audio.to(self.device)
        mel = self.mel(audio)
        mel = self.power_to_db(mel)
        mel = self.norm(mel)
        return mel

    def __call__(self, input):
        """
        Run the model on the input spectrogram.

        Parameters
        ----------
        input : torch.Tensor
            input mel spectrogram

        Returns
        -------
        torch.Tensor
            model output
        """
        res = self.model(input.unsqueeze(1))
        return res
