from torchaudio import transforms as T
import torch
from bacpipe.model_specific_utils.protoclr.cvt import cvt13
from ..utils import ModelBaseClass
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
        super().__init__()
        self.batch_size = BATCH_SIZE

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
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
        audio = audio.to(self.device)
        mel = self.mel(audio)
        mel = self.power_to_db(mel)
        mel = self.norm(mel)
        return mel

    def __call__(self, input):
        res = self.model(input.unsqueeze(1))
        return res
