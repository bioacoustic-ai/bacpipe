from torchaudio import transforms as T
import torch
from bacpipe.model_utils.protoclr.cvt import cvt13
from .utils import ModelBaseClass

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(SAMPLE_RATE * 6)


# Mel Spectrogram
DEVICE = "cpu"  # device to use ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...]
NMELS = 128  # number of mels
NFFT = 1024  # size of FFT
HOPLEN = 320  # hop between STFT windows
FMAX = 8000  # fmax
FMIN = 50  # fmin


class Normalization(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES)

        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=NFFT,
            hop_length=HOPLEN,
            f_min=FMIN,
            f_max=FMAX,
            n_mels=NMELS,
        ).to(DEVICE)
        self.power_to_db = T.AmplitudeToDB()
        self.norm = Normalization()

        self.model = cvt13()
        state_dict = torch.load(
            "bacpipe/models/protoclr/protoclr_300.pth", map_location="cpu"
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(DEVICE)

    def preprocess(self, audio):
        mel = self.mel(audio)
        mel = self.power_to_db(mel)
        mel = self.norm(mel)
        return mel

    def __call__(self, input):
        res = self.model(input.unsqueeze(1))
        return res
