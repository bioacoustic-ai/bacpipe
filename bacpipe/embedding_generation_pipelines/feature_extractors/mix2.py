from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from bacpipe.model_specific_utils.mix2.mobile_net_v3 import mobilenetv3, MinMaxNorm
import torch

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(3 * SAMPLE_RATE)

from ..utils import ModelBaseClass


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        self.model = mobilenetv3()
        dict = torch.load(
            self.model_base_path / "mix2/mix2.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(dict["encoder"])
        self.mel = MelSpectrogram(n_fft=512, hop_length=128, n_mels=128)
        self.ampl2db = AmplitudeToDB()
        self.min_max_norm = MinMaxNorm()

    def preprocess(self, audio):
        audio = audio.cpu()
        audio = self.mel(audio)
        audio = self.ampl2db(audio)
        audio = self.min_max_norm(audio)
        return audio.unsqueeze(dim=1)

    def __call__(self, x):
        return self.model(x)
