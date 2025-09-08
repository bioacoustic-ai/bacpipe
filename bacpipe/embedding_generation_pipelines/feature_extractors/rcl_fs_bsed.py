import torch
from torchaudio import transforms as T
from bacpipe.model_specific_utils.rcl_fs_bsed.resnet import ResNet
from ..utils import ModelBaseClass

SAMPLE_RATE = 22050
LENGTH_IN_SAMPLES = int(0.2 * SAMPLE_RATE)

# spectrogram parameters
N_FFT = 512
HOP_LENGTH = 128
F_MIN = 50
F_MAX = 11025
N_MELS = 128


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        self.model = ResNet()
        state_dict = torch.load(
            self.model_base_path / "rcl_fs_bsed/bioacoustics_model.pth",
            weights_only=True,
            map_location=self.device,
        )
        enc_sd = state_dict["encoder"]
        drop_keys = ["lin.0.weight", "lin.0.bias", "lin.2.weight", "lin.2.bias"]
        enc_sd = {k: v for k, v in enc_sd.items() if k not in drop_keys}
        self.model.load_state_dict(enc_sd)
        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
        )
        self.power_to_db = T.AmplitudeToDB()
        self.model.eval()

    def preprocess(self, audio):
        audio = audio.cpu()
        mel = self.mel(torch.tensor(audio))
        mel_db = self.power_to_db(mel)
        return mel_db

    @torch.inference_mode()
    def __call__(self, input):
        res = self.model(input.unsqueeze(1))
        return res
