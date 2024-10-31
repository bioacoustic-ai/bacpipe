from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from bacpipe.model_utils.mix2.mobile_net_v3 import mobilenetv3, MinMaxNorm
import torch

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(3*SAMPLE_RATE)

from .utils import ModelBaseClass
class Model(ModelBaseClass):
    def __init__(self):
        super().__init__()
        self.model = mobilenetv3()
        dict = torch.load('bacpipe/models/mix2/mix2.pth', map_location='cpu', 
                          weights_only=True)
        self.model.load_state_dict(dict["encoder"])
        self.mel = MelSpectrogram(n_fft=512, hop_length=128, n_mels=128)
        self.ampl2db = AmplitudeToDB()
        self.min_max_norm = MinMaxNorm()
    
    def preprocess(self, x):
        x = torch.from_numpy(x[: LENGTH_IN_SAMPLES]).float() # FIXME: enable windowing so batching is possible
        x = self.mel(x)
        x = self.ampl2db(x)
        x = self.min_max_norm(x)
        return x.reshape([1, 1, 128, 376])
    
    def __call__(self, x):
        return self.model(x).detach().numpy()