import torch
import numpy as np
import librosa

from bacpipe.model_specific_utils.bat.module import BAT
from bacpipe.model_specific_utils.bat.prepare_data import prepareData, getSequences, slideWindow, germanBats
from ..model_utils import ModelBaseClass

IS_EXPANDED = False
if IS_EXPANDED:
    SAMPLE_RATE = 22050
    LENGTH_IN_SAMPLES = int(0.78 * SAMPLE_RATE * 10)
else:
    SAMPLE_RATE = 22050 * 10    # time expand
    LENGTH_IN_SAMPLES = int(0.78 * SAMPLE_RATE)


class Model(ModelBaseClass):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)

        self.threshold = threshold
        self.classes = list(germanBats)

        self.model = BAT(
            max_len=60,
            patch_dim=44 * 257,
            d_model=64,
            num_classes=len(self.classes),
            nhead=2,
            dim_feedforward=32,
            num_layers=2,
            seq=False,
        )

        state_dict = torch.load(
            self.model_base_path / "bat/bat_2_convnet_mixed.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, audio: torch.Tensor):
        b_y = audio.numpy()   # b n

        input_seq = []
        for y in b_y:
            # Spectrogram
            D = librosa.stft(y, n_fft=512)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # H, W

            # Custom filtering + denoising
            S_db = prepareData(y)

            # Sequence extraction
            sequence = np.asarray(slideWindow(S_db, size=44, step=22)[:-1])
            n, w, h = sequence.shape
            input_seq.append(torch.tensor(sequence, dtype=torch.float32).reshape(n * w, h))

        return torch.stack(input_seq, dim=0)

    def classifier_predictions(self, cls_token):
        with torch.no_grad():
            logits = self.model.classifier(cls_token)
        return torch.sigmoid(logits)

    def __call__(self, x, return_class_results=False):
        with torch.no_grad():
            cls_token = self.model(x, return_token=True)
            if not return_class_results:
                return cls_token
            return cls_token, self.classifier_predictions(cls_token)
