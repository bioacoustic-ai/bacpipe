import yaml
import librosa as lb
import numpy as np

MODEL_BASE_PATH = "bacpipe/models"


class ModelBaseClass:
    def __init__(self, sr, segment_length, **kwargs):
        with open("bacpipe/config.yaml", "rb") as f:
            self.config = yaml.safe_load(f)
            self.sr = sr
            self.segment_length = segment_length
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_and_resample(self, path):
        re_audio, sr = lb.load(path, sr=self.sr)
        return re_audio

    def window_audio(self, audio):
        num_frames = np.ceil(len(audio) / self.segment_length)
        padded_audio = lb.util.fix_length(
            audio, size=int(num_frames * self.segment_length), mode="reflect"
        )
        frames = lb.util.frame(
            padded_audio,
            frame_length=self.segment_length,
            hop_length=self.segment_length,
            axis=0,
            writeable=True,
        )
        return frames
