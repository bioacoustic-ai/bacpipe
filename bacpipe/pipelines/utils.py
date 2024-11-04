import yaml
import librosa as lb
import importlib
import numpy as np

MODEL_BASE_PATH = 'bacpipe/models'

class ModelBaseClass:
    def __init__(self, sr, segment_length, **kwargs):
        with open('bacpipe/config.yaml', 'rb') as f:
            self.config = yaml.safe_load(f)
            self.sr = sr
            self.segment_length = segment_length
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def resample(self, input_tup):
        if not input_tup[1]:
            embeddings = input_tup[0]
            return embeddings
        else:
            audio, sr = input_tup
            target_sr = self.sr
            re_audio = lb.resample(audio, orig_sr=sr, target_sr=target_sr)
            # return self.window_audio(re_audio)
            return re_audio
        
    def window_audio(self, audio):
        target_input_length = int(importlib
                               .import_module(
                                    f"bacpipe.pipelines.{self.config['embedding_model']}")
                                .LENGTH_IN_SAMPLES)
        num = np.ceil(len(audio) / target_input_length)
        # zero pad in case the end is reached
        audio = [*audio, 
                 *np.zeros([int(num * target_input_length - len(audio))])]
        return np.array(audio).reshape([int(num), target_input_length])