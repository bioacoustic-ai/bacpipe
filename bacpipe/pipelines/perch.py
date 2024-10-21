from ml_collections import config_dict
from bacpipe.perch_chirp.chirp.inference.embed_lib import EmbedFn
from bacpipe.perch_chirp.chirp.projects.zoo.models import get_preset_model_config
import numpy as np
import librosa as lb

from .utils import ModelBaseClass

SAMPLE_RATE = 32000
LENGH_IN_SAMPLES = 160000

class Model(ModelBaseClass):
    def __init__(self):
        super().__init__()
        
        model_choice = 'perch_8'
        config = config_dict.ConfigDict()
        config.embed_fn_config = config_dict.ConfigDict()
        config.embed_fn_config.model_config = config_dict.ConfigDict()
        model_key, embedding_dim, model_config = get_preset_model_config(
            model_choice)
        config.embed_fn_config.model_key = model_key
        config.embed_fn_config.model_config = model_config

        # Only write embeddings to reduce size.
        config.embed_fn_config.write_embeddings = True
        config.embed_fn_config.write_logits = False
        config.embed_fn_config.write_separated_audio = False
        config.embed_fn_config.write_raw_audio = False
        config.embed_fn_config.file_id_depth = 1
        embed_fn = EmbedFn(**config.embed_fn_config)
        embed_fn.setup()
        self.model = embed_fn.embedding_model.embed
        
        
    def preprocess(self, audio):
        re_audio = lb.resample(audio, 
                                orig_sr = self.config['sr'], 
                                target_sr = SAMPLE_RATE)
        num = np.ceil(len(re_audio) / LENGH_IN_SAMPLES)
        # zero pad in case the end is reached
        re_audio = [*re_audio, 
                    *np.zeros([int(num * LENGH_IN_SAMPLES - len(re_audio))])]
        wins = np.array(re_audio).reshape([int(num), LENGH_IN_SAMPLES])

        return np.array(wins, dtype=np.float32)


    def __call__(self, input):
        return self.model(input).embeddings.squeeze()
    