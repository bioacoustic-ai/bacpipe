from ..utils import ModelBaseClass
import numpy as np
from datetime import datetime
from bacpipe.generate_embeddings import Loader
from pathlib import Path
SAMPLE_RATE = 48000
LENGTH_IN_SAMPLES = 144000

class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        kwargs['model_name'] = kwargs['model_name'].split('Aug')[0]
        self.ld = Loader(**kwargs)
        self.file_length = (
            self.ld.metadata_dict['segment_length (samples)']
            / self.ld.metadata_dict['sample_rate (Hz)']
            )
    
    def __call__(self, file):
        embeds = []
        for file in Path(self.ld.metadata_dict['embed_dir']).rglob('*.npy'):
            embeds.extend(np.load(file))
        embeds = np.array(embeds)
        
        default_labels = np.load(
            self.ld.evaluations_dir 
            / self.ld.metadata_dict['model_name']
            / 'labels/default_labels.npy',
            allow_pickle=True
            ).item()
        TOD = default_labels['time_of_day']
        TOD_ts = [datetime.strptime(ts ,'%H-%M-%S').timestamp() for ts in TOD]
        
        ts_min = min(TOD_ts)
        ts_max = max(TOD_ts)
        tod_ts_n = [
            (ts - ts_min) 
            / (ts_max - ts_min) 
            for ts in TOD_ts
            ]
        tod_ts_n = np.array(tod_ts_n, dtype=np.float32)
        
        embed_ts = np.zeros([embeds.shape[0], embeds.shape[1] + 1])
        # embed_ts = np.zeros(embeds.shape)
        for idx, embed in enumerate(embeds):
            embed_ts[idx] = np.append(embed, tod_ts_n[idx])
            # embed_ts[idx] = embed * self.tod_ts_n[idx]
        return embed_ts