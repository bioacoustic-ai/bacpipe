from ..utils import ModelBaseClass
import numpy as np
from datetime import datetime
SAMPLE_RATE = 48000
LENGTH_IN_SAMPLES = 144000

class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        src = '/mnt/swap/Work/Embeddings/PER - soundscapes southwestern Amazon/evaluations/birdnet/labels/default_labels.npy'
        self.labels = np.load(src, allow_pickle=True).item()
        self.tod = self.labels['time_of_day']
        self.tod_ts = [datetime.strptime(ts, '%H-%M-%S').timestamp() for ts in self.tod]
        ts_min = min(self.tod_ts)
        ts_max = max(self.tod_ts)
        self.tod_ts_n = [
            (ts - ts_min) 
            / (ts_max - ts_min) 
            for ts in self.tod_ts
            ]
        self.tod_ts_n = np.array(self.tod_ts_n, dtype=np.float32)
    
    def __call__(self, embeds, file):
        embed_ts = np.zeros([embeds.shape[0], embeds.shape[1] + 1])
        # embed_ts = np.zeros(embeds.shape)
        audio_file = file.parent.stem + '/' + file.stem.split(f'_birdnet')[0] + '.flac'
        bool_arr = [aud == audio_file for aud in self.labels['audio_file_name']]
        tod_ts_n_this_file = self.tod_ts_n[bool_arr]
        for idx, embed in enumerate(embeds):
            embed_ts[idx] = np.append(embed, tod_ts_n_this_file[idx])
            # embed_ts[idx] = embed * self.tod_ts_n[idx]
        return embed_ts