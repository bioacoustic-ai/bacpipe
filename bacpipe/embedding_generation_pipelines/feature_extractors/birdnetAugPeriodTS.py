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
        kwargs['check_if_combination_exists'] = True
        self.ld = Loader(**kwargs)
        self.file_length = (
            self.ld.metadata_dict['segment_length (samples)']
            / self.ld.metadata_dict['sample_rate (Hz)']
            )
    
    def __call__(self, file):
        # embeds = []
        # file_path = self.ld.metadata_dict['embed_dir'].replace('bacpipe', 'bacpipe_results')
        # # file_path = self.ld.metadata_dict['embed_dir'].replace('bacpipe_results', '/mnt/swap/Work/Embeddings')
        # embed_files = Path(file_path).rglob('*.npy')
        
        # for embed_file in embed_files:
        #     if file.stem in embed_file.stem:
        #         embeds.extend(np.load(embed_file))
        # embeds = np.array(embeds)
        embeds = []
        from bacpipe import settings
        orig_embed_dir = Path(self.ld.metadata_dict['embed_dir'])
        file_path = str(orig_embed_dir).replace(str(list(orig_embed_dir.parents)[1]), 
                                                settings.main_results_dir + '/' + str(orig_embed_dir).split('-')[-1])
        
        embed_files = Path(file_path).rglob('*.npy')
        
        for embed_file in embed_files:
            if file.stem in embed_file.stem:
                embeds.extend(np.load(embed_file))
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
        tod_ts_n = np.array(tod_ts_n, dtype=np.float32) * 2*np.pi
        sin_tod_ts_n = np.sin(tod_ts_n)
        cos_tod_ts_n = np.cos(tod_ts_n)
        
        embed_ts = np.zeros([embeds.shape[0], embeds.shape[1] + 100])
        # embed_ts = np.zeros(embeds.shape)
        for idx, embed in enumerate(embeds):
            embed_ts[idx] = np.append(embed, [sin_tod_ts_n[idx]] * 50 + [cos_tod_ts_n[idx]] * 50)
            # embed_ts[idx] = embed * self.tod_ts_n[idx]
        return embed_ts