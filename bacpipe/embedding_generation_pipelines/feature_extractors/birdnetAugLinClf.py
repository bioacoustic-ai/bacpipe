from ..utils import ModelBaseClass
import numpy as np
from datetime import datetime
from bacpipe.generate_embeddings import Loader
from pathlib import Path
import torch
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
        self.clf = torch.load('birdnet_lin_clf.pth', weights_only=False)
    
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
        embeds = torch.Tensor(embeds)
        
        embeds_logits = self.clf(embeds)
        embeds_logits = embeds_logits.detach().cpu().numpy()
        return embeds_logits