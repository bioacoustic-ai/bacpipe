from torchaudio.models import wav2vec2_model
import json
import torch
import torch.nn as nn
# extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
import numpy as np

import logging
logger = logging.getLogger('ievad')
logger.setLevel(level=logging.DEBUG)

from .utils import ModelBaseClass

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = 16000

# paper: https://arxiv.org/abs/2210.14493

class Model(ModelBaseClass, nn.Module):
    def __init__(self, pooling='mean'):

        super().__init__()
        nn.Module.__init__(self)

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        base_path = 'ievad/models'
        if self.config['embedding_model'] == 'birdaves':
            model_config_path = f'{base_path}/birdaves/birdaves-bioxn-large.torchaudio.model_config.json'
            model_path = f'{base_path}/birdaves/birdaves-bioxn-large.torchaudio.pt'
        else:
            model_config_path = f'{base_path}/aves/aves-base-bio.torchaudio.model_config.json'
            model_path = f'{base_path}/aves/aves-base-bio.torchaudio.pt'
        model_config = json.load(open(model_config_path, 'r'))
        self.pooling = pooling
        self.batch_size = 64
        self.model = wav2vec2_model(**model_config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(False)
        self.model.eval()
    
    def preprocess(self, audio):
        num_of_segments = int(audio.shape[0]/LENGTH_IN_SAMPLES)
        audio = audio[:num_of_segments*LENGTH_IN_SAMPLES]
        audio = audio.reshape(num_of_segments, LENGTH_IN_SAMPLES)
        
        return torch.tensor(audio)

    def __call__(self, input):
        for i in range(input.shape[0]//self.batch_size+1):
            batch = input[i*self.batch_size:(i+1)*self.batch_size]
            out = self.model.extract_features(batch)[0][-1]
            out = getattr(torch, self.pooling)(out, dim=1)
            if i == 0:
                out_np = out.detach().numpy()
            else:
                out_np = np.append(out_np, out.detach().numpy(), axis=0)        
        return out_np

    
if __name__ == '__main__':
    torchaudio_model = Model('mean')
    torchaudio_model.eval()
    waveform = torch.rand((16_000))
    x = waveform.unsqueeze(0)
    a = torchaudio_model(x)
    
    