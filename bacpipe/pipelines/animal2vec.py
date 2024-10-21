from bacpipe.animal2vec_nn.nn import chunk_and_normalize
from fairseq import checkpoint_utils
from .utils import ModelBaseClass
from tqdm import tqdm
import numpy as np
import torch

SAMPLE_RATE = 8000
LENGTH_IN_SAMPLES = int(10 * SAMPLE_RATE)

class Model(ModelBaseClass):
    def __init__(self):
        super().__init__()
        path_to_pt_file = "bacpipe/models/animal2vec/animal2vec_large_finetuned_MeerKAT_240507.pt"
        models, _ = checkpoint_utils.load_model_ensemble([path_to_pt_file])
        self.model = models[0].to("cpu")
        self.model.eval()
        
    
    def preprocess(self, audio):
        chunk = chunk_and_normalize(
            torch.tensor(audio),
            segment_length=LENGTH_IN_SAMPLES/SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            normalize=True,
            max_batch_size=16
        )
        return chunk[0]
        
    @torch.inference_mode()
    def __call__(self, input):
        all_embeds = []
        for batch in tqdm(input):
            res = self.model(source=batch.view(1, -1))
            embeds = res['layer_results']
            np_embeds = [a.detach().numpy() for a in embeds]
            all_embeds.append(np_embeds)
        return np.array(all_embeds)