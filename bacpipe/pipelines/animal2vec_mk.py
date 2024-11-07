from fairseq import checkpoint_utils
from tqdm import tqdm
import numpy as np
import torch
from bacpipe.model_utils.animal2vec_nn.nn import chunk_and_normalize
from .utils import ModelBaseClass

SAMPLE_RATE = 8000
LENGTH_IN_SAMPLES = int(10 * SAMPLE_RATE)


class Model(ModelBaseClass):
    def __init__(self, xeno_canto=False):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES)
        if xeno_canto:
            from . import animal2vec_xc as A2VXC

            path_to_pt_file = A2VXC.PATH_TO_PT_FILE
            self.sr = A2VXC.SAMPLE_RATE
            self.segment_length = A2VXC.LENGTH_IN_SAMPLES
        else:
            path_to_pt_file = (
                "bacpipe/models/animal2vec/animal2vec_large_finetuned_MeerKAT_240507.pt"
            )
        models, _ = checkpoint_utils.load_model_ensemble(
            [path_to_pt_file]
        )  # , weights_only=True)
        self.model = models[0].to("cpu")
        self.model.eval()

    def preprocess(self, audio):
        chunk = chunk_and_normalize(
            torch.tensor(audio.reshape(1, -1)),
            segment_length=self.segment_length / self.sr,
            sample_rate=self.sr,
            normalize=True,
            max_batch_size=16,
        )
        return chunk[0]

    @torch.inference_mode()
    def __call__(self, input):
        input = torch.from_numpy(np.array(input))
        res = self.model(source=input, mode="AUDIO", features_only=True)
        embeds = res["layer_results"]
        embeds = (
            np.array([a.detach().numpy() for a in embeds]).mean(axis=0).mean(axis=1)
        )
        return embeds
