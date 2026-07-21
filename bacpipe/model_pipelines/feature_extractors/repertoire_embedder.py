import torch, numpy as np
from bacpipe.model_pipelines.model_specific_utils.repertoire_embedder.module import frontend_medfilt, sparrow_encoder, sparrow_decoder
from ..model_utils import ModelBaseClass

# Author: Paul Best
# This model was originally published in this repo https://gitlab.lis-lab.fr/paul.best/repertoire_embedder
# See the associated publication (to use for citation) https://doi.org/10.1371/journal.pone.0283396

SAMPLE_RATE = 22050
LENGTH_IN_SEC = 1
LENGTH_IN_SAMPLES = int(LENGTH_IN_SEC * SAMPLE_RATE)


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        nMel, bottleneck, NFFT, sampleDur = 128, 256, 1024, LENGTH_IN_SEC
        self.frontend = frontend_medfilt(SAMPLE_RATE, NFFT, sampleDur, nMel)
        encoder = sparrow_encoder(bottleneck // (nMel//32 * 4), (nMel//32, 4))
        decoder = sparrow_decoder(bottleneck, (nMel//32, 4))
        self.model = torch.nn.Sequential(encoder, decoder)
        state_dict = torch.load(
            self.model_base_path / "repertoire_embedder/generic_embedder.weights",
            map_location='cpu'
        )
        self.model.load_state_dict(state_dict)

    def preprocess(self, audio): # audio is a torch.tensor object
        return (audio - torch.mean(audio) ) / torch.std(audio)

    @torch.inference_mode()
    def __call__(self, input):
        x = self.frontend(input)
        embeddings = self.model[0](x)
        return embeddings