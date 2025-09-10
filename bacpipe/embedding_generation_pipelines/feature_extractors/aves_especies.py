from torchaudio.models import wav2vec2_model
import json
import torch
import torch.nn as nn

# extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
from ..utils import ModelBaseClass

BATCH_SIZE = 1  # necessary due to padding problem, experiment with this

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = 16000

# paper: https://arxiv.org/abs/2210.14493


class Model(ModelBaseClass, nn.Module):
    def __init__(self, birdaves=False, nonbioaves=False, **kwargs):

        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        nn.Module.__init__(self)

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        base_path = self.model_base_path
        if birdaves:
            model_config_path = f"{base_path}/birdaves_especies/birdaves-biox-large.torchaudio.model_config.json"
            model_path = (
                f"{base_path}/birdaves_especies/birdaves-biox-large.torchaudio.pt"
            )
        elif nonbioaves:
            model_config_path = f"{base_path}/nonbioaves_especies/aves-base-nonbio.torchaudio.model_config.json"
            model_path = (
                f"{base_path}/nonbioaves_especies/aves-base-nonbio.torchaudio.pt"
            )
        else:
            model_config_path = (
                f"{base_path}/aves_especies/aves-base-bio.torchaudio.model_config.json"
            )
            model_path = f"{base_path}/aves_especies/aves-base-bio.torchaudio.pt"
        model_config = json.load(open(model_config_path, "r"))
        self.model = wav2vec2_model(**model_config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.feature_extractor.requires_grad_(False)
        self.model.eval()

    def preprocess(self, audio):
        return audio

    @torch.inference_mode()
    def __call__(self, input):
        embeds = []
        for batch in input.split(BATCH_SIZE):
            out_raw = self.model.extract_features(batch)[0]
            # get final layer output
            out_raw = torch.stack(out_raw)[-1]
            # mean pooling
            out = out_raw.mean(axis=1)
            embeds.append(out)
        return torch.cat(embeds)


if __name__ == "__main__":
    torchaudio_model = Model("mean")
    torchaudio_model.eval()
    waveform = torch.rand((16_000))
    x = waveform.unsqueeze(0)
    a = torchaudio_model(x)
