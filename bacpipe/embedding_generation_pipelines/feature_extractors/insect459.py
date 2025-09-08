import torch
from types import SimpleNamespace
import yaml
import torch
from .insect66 import SpectrogramCNN
from ..utils import ModelBaseClass

SAMPLE_RATE = 44100
LENGTH_IN_SAMPLES = int(5.5 * SAMPLE_RATE)


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        with open(f"{self.model_base_path}/insect66/config_insecteffnet.yaml", "rt") as infp:
            cfg = SimpleNamespace(**yaml.safe_load(infp))

        checkpoint = torch.load(
            f"{self.model_base_path}/insect459/last-v3-insecteffnet459-mel-mambo.ckpt",
            weights_only=False,
            map_location=self.device,
        )
        state_dict = {
            k.replace("model.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if not k == "loss_fn.weight"
        }

        self.model = SpectrogramCNN(cfg)
        self.model.load_state_dict(state_dict)

    def preprocess(self, audio):
        audio = audio[:, None, :]

        # (bs, channel, mel, time)
        return self.model.wav2timefreq(audio)

    @torch.inference_mode()
    def __call__(self, input):
        self.model.block_features = self.model.backbone.blocks(
            self.model.backbone.bn1(self.model.backbone.conv_stem(input))
        )

        self.model.embeddings = self.model.backbone.global_pool(
            self.model.backbone.bn2(
                self.model.backbone.conv_head(self.model.block_features)
            )
        )

        return self.model.embeddings
