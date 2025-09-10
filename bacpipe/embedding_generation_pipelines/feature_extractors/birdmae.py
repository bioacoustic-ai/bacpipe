import torch
from transformers import AutoFeatureExtractor, AutoModel
from ..utils import ModelBaseClass

SAMPLE_RATE = 32_000
LENGTH_IN_SAMPLES = 160_000


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)

        self.audio_processor = AutoFeatureExtractor.from_pretrained(
            "DBD-research-group/Bird-MAE-Base",
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            "DBD-research-group/Bird-MAE-Huge",
            trust_remote_code=True,
            dtype="auto",
        )
        self.model.to(self.device)
        self.audio_processor.to(self.device)
        self.model.eval()

    def preprocess(self, audio):
        processed_audio = self.audio_processor(audio).unsqueeze(1)
        return processed_audio.to(self.device)

    @torch.inference_mode()
    def __call__(self, input):
        return self.model(input).last_hidden_state
