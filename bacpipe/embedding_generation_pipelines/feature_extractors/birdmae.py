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
        )
        self.model.to(self.device)
        self.model.eval()
        self.preproc_batch_size = 5114

    def preprocess(self, audio):
        batched_windows = torch.Tensor([])
        for i in range(0, len(audio), self.preproc_batch_size):
            batched_windows = torch.cat(
                [batched_windows, 
                 self.audio_processor(
                     audio[i:i+self.preproc_batch_size]
                     ).unsqueeze(1)]
                )
        return batched_windows.squeeze(1).to(self.device)

    @torch.inference_mode()
    def __call__(self, input):
        return self.model(input).last_hidden_state
