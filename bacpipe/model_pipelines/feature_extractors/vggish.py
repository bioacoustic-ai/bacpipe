import numpy as np

from .perch_v2 import Model

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(1 * SAMPLE_RATE)


class Model(Model):
    def __init__(self, **kwargs):
        super().__init__(
            sr=SAMPLE_RATE, 
            segment_length=LENGTH_IN_SAMPLES, 
            model_choice="vggish",
            **kwargs
            )

    def __call__(self, input):
        for i, frame in enumerate(input):
            results = self.model(frame)
            if i == 0:
                cumulative_embeds = results.embeddings.squeeze()
            else:
                cumulative_embeds = np.vstack([cumulative_embeds, results.embeddings.squeeze()])
        
        return cumulative_embeds
