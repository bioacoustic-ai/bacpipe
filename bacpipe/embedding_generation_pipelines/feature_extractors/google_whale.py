from .perch_bird import Model
import numpy as np

SAMPLE_RATE = 24_000
LENGH_IN_SAMPLES = 50_000


class Model(Model):
    def __init__(self):
        super().__init__(
            sr=SAMPLE_RATE,
            segment_length=LENGH_IN_SAMPLES,
            model_choice="multispecies_whale",
        )

    def __call__(self, input):
        embeds = []
        for frame in input:
            embeds.append(self.model(frame).embeddings.squeeze())
        return np.array(embeds)
