import tensorflow_hub as hub
import numpy as np
from .utils import ModelBaseClass


SAMPLE_RATE = 16000
LENGH_IN_SAMPLES = int(0.96 * SAMPLE_RATE)


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGH_IN_SAMPLES)
        self.model = hub.load("bacpipe/models/vggish")

    def preprocess(self, audio):
        return (audio * 32767).astype(np.int16).reshape(1, -1).squeeze()

    def __call__(self, input):
        return self.model(input).numpy()
