import tensorflow_hub as hub
import numpy as np
from ..utils import ModelBaseClass
import tensorflow as tf


SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(0.96 * SAMPLE_RATE)


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        self.model = hub.load(str(self.model_base_path / "vggish"))

    def preprocess(self, audio):
        audio = audio.cpu()
        return tf.reshape(tf.convert_to_tensor(audio * 32767, dtype=tf.int16), (1, -1))

    def __call__(self, input):
        return self.model(input[0].cpu().numpy())
