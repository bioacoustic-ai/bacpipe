import tensorflow_hub as hub
import numpy as np
from .utils import ModelBaseClass
import tensorflow as tf


SAMPLE_RATE = 16000
LENGH_IN_SAMPLES = int(0.96 * SAMPLE_RATE)


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGH_IN_SAMPLES)
        self.model = hub.load("bacpipe/models/vggish")

    def preprocess(self, audio):
        return tf.reshape(tf.convert_to_tensor(audio * 32767, dtype=tf.int16), (1, -1))

    def __call__(self, input):
        return self.model(input[0].numpy())
