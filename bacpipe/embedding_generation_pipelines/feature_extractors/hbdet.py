import tensorflow as tf

from ..utils import ModelBaseClass

SAMPLE_RATE = 2000
LENGTH_IN_SAMPLES = 7755


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        loaded_model = tf.saved_model.load(self.model_base_path / "hbdet")
        self.model = loaded_model.signatures['serving_default']

    def preprocess(self, audio):
        return tf.convert_to_tensor(audio.cpu())

    def __call__(self, input):
        return self.model(input)['pool']
