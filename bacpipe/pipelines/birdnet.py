import tensorflow as tf
import numpy as np
import librosa as lb
import logging
logger = logging.getLogger('bacpipe')
logger.setLevel(level=logging.DEBUG)

SAMPLE_RATE = 48000
LENGTH_IN_SAMPLES = 144000

from .utils import ModelBaseClass

class Model(ModelBaseClass):
    
    def __init__(self):
        super().__init__(sr = SAMPLE_RATE, segment_length = LENGTH_IN_SAMPLES)
        model = tf.keras.models.load_model('bacpipe/models/birdnet', 
                                                compile=False)
        self.model = tf.keras.Sequential(model.embeddings_model)

    def preprocess(self, audio):
        return tf.convert_to_tensor(audio, dtype=tf.float32)

        
    def __call__(self, input):
        return self.model(input)