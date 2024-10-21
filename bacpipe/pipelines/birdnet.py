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
        super().__init__()
        model = tf.keras.models.load_model('bacpipe/models/birdnet', 
                                                compile=False)
        self.model = tf.keras.Sequential(model.embeddings_model)

    def preprocess(self, audio):
        re_audio = lb.resample(audio, 
                                orig_sr = self.config['sr'], 
                                target_sr = SAMPLE_RATE)
        num = np.ceil(len(re_audio) / LENGTH_IN_SAMPLES)
        # zero pad in case the end is reached
        re_audio = [*re_audio, *np.zeros([int(num * LENGTH_IN_SAMPLES - len(re_audio))])]
        wins = np.array(re_audio).reshape([int(num), LENGTH_IN_SAMPLES])

        return tf.convert_to_tensor(wins, dtype=tf.float32)

        
    def __call__(self, input):
        return self.model(input)