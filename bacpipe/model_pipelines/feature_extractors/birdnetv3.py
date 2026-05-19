import torch
import keras
import pandas as pd
import numpy as np

SAMPLE_RATE = 32_000
LENGTH_IN_SAMPLES = 96_000

from ..model_utils import ModelBaseClass


class Model(ModelBaseClass):

    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        
        model_path = self.model_base_path / 'birdnetv3/BirdNET+_V3.0-preview3_Global_11K_FP32.pt'
        labels = self.model_base_path / 'birdnetv3/BirdNET+_V3.0-preview3_Global_11K_Labels.csv'
        device = torch.device(self.device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        self.classes = labels

    def preprocess(self, audio):
        return audio
    #     audio = audio.cpu()
    #     for idx in range(0, audio.shape[0], 511):
    #         if idx == 0:
    #             processed = self.preprocessor(tf.convert_to_tensor(audio[:511], 
    #                                                                dtype=tf.float32)).numpy()
    #         else:
    #             processed = np.vstack([
    #                 processed,
    #                 self.preprocessor(tf.convert_to_tensor(audio[idx:idx+511], 
    #                                                     dtype=tf.float32)).numpy()
    #                 ])
    #     return tf.convert_to_tensor(processed, dtype=tf.float32)

    def __call__(self, input):
        self.out = self.model(input)
        embeds, preds = self.out
        return embeds

    def classifier_predictions(self, embeddings):
        _, preds = self.out
        return preds