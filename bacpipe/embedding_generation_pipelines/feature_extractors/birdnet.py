import tensorflow as tf
import keras
import pandas as pd
import numpy as np

SAMPLE_RATE = 48000
LENGTH_IN_SAMPLES = 144000

from ..utils import ModelBaseClass


class Model(ModelBaseClass):

    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        self.model = tf.keras.models.load_model(
            self.model_base_path / "birdnet/birdnetv2.4.keras", compile=False
        )
        
        loaded_preprocessor = tf.saved_model.load(
            self.model_base_path / "birdnet/BirdNET_Preprocessor",
        )
        self.preprocessor = lambda x: (
            loaded_preprocessor.signatures['serving_default'](x)['concatenate']
            )
        
        all_classes = pd.read_csv(
            self.model_utils_base_path /
            "birdnet/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt",
            header=None,
        )
        self.classes = [s.split("_")[-1] for s in all_classes.values.squeeze()]
        
        self.embeds = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output,
            name="embeddings_model"
        )
        
        x = keras.Input(shape=self.model.layers[-3].output.shape[1:])
        y = self.model.layers[-2](x)
        y = self.model.layers[-1](y)
        self.classifier = tf.keras.Model(x, y, name="classifier_model")

    def preprocess(self, audio):
        audio = audio.cpu()
        for idx in range(0, audio.shape[0], 511):
            if idx == 0:
                processed = self.preprocessor(tf.convert_to_tensor(audio[:511], 
                                                                   dtype=tf.float32)).numpy()
            else:
                processed = np.vstack([
                    processed,
                    self.preprocessor(tf.convert_to_tensor(audio[idx:idx+511], 
                                                        dtype=tf.float32)).numpy()
                    ])
        return tf.convert_to_tensor(processed, dtype=tf.float32)

    def __call__(self, input, return_class_results=False):
        if not return_class_results:
            return self.embeds(input, training=False)
        else:
            embeds = self.embeds(input, training=False)
            class_preds = self.classifier_predictions(embeds)
            return embeds, class_preds

    def classifier_predictions(self, inferece_results):
        logits = self.classifier(inferece_results).numpy()
        return tf.nn.sigmoid(logits).numpy()
