import tensorflow as tf
import pandas as pd

SAMPLE_RATE = 48000
LENGTH_IN_SAMPLES = 144000

from ..utils import ModelBaseClass


class Model(ModelBaseClass):

    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        model = tf.keras.models.load_model(
            self.model_base_path / "birdnet", compile=False
        )
        all_classes = pd.read_csv(
            self.model_utils_base_path /
            "birdnet/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt",
            header=None,
        )
        self.classes = [s.split("_")[-1] for s in all_classes.values.squeeze()]
        self.classifier = tf.keras.Sequential(model.model.layers[-2:])
        self.embeds = tf.keras.Sequential(model.embeddings_model)

    def preprocess(self, audio):
        audio = audio.cpu()
        # if i want to change this to actually do the mel specs, i would need
        # to compute everything up to self.model.layers[0].layers[:4] and
        # then embed using self.model.layers[0].layers[5:]
        return tf.convert_to_tensor(audio, dtype=tf.float32)

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
