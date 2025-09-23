from bacpipe.model_specific_utils.perch_v2.perch_hoplite.zoo.model_configs import load_model_by_name
import tensorflow as tf
import numpy as np
import pandas as pd

from ..utils import ModelBaseClass

SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = 160000


class Model(ModelBaseClass):
    def __init__(
        self, model_choice="perch_v2_cpu", sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
    ):
        super().__init__(sr=sr, segment_length=segment_length, **kwargs)
        
        if self.device == 'cuda' and model_choice.startswith('perch_v2'):
            model_choice = 'perch_v2'
        mod = load_model_by_name(model_choice)

        self.model = mod.embed
        if not hasattr(self, 'class_label_key'):
            self.class_label_key = "labels"
        
        if model_choice in ['vggish']:
            return
        elif not model_choice in ["multispecies_whale"]:
            self.class_list = mod.class_list[self.class_label_key].classes
            self.ebird2name = pd.read_csv(
                self.model_utils_base_path /
                "perch_v2/perch_hoplite/eBird2name.csv"
            )
            self.classes = self.class_list
            self.classes = [
                (
                    self.ebird2name["English name"][
                        self.ebird2name.species_code == cls
                    ].iloc[0]
                    if cls in self.ebird2name.species_code.values
                    else cls
                )
                for cls in self.classes
            ]
        else:
            self.class_list = mod.class_list
            
        if model_choice.startswith('perch_v2'):
            self.class_label_key = 'label'

    def preprocess(self, audio):
        audio = audio.cpu()
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    def __call__(self, input, return_class_results=False):
        results = self.model(input)
        embeddings = results.embeddings
        if return_class_results:
            cls_vals = self.classifier_predictions(results.logits[self.class_label_key])
            return embeddings, cls_vals
        else:
            return embeddings

    def classifier_predictions(self, inferece_results):
        return tf.nn.sigmoid(inferece_results).numpy()
