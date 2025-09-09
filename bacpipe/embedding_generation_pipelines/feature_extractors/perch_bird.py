from ml_collections import config_dict
from bacpipe.model_specific_utils.perch_chirp.chirp.inference.embed_lib import EmbedFn
from bacpipe.model_specific_utils.perch_chirp.chirp.projects.zoo.models import (
    get_preset_model_config,
)
import tensorflow as tf
import pandas as pd

from ..utils import ModelBaseClass

SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = 160000


class Model(ModelBaseClass):
    def __init__(
        self, model_choice="perch_8", sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
    ):
        super().__init__(sr=sr, segment_length=segment_length, **kwargs)

        config = config_dict.ConfigDict()
        config.embed_fn_config = config_dict.ConfigDict()
        config.embed_fn_config.model_config = config_dict.ConfigDict()
        model_key, embedding_dim, model_config = get_preset_model_config(model_choice)
        config.embed_fn_config.model_key = model_key
        config.embed_fn_config.model_config = model_config

        # Only write embeddings to reduce size.
        config.embed_fn_config.write_embeddings = True
        config.embed_fn_config.write_logits = False
        config.embed_fn_config.write_separated_audio = False
        config.embed_fn_config.write_raw_audio = False
        config.embed_fn_config.file_id_depth = 1
        embed_fn = EmbedFn(**config.embed_fn_config)
        embed_fn.setup()
        self.model = embed_fn.embedding_model
        self.class_list = embed_fn.embedding_model.class_list
        self.class_label_key = "label"
        if not model_choice == "multispecies_whale":
            self.ebird2name = pd.read_csv(
                self.model_utils_base_path /
                "perch_chirp/chirp/eBird2name.csv"
            )
            self.classes = self.class_list[self.class_label_key].classes
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

    def preprocess(self, audio):
        audio = audio.cpu()
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    def __call__(self, input, return_class_results=False):
        results = self.model.embed(input)
        embeddings = results.embeddings
        if return_class_results:
            cls_vals = self.classifier_predictions(results)
            return embeddings, cls_vals
        else:
            return embeddings

    def classifier_predictions(self, inferece_results):
        return tf.nn.sigmoid(inferece_results.logits[self.class_label_key]).numpy()
