from ml_collections import config_dict
from bacpipe.model_utils.perch_chirp.chirp.inference.embed_lib import EmbedFn
from bacpipe.model_utils.perch_chirp.chirp.projects.zoo.models import (
    get_preset_model_config,
)
import tensorflow as tf

from .utils import ModelBaseClass

SAMPLE_RATE = 32000
LENGH_IN_SAMPLES = 160000


class Model(ModelBaseClass):
    def __init__(
        self, model_choice="perch_8", sr=SAMPLE_RATE, segment_length=LENGH_IN_SAMPLES
    ):
        super().__init__(sr=sr, segment_length=segment_length)

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
        self.model = embed_fn.embedding_model.embed

    def preprocess(self, audio):
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    def __call__(self, input):
        return self.model(input).embeddings
