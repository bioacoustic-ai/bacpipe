from bacpipe.model_pipelines.model_specific_utils.perch_v2.perch_hoplite.zoo.model_configs import (
    load_model_by_name,
)
import tensorflow as tf
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("bacpipe")

tf.keras.backend.clear_session()

from ..model_utils import ModelBaseClass

SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = 160000


class Model(ModelBaseClass):
    def __init__(
        self,
        model_choice="perch_8",
        sr=SAMPLE_RATE,
        segment_length=LENGTH_IN_SAMPLES,
        **kwargs,
    ):
        """
        Initialize the Perch model.

        Parameters
        ----------
        model_choice : str
            name of the model configuration to load
        sr : int
            sample rate
        segment_length : int
            length of each audio segment in samples
        **kwargs
            additional keyword arguments passed to the base class
        """
        super().__init__(sr=sr, segment_length=segment_length, **kwargs)

        if model_choice == "vggish":
            self.bool_classifier = False

        if self.device == "cuda" and model_choice.startswith("perch_v2"):
            if len(tf.config.list_physical_devices("GPU")) > 0:
                model_choice = "perch_v2"
        mod = load_model_by_name(model_choice)

        self.model = mod.embed

        self.class_label_key = "label"

        if model_choice in ["vggish"]:
            return
        elif not model_choice in ["multispecies_whale"]:
            self.class_list = mod.class_list[self.class_label_key].classes
            self.ebird2name = pd.read_csv(
                self.model_utils_base_path
                / "perch_v2/perch_hoplite/eBird2name.csv"
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

    def preprocess(self, audio):
        """
        Convert the audio tensor to a TensorFlow tensor.

        Parameters
        ----------
        audio : torch.Tensor
            audio samples to be preprocessed

        Returns
        -------
        tf.Tensor
            audio tensor
        """
        audio = audio.cpu()
        return tf.convert_to_tensor(audio, dtype=tf.float32)

    def __call__(self, input):
        """
        Run the embedding model on the input.

        Parameters
        ----------
        input : tf.Tensor
            preprocessed audio

        Returns
        -------
        tf.Tensor
            embeddings
        """
        self.results = self.model(input)
        return self.results.embeddings.squeeze(1)

    def classifier_predictions(self, embeddings):
        """
        Return the sigmoid class logits from the last call.

        Parameters
        ----------
        embeddings : tf.Tensor
            embeddings from the last model call (unused)

        Returns
        -------
        np.array
            sigmoid class logits
        """
        inferece_results = self.results.logits[self.class_label_key]
        return tf.nn.sigmoid(inferece_results).numpy()

