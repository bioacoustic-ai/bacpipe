import torch
import os
import logging

# Force Hugging Face to use PyTorch and ignore TensorFlow
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"


logger = logging.getLogger("bacpipe")

from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForSequenceClassification,
)
import pandas as pd

SAMPLE_RATE = 32_000
LENGTH_IN_SAMPLES = 160_000

from ..model_utils import ModelBaseClass


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        """
        Initialize the AudioProtoPNet model.
        """
        super().__init__(
            sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
        )
        self.batch_size = 4
        model = AutoModelForSequenceClassification.from_pretrained(
            "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
            trust_remote_code=True,
        )

        # optional: patch missing attribute if other code expects it
        if not hasattr(model, "incorrect_class_connection"):
            model.incorrect_class_connection = None

        self.preprocessor = AutoFeatureExtractor.from_pretrained(
            "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
            trust_remote_code=True,
        )

        self.model = model.model.backbone.to(self.device)
        self.classifier = model.head.to(self.device)

        self.model.eval()

        id2label = model.config.id2label
        ebird2name = pd.read_csv(
            self.model_utils_base_path
            / "perch_v2/perch_hoplite/eBird2name.csv"
        )
        self.classes = [
            (
                ebird2name["English name"][
                    ebird2name.species_code == cls
                ].iloc[0]
                if cls in ebird2name.species_code.values
                else cls
            )
            for cls in id2label.values()
        ]

    def preprocess(self, audio):
        """
        Preprocess the audio samples with the model's feature extractor.

        Parameters
        ----------
        audio : torch.Tensor or numpy.ndarray
            audio samples to be preprocessed (one row per window)

        Returns
        -------
        torch.Tensor
            preprocessor outputs
        """
        return self.preprocessor(audio)

    def __call__(self, x):
        """
        Run the backbone on the input.

        Parameters
        ----------
        x : dict
            preprocessed input

        Returns
        -------
        torch.Tensor
            pooler output embeddings
        """
        self.results = self.model(x)
        return self.results.pooler_output

    def classifier_predictions(self, embeddings):
        """
        Run the classifier head on the last hidden state.

        Parameters
        ----------
        embeddings : torch.Tensor
            embeddings from the last model call (unused)

        Returns
        -------
        torch.Tensor
            sigmoid class logits
        """
        if not hasattr(self, "results") or self.results is None:
            # The classifier head consumes the last hidden state of the
            # backbone, which only exists after a forward pass of the model
            # itself. This makes the offline path (``run_default_classifier``
            # on already-computed embeddings) impossible for this model.
            logger.warning(
                "AudioProtoPNet's classifier head needs the last hidden "
                "state of a backbone forward pass (`self.results`), which "
                "is only set when the model itself is called. Running the "
                "pretrained classifier offline on already-computed "
                "embeddings (e.g. `run_default_classifier`) is therefore "
                "not supported for this model. Please recompute the "
                "embeddings with `run_pretrained_classifier=True` so the "
                "classifier can be run while the backbone is being called."
            )
            raise AttributeError(
                "AudioProtoPNet.classifier_predictions() requires the last "
                "hidden state from a prior forward pass of the backbone "
                "(self.results); the classifier cannot be run offline on "
                "precomputed embeddings."
            )
        logits, _ = self.classifier(self.results.last_hidden_state)
        return torch.sigmoid(logits).detach()
