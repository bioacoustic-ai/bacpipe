import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForSequenceClassification,
)
import pandas as pd


SAMPLE_RATE = 32_000
LENGTH_IN_SAMPLES = 160_000

from ..utils import ModelBaseClass


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
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
            self.model_utils_base_path / "perch_chirp/chirp/eBird2name.csv"
        )
        self.classes = [
            (
                ebird2name["English name"][ebird2name.species_code == cls].iloc[0]
                if cls in ebird2name.species_code.values
                else cls
            )
            for cls in id2label.values()
        ]


    def preprocess(self, audio):
        return self.preprocessor(audio)

    def __call__(self, x, return_class_results=False):
        if not return_class_results:
            return self.model(x).pooler_output
        else:
            embeds = self.model(x)
            class_preds = self.classifier_predictions(embeds.last_hidden_state)
            return embeds.pooler_output, class_preds

    def classifier_predictions(self, embeddings):
        logits, _ = self.classifier(embeddings)
        return torch.sigmoid(logits).detach()
