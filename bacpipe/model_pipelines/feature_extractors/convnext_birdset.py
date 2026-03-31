import torch
from transformers import AutoModelForImageClassification
import pandas as pd


SAMPLE_RATE = 32_000
LENGTH_IN_SAMPLES = 160_000

from bacpipe.model_pipelines.model_specific_utils.convnext_birdset.preprocess import ConvNextPreProcess
from ..model_utils import ModelBaseClass


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        model = AutoModelForImageClassification.from_pretrained(
            "DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
            trust_remote_code=True,
        )
        preproc = ConvNextPreProcess(SAMPLE_RATE, device=self.device)
        self.preprocessor = preproc.preprocess
        
        self.model = model.convnext.to(self.device)
        self.classifier = model.classifier.to(self.device)
        
        self.model.eval()

        id2label = model.config.id2label
        ebird2name = pd.read_csv(
            self.model_utils_base_path / "perch_v2/perch_hoplite/eBird2name.csv"
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

    def __call__(self, x):
        results = self.model(x)
        return results.pooler_output

    def classifier_predictions(self, embeddings):
        logits = self.classifier(embeddings)
        return torch.sigmoid(logits).detach()
