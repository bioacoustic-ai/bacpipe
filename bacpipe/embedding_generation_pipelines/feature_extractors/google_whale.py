import tensorflow as tf
import numpy as np

from .perch_bird import Model

SAMPLE_RATE = 24_000
LENGTH_IN_SAMPLES = 50_000


class Model(Model):
    def __init__(self, **kwargs):
        super().__init__(
            sr=SAMPLE_RATE,
            segment_length=LENGTH_IN_SAMPLES,
            model_choice="multispecies_whale",
            **kwargs
        )

        self.abbrev2label = {
            "Mn": "Humpback",
            "Oo": "Orca",
            "Be": "Bryde's",
            "Ba": "Minke",
            "Bm": "Blue",
            "Bp": "Fin",
            "Eg": "Right (Atlantic)",
            "Upcall": "Right (Pacific, upcall)",
            "Gunshot": "Right (Pacific, gunshot)",
            "Echolocation": "Orca echolocation",
            "Whistle": "Orca whistle",
            "Call": "Orca call",
        }
        self.class_label_key = "multispecies_whale"
        self.classes = [self.abbrev2label[v] for v in self.class_list.classes]

    def __call__(self, input, return_class_results=False):
        if return_class_results:
            embeds, class_preds = [], []
        embeds = []
        for frame in input:
            results = self.model.embed(frame)
            if return_class_results:
                cls_vals = self.classifier_predictions(results)
                class_preds.append(cls_vals)
            embeds.append(results.embeddings.squeeze())
        class_preds = np.array(class_preds).squeeze()

        if return_class_results:
            return np.array(embeds), class_preds
        else:
            return np.array(embeds)

    def classifier_predictions(self, inferece_results):
        return tf.nn.sigmoid(inferece_results.logits[self.class_label_key]).numpy()
