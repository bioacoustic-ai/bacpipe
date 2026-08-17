import numpy as np

from .perch_bird import Model

SAMPLE_RATE = 24_000
LENGTH_IN_SAMPLES = 50_000


class Model(Model):
    def __init__(self, **kwargs):
        """
        Initialize the Google Whale (multispecies whale) model.
        """
        super().__init__(
            sr=SAMPLE_RATE,
            segment_length=LENGTH_IN_SAMPLES,
            model_choice="multispecies_whale",
            **kwargs,
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
        """
        Run the Google whale model on each frame of input audio.

        Parameters
        ----------
        input : array-like
            iterable of audio frames to embed
        return_class_results : bool
            whether to return class results (currently unused)

        Returns
        -------
        np.array
            embeddings for each input frame
        """
        # if return_class_results:
        #     embeds, class_preds = [], []
        embeds = []
        self.logits = []
        for frame in input:
            results = self.model(frame)
            self.logits.append(list(results.logits.values()))
            embeds.append(results.embeddings.squeeze())
        return np.array(embeds)

    def classifier_predictions(self, embeddings):
        """
        Return the class logits stored during the last call.

        Parameters
        ----------
        embeddings : np.array
            embeddings from the last model call (unused)

        Returns
        -------
        np.array
            class logits for each frame
        """
        return np.array(self.logits).squeeze()
