import numpy as np

from .perch_bird import Model

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(1 * SAMPLE_RATE)


class Model(Model):
    def __init__(self, **kwargs):
        """
        Initialize the VGGish model.
        """
        super().__init__(
            sr=SAMPLE_RATE,
            segment_length=LENGTH_IN_SAMPLES,
            model_choice="vggish",
            **kwargs,
        )

    def __call__(self, input):
        """
        Run the VGGish model on the input frames and concatenate the
        resulting embeddings.

        Parameters
        ----------
        input : list
            list of input audio frames

        Returns
        -------
        np.array
            concatenated embeddings for all input frames
        """
        for i, frame in enumerate(input):
            results = self.model(frame)
            if i == 0:
                cumulative_embeds = results.embeddings.squeeze()
            else:
                cumulative_embeds = np.vstack(
                    [cumulative_embeds, results.embeddings.squeeze()]
                )

        return cumulative_embeds
