import numpy as np
import torch

from ..utils import ModelBaseClass

SAMPLE_RATE = 256_000

LENGTH_IN_SAMPLES = int(1 * SAMPLE_RATE)


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(
            sr=SAMPLE_RATE,
            segment_length=LENGTH_IN_SAMPLES,
            **kwargs,
        )

        from batdetect2 import api

        self.model, _ = api.load_model()
        self.generate_spectrogram = api.generate_spectrogram
        self.postprocess = api.postprocess

    def preprocess(self, audio):
        segments = audio.numpy()
        # NOTE: Need to pre-process each segment separately
        spectrograms = [
            self.generate_spectrogram(
                segment,
                samp_rate=self.sr,
                device=self.device,  # type: ignore
            )
            for segment in segments
        ]
        return torch.from_numpy(np.concatenate(spectrograms, axis=0))

    def __call__(self, x):
        from batdetect2.types import ModelOutput

        output = self.model(x)

        num_segments = x.shape[0]

        ret = []

        # NOTE: There's a bug in bd2 so outputs need to be post-processed
        # seperately too.
        for index in range(num_segments):
            segment_output = ModelOutput(
                pred_det=output.pred_det[[index]],
                pred_size=output.pred_size[[index]],
                pred_class=output.pred_class[[index]],
                pred_class_un_norm=output.pred_class_un_norm[[index]],
                features=output.features[[index]],
            )

            detections, features = self.postprocess(segment_output)
            ret.append(torch.tensor(features.mean(axis=0)))

        return torch.stack(ret)
