from functools import partial

import numpy as np
import torch

from ..utils import ModelBaseClass

SAMPLE_RATE = 256_000
DEFAULT_SEGMENT_DURATION = 1
DEFAULT_DETECTION_THRESHOLD = 0.3
NUM_FEATURES = 32
NUM_CLASSES = 17


class Model(ModelBaseClass):
    def __init__(
        self,
        segment_duration=DEFAULT_SEGMENT_DURATION,
        detection_threshold=DEFAULT_DETECTION_THRESHOLD,
        top_k_detections=None,
        **kwargs,
    ):
        super().__init__(
            sr=SAMPLE_RATE,
            segment_length=int(segment_duration * SAMPLE_RATE),
            **kwargs,
        )

        import batdetect2.detector.post_process as pp
        from batdetect2 import api

        self.detection_threshold = detection_threshold
        self.top_k_detections = top_k_detections

        self.config = api.get_config(detection_threshold=self.detection_threshold)
        self.model, _ = api.load_model(device=self.device)  # type: ignore

        self.generate_spectrogram = partial(
            api.generate_spectrogram,
            config=self.config,
            samp_rate=SAMPLE_RATE,
            device=self.device,
        )
        self.non_max_suppression = partial(
            pp.run_nms,
            params={
                "nms_kernel_size": self.config["nms_kernel_size"],
                "max_freq": self.config["max_freq"],
                "min_freq": self.config["min_freq"],
                "fft_win_length": self.config["fft_win_length"],
                "fft_overlap": self.config["fft_overlap"],
                "resize_factor": self.config["resize_factor"],
                "nms_top_k_per_sec": self.config["nms_top_k_per_sec"],
                "detection_threshold": self.detection_threshold,
            },
        )

        self.classes = self.config["class_names"]

    def preprocess(self, audio):
        segments = audio.numpy()
        # NOTE: Need to pre-process each segment separately
        spectrograms = [self.generate_spectrogram(segment) for segment in segments]
        return torch.from_numpy(np.concatenate(spectrograms, axis=0))

    @torch.no_grad()
    def __call__(self, x, return_class_results=False):
        output = self.model(x)

        results, features = self.non_max_suppression(
            output,
            sampling_rate=np.array([SAMPLE_RATE] * x.shape[0]),
        )

        output_features = []
        output_class_scores = []

        for res, feats in zip(results, features):
            feat, class_scores = get_mean_detection_features(
                res,
                feats,
                top_k=self.top_k_detections,
            )

            output_features.append(feat)
            output_class_scores.append(class_scores)

        output_features = torch.stack(output_features)
        output_class_scores = torch.stack(output_class_scores)

        if return_class_results:
            return output_features, output_class_scores

        return output_features

    def classifier_predictions(self, inference_results):
        # NOTE: This method is left unimplemented. Since 'inference_results'
        # are averaged across several detections to map to the single-feature
        # interface, running a classifier on these aggregated features won't
        # produce the intended results.
        raise NotImplementedError(
            "Classifier predictions are invalid for averaged features."
        )


def get_mean_detection_features(
    results,
    features,
    top_k=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    detection_scores = results["det_probs"]

    # NOTE: Last element is the background class
    class_scores = results["class_probs"][:-1]

    if len(detection_scores) == 0:
        return torch.zeros(NUM_FEATURES), torch.zeros(NUM_CLASSES)

    if top_k is not None:
        top_k = min(top_k, len(detection_scores))
        top_k_detections = np.argpartition(detection_scores, -top_k)[-top_k:]
        features = features[top_k_detections]
        class_scores = class_scores[:, top_k_detections]

    # NOTE: Batch dimension here is first
    mean_features = features.mean(axis=0)

    # NOTE: Batch dimension here is last
    max_class_scores = class_scores.max(axis=1)

    return torch.from_numpy(mean_features), torch.from_numpy(max_class_scores)
