from functools import partial

import numpy as np
import torch

from ..model_utils import ModelBaseClass

SAMPLE_RATE = 256_000
LENGTH_IN_SAMPLES = 256_000
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
        """
        Initialize the BatDetect2 detection-averaged model.

        Parameters
        ----------
        segment_duration : float
            duration of each audio segment in seconds
        detection_threshold : float
            confidence threshold for detections
        top_k_detections : int or None
            maximum number of detections to average per segment
        **kwargs
            additional keyword arguments passed to the base class
        """
        super().__init__(
            sr=SAMPLE_RATE,
            segment_length=int(segment_duration * SAMPLE_RATE),
            **kwargs,
        )

        import batdetect2.detector.post_process as pp
        from batdetect2 import api

        self.detection_threshold = detection_threshold
        self.top_k_detections = top_k_detections

        self.config = api.get_config(
            detection_threshold=self.detection_threshold
        )
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
        """
        Generate a spectrogram for each audio segment.

        Parameters
        ----------
        audio : torch.Tensor
            audio samples to be preprocessed

        Returns
        -------
        torch.Tensor
            stacked spectrograms
        """
        if audio.device.type == "cuda":
            segments = audio.cpu().numpy()
        else:
            segments = audio.numpy()
        # NOTE: Need to pre-process each segment separately
        spectrograms = [
            self.generate_spectrogram(segment) for segment in segments
        ]
        return torch.stack(spectrograms, axis=0).squeeze()

    @torch.no_grad()
    def __call__(self, x, return_class_results=False):
        """
        Run the model on the input spectrograms and average detection features.

        Parameters
        ----------
        x : torch.Tensor
            preprocessed spectrograms
        return_class_results : bool
            whether to also return class scores

        Returns
        -------
        torch.Tensor or tuple
            detection-averaged features, or a tuple of
            (features, class scores) when return_class_results is True
        """
        x = x.unsqueeze(1)
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

    # def classifier_predictions(self, inference_results):
    #     # NOTE: This method is left unimplemented. Since 'inference_results'
    #     # are averaged across several detections to map to the single-feature
    #     # interface, running a classifier on these aggregated features won't
    #     # produce the intended results.
    #     raise NotImplementedError(
    #         "Classifier predictions are invalid for averaged features."
    #     )


def get_mean_detection_features(
    results,
    features,
    top_k=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate detection features by averaging and take max class scores.

    Parameters
    ----------
    results : dict
        detection results with "det_probs" and "class_probs" keys
    features : np.array
        per-detection features
    top_k : int or None
        if given, only use the top-k detections by detection score

    Returns
    -------
    tuple of torch.Tensor
        mean features and max class scores
    """
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
