from .batdetect2_clip_avg import Model, NUM_FEATURES, NUM_CLASSES
import numpy as np
import torch

class Model(Model):
    def __init__(self, **kwargs):
        super().__init__(use_detections=True, **kwargs)


    @torch.no_grad()
    def __call__(self, x):
        output = self.model(x)

        results, features = self.non_max_suppression(
            output,
            sampling_rate=np.array([self.sr] * x.shape[0]),
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
        self.output_class_scores = torch.stack(output_class_scores)

        return output_features

    def classifier_predictions(self, inference_results):
        return self.output_class_scores

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
