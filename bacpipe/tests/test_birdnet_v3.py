"""
BirdNET v3: predictions computed directly from audio versus predictions
computed from embeddings with ``classifier_predictions``.

Two routes are compared on the audio files of ``tests/test_data``:

direct
    audio frames -> BirdNET v3 -> (predictions, embeddings), run frame batch by
    frame batch on the underlying ONNX model.
indirect
    ``Embedder.get_embeddings_from_model(audio_file)`` -> embeddings ->
    ``Model.classifier_predictions(embeddings)``.

The indirect route is checked in two situations:

warm
    straight after the embedding step, as in the pipeline. Only the last
    batch of frames is still cached by the model, so its rows must be exact
    and the other rows are classified from the embeddings alone.
cold
    nothing is cached (e.g. embeddings loaded from disk), so every row is
    classified from the embeddings alone.

The BirdNET v3 head also reads backbone feature maps that are not part of the
embeddings, so embedding-only predictions can only be close, not identical.
The tolerances below are set from the values measured on the test audio.

Caution : test functions vibe-coded with CLAUDE Sonnet 5.5
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import bacpipe

MODEL_NAME = "birdnet_v3"
N_CLASSES = 11560
EMBEDDING_DIM = 1280
SEGMENT_SECONDS = 3
SCORE_THRESHOLD = 0.3

TEST_DATA = Path(bacpipe.__file__).parent / "tests" / "test_data"
CHECKPOINT = Path(bacpipe.settings.model_base_path) / MODEL_NAME / "model.onnx"

# annotation name -> BirdNET v3 label, where the taxonomies differ
ANNOTATION_TO_LABEL = {
    "Common Chaffinch": "Eurasian Chaffinch",
    "Eurasian Blackbird": "Common Blackbird",
    "Eurasian Kestrel": "Common Kestrel",
}

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="birdnet_v3 checkpoint not available"
)


def audio_files():
    return sorted(TEST_DATA.glob("audio/*/*"))


def relative_name(path):
    return path.relative_to(TEST_DATA).as_posix()


def labels_with_scores(probabilities, classes, threshold=SCORE_THRESHOLD):
    """Per segment, the (label, score) pairs above the threshold, best first."""
    out = []
    for row in probabilities:
        idx = torch.nonzero(row > threshold).flatten()
        idx = idx[torch.argsort(row[idx], descending=True)]
        out.append([(classes[i], float(row[i])) for i in idx])
    return out


@pytest.fixture(scope="module")
def embedder():
    return bacpipe.Embedder(
        model_name=MODEL_NAME,
        device="cpu",
        audio_dir=str(TEST_DATA),
    )


@pytest.fixture(scope="module")
def outputs(embedder):
    """For every test file: direct, warm and cold results plus batch info."""
    net = embedder.model
    result = {}
    for path in audio_files():
        frames = embedder.prepare_audio(path)

        direct_predictions, direct_embeddings = [], []
        for i in range(0, len(frames), net.batch_size):
            predictions, embeddings = net.model(
                np.array(frames[i : i + net.batch_size])
            )
            direct_predictions.append(predictions)
            direct_embeddings.append(embeddings)

        embeddings = embedder.get_embeddings_from_model(path)
        warm = net.classifier_predictions(torch.tensor(embeddings))

        net(frames[:1])  # the model now only remembers an unrelated frame
        cold = net.classifier_predictions(torch.tensor(embeddings))

        result[relative_name(path)] = {
            "direct": torch.cat(direct_predictions),
            "direct_embeddings": torch.cat(direct_embeddings),
            "embeddings": embeddings,
            "warm": warm,
            "cold": cold,
            "last_batch": len(frames) % net.batch_size or net.batch_size,
        }
    return result


class TestEmbeddingStep:
    def test_model_dimensions(self, embedder):
        assert len(embedder.model.classes) == N_CLASSES
        assert embedder.model.bool_classifier

    def test_embeddings_have_one_row_per_segment(self, outputs):
        for name, o in outputs.items():
            assert o["embeddings"].shape == (len(o["direct"]), EMBEDDING_DIM), name

    def test_embedder_matches_direct_embeddings(self, outputs):
        for name, o in outputs.items():
            assert np.allclose(
                o["embeddings"], o["direct_embeddings"].numpy(), atol=1e-3
            ), name


class TestIndirectPredictionsShape:
    @pytest.mark.parametrize("route", ["warm", "cold"])
    def test_shape_dtype_and_range(self, outputs, route):
        for name, o in outputs.items():
            p = o[route]
            assert p.shape == o["direct"].shape == (len(o["embeddings"]), N_CLASSES), name
            assert p.dtype == torch.float32
            assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


class TestWarmRoute:
    def test_last_batch_is_exact(self, outputs):
        for name, o in outputs.items():
            n = o["last_batch"]
            assert torch.allclose(o["warm"][-n:], o["direct"][-n:], atol=1e-6), name

    def test_files_within_one_batch_are_exact(self, embedder, outputs):
        small = [
            n for n, o in outputs.items() if len(o["direct"]) <= embedder.model.batch_size
        ]
        assert small, "expected at least one file that fits in a single batch"
        for name in small:
            assert torch.allclose(
                outputs[name]["warm"], outputs[name]["direct"], atol=1e-6
            ), name

    def test_other_rows_are_close(self, outputs):
        for name, o in outputs.items():
            diff = (o["warm"] - o["direct"]).abs()
            assert float(diff.max()) < 0.25, name


class TestColdRouteVersusAudio:
    """Embedding-only predictions compared with the predictions from audio."""

    @pytest.fixture(scope="module")
    def pooled(self, outputs):
        direct = torch.cat([o["direct"] for o in outputs.values()])
        cold = torch.cat([o["cold"] for o in outputs.values()])
        return direct, cold

    def test_top1_label_agrees_on_confident_segments(self, pooled):
        direct, cold = pooled
        confident = direct.max(1).values > SCORE_THRESHOLD
        assert confident.sum() >= 30
        agree = (direct.argmax(1) == cold.argmax(1))[confident].float().mean()
        assert float(agree) >= 0.9

    def test_top5_labels_overlap(self, pooled):
        direct, cold = pooled
        overlap = np.mean(
            [
                len(set(a.tolist()) & set(b.tolist())) / 5
                for a, b in zip(direct.topk(5).indices, cold.topk(5).indices)
            ]
        )
        assert overlap >= 0.8

    def test_labels_above_threshold_are_recovered(self, outputs, embedder):
        classes = list(embedder.model.classes)
        recalls = []
        for o in outputs.values():
            direct = labels_with_scores(o["direct"], classes)
            cold = labels_with_scores(o["cold"], classes)
            for d, c in zip(direct, cold):
                if d:
                    found = {label for label, _ in c}
                    recalls.append(np.mean([label in found for label, _ in d]))
        assert len(recalls) >= 30
        assert np.mean(recalls) >= 0.9

    def test_scores_of_the_direct_top1_label_are_close(self, pooled):
        direct, cold = pooled
        top = direct.argmax(1, keepdim=True)
        diff = (cold.gather(1, top) - direct.gather(1, top)).abs().flatten()
        assert float(diff.mean()) < 0.05
        assert float(diff.quantile(0.95)) < 0.15
        assert float(diff.max()) < 0.25

    def test_error_on_relevant_classes_is_small(self, pooled):
        direct, cold = pooled
        relevant = (direct > 0.05) | (cold > 0.05)
        assert float((direct - cold).abs()[relevant].mean()) < 0.1


class TestAnnotations:
    """Annotated species should be found by both routes, with similar rates."""

    @pytest.fixture(scope="module")
    def hits(self, outputs, embedder):
        classes = list(embedder.model.classes)
        annotations = pd.read_csv(TEST_DATA / "annotations.csv")
        rows = []
        for _, a in annotations.iterrows():
            label = ANNOTATION_TO_LABEL.get(a["label:species"], a["label:species"])
            if label not in classes or a["audiofilename"] not in outputs:
                continue
            o = outputs[a["audiofilename"]]
            segments = [
                i
                for i in range(len(o["direct"]))
                if i * SEGMENT_SECONDS >= a["start"]
                and (i + 1) * SEGMENT_SECONDS <= a["end"]
            ]
            for i in segments:
                j = classes.index(label)
                rows.append(
                    {
                        route: int(j in o[route][i].topk(5).indices.tolist())
                        for route in ("direct", "warm", "cold")
                    }
                )
        return pd.DataFrame(rows)

    def test_enough_annotated_segments(self, hits):
        assert len(hits) >= 30

    def test_audio_route_finds_the_annotated_species(self, hits):
        assert hits["direct"].mean() >= 0.8

    @pytest.mark.parametrize("route", ["warm", "cold"])
    def test_embedding_route_is_not_worse_than_audio_route(self, hits, route):
        assert hits[route].mean() >= hits["direct"].mean() - 0.1

    @pytest.mark.parametrize("route", ["warm", "cold"])
    def test_routes_agree_segment_by_segment(self, hits, route):
        assert (hits[route] == hits["direct"]).mean() >= 0.9

    def test_chaffinch_is_top1_in_both_routes(self, outputs, embedder):
        classes = list(embedder.model.classes)
        j = classes.index("Eurasian Chaffinch")
        o = outputs["audio/FewShot/CHE_01_20190101_163410.wav"]
        annotations = pd.read_csv(TEST_DATA / "annotations.csv")
        annotations = annotations[
            annotations["audiofilename"].str.contains("CHE_01")
            & (annotations["label:species"] == "Common Chaffinch")
        ]
        start, end = annotations["start"].min(), annotations["end"].max()
        segments = [
            i
            for i in range(len(o["direct"]))
            if i * SEGMENT_SECONDS >= start and (i + 1) * SEGMENT_SECONDS <= end
        ]
        assert len(segments) >= 10
        for route in ("direct", "warm", "cold"):
            top1 = o[route][segments].argmax(1)
            assert float((top1 == j).float().mean()) >= 0.7, route
