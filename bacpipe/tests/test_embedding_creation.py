import sys
import numpy as np
import yaml
import importlib.resources as pkg_resources
from pathlib import Path

sys.path.insert(0, ".")

import bacpipe
from bacpipe.main import get_embeddings, embeds_array_without_noise
from bacpipe.generate_embeddings import Loader, Embedder
from bacpipe.embedding_evaluation.label_embeddings import (
    generate_annotations_for_classification_task,
    make_set_paths_func,
    ground_truth_by_model,
)
from bacpipe.embedding_evaluation.classification.classify import classification_pipeline
from bacpipe.embedding_evaluation.clustering.cluster import clustering


# -------------------------------------------------------------------------
# Load settings and config
# -------------------------------------------------------------------------
with pkg_resources.open_text(bacpipe, "settings.yaml") as f:
    settings = yaml.load(f, Loader=yaml.CLoader)

with pkg_resources.open_text(bacpipe, "config.yaml") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

settings["overwrite"] = True
settings["testing"] = True
kwargs = {**config, **settings}


# -------------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------------
embedding_dimensions = {
    "audiomae": 768,
    "audioprotopnet": 1024,
    "avesecho_passt": 768,
    "aves_especies": 768,
    "beats": 768,
    "birdaves_especies": 1024,
    "biolingual": 512,
    "birdnet": 1024,
    "birdmae": 1280,
    "hbdet": 2048,
    "insect66": 1280,
    "insect459": 1280,
    "mix2": 960,
    "naturebeats": 768,
    "perch_bird": 1280,
    "protoclr": 384,
    "rcl_fs_bsed": 2048,
    "surfperch": 1280,
    "google_whale": 1280,
    "vggish": 128,
}

needs_checkpoint = [
    "audiomae",
    "avesecho_passt",
    "aves_especies",
    "beats",
    "birdaves_especies",
    "birdnet",
    "hbdet",
    "insect66",
    "insect459",
    "mix2",
    "naturebeats",
    "protoclr",
    "rcl_fs_bsed",
    "vggish",
]

embeddings = {}
with pkg_resources.path(__package__ + ".test_data", "") as audio_dir:
    audio_dir = Path(audio_dir)
config["audio_dir"] = audio_dir
get_paths = make_set_paths_func(**kwargs)


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def embedder_fn(loader, model_name):
    """Return embeddings from a single model using the test loader."""
    embedder = Embedder(model_name, **kwargs)
    return embedder.get_embeddings_from_model(loader.files[0])


def loader_fn():
    """Return a Loader for the test audio directory."""
    loader = Loader(check_if_combination_exists=False, model_name="aves", **kwargs)
    assert loader.files, "No audio files found in test data directory"
    return loader


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------
def test_embedding_generation(model):
    embeddings[model] = get_embeddings(
        model_name=model,
        check_if_primary_combination_exists=False,
        check_if_secondary_combination_exists=False,
        **kwargs,
    )
    assert embeddings[model].files, f"No embeddings generated for {model}"


def test_embedding_dimensions(model):
    assert (
        embeddings[model].metadata_dict["embedding_size"] == embedding_dimensions[model]
    ), f"Embedding dimension mismatch for {model}"


def test_evaluation(model):
    embeds = embeddings[model].embedding_dict()
    paths = get_paths(model)

    try:
        ground_truth = ground_truth_by_model(paths, model, **kwargs)
    except FileNotFoundError:
        ground_truth = None

    assert len(embeds) > 1, (
        f"Too few files to evaluate embeddings with classifier for {model}. "
        "Check that you have the right test data."
    )

    generate_annotations_for_classification_task(paths, **kwargs)

    class_embeds = embeds_array_without_noise(embeds, ground_truth, **kwargs)
    for class_config in settings["class_configs"].values():
        if class_config["bool"]:
            assert len(class_embeds) > 0, (
                f"No embeddings found for classification task ({model}). "
                "Check that annotations.csv is linked correctly. "
                "Remove the classification task from config.yaml if not intended."
            )
            classification_pipeline(paths, class_embeds, **class_config, **kwargs)

    embeds_array = np.concatenate(list(embeds.values()))
    clustering(paths, embeds_array, ground_truth, **kwargs)
