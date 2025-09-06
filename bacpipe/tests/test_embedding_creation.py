import sys

sys.path.insert(0, ".")
from bacpipe.main import get_embeddings
from bacpipe.generate_embeddings import Loader, Embedder
import numpy as np
from pathlib import Path

import importlib.resources as pkg_resources
import bacpipe

# INITIALIZATION
# Find all models in the pipelines directory
models = [  # "avesecho_passt"]
    mod.stem
    for mod in Path(
        bacpipe.PACKAGE_MAIN / "embedding_generation_pipelines/feature_extractors"
    ).glob("*.py")
]

# Only test models whos checkpoints have been downloaded
models_requiring_checkpoints = [
    "animal2vec_mk",
    "animal2vec_xc",
    "audiomae",
    "aves_especies",
    "avesecho_passt",
    "birdaves_especies",
    "nonbioaves_especies",
    "hbdet",
    "insect66",
    "insect459",
    "mix2",
    "protoclr",
    "vggish",
]
for model in models_requiring_checkpoints:
    if not Path(f"bacpipe/model_checkpoints/{model}").exists():
        models.remove(model)


embedding_dimensions = {
    "animal2vec_xc": 768,
    "animal2vec_mk": 1024,
    "audiomae": 768,
    "aves_especies": 768,
    "biolingual": 512,
    "birdaves_especies": 1024,
    "nonbioaves_especies": 768,
    "birdnet": 1024,
    "avesecho_passt": 768,
    "hbdet": 2048,
    "insect66": 1280,
    "insect459": 1280,
    "mix2": 960,
    "perch_bird": 1280,
    "protoclr": 384,
    "rcl_fs_bsed": 2048,
    "surfperch": 1280,
    "google_whale": 1280,
    "vggish": 128,
}

embeddings = {}

audio_dir = "bacpipe/tests/test_data"

# TESTING


def embedder_fn(loader, model_name):
    embedder = Embedder(model_name, testing=True)
    input = loader.files[0]
    return embedder.get_embeddings_from_model(input)


def loader_fn():
    loader = Loader(
        audio_dir=audio_dir,
        check_if_combination_exists=False,
        model_name="aves",
        testing=True,
    )
    assert loader.files is not None and len(loader.files) > 0
    return loader


# Define the pytest_generate_tests hook to generate test cases
def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        # Generate test cases based on the test_data list
        metafunc.parametrize("model", models)


# Define the actual test function
def test_embedding_generation(model):
    embeddings[model] = get_embeddings(
        model_name=model,
        dim_reduction_model="umap",
        check_if_primary_combination_exists=False,
        check_if_secondary_combination_exists=False,
        audio_dir=audio_dir,
        classifier_threshold=0.3,
        testing=True,
    )
    assert embeddings[model].files is not None and len(embeddings[model].files) > 0


def test_embedding_dimensions(model):
    assert (
        embeddings[model].metadata_dict["embedding_size"] == embedding_dimensions[model]
    )


# test_embedding_generation("avesecho_passt")
# test_embedding_dimensions("avesecho_passt")
# pytest -v --disable-warnings test_embedding_creation.py
