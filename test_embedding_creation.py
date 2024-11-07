from bacpipe.generate_embeddings import generate_embeddings, Loader, Embedder
import numpy as np
from pathlib import Path

# Find all models in the pipelines directory
models = [
    mod.stem
    for mod in Path("bacpipe/pipelines").glob("*.py")
    if not mod.stem in ["__init__", "utils", "umap"]
]

embedding_dimensions = {
    "animal2vec_xc": 768,
    "animal2vec_mk": 1024,
    "audiomae": 768,
    "aves": 768,
    "biolingual": 512,
    "birdaves": 768,
    "birdnet": 1024,
    "echopasst": 768,
    "hbdet": 2048,
    "insect66": 1280,
    "mix2": 960,
    "perch": 1280,
    "protoclr": 384,
    "rcl_fs_bsed": 2048,
    "surfperch": 1280,
    "whaleperch": 1280,
    "vggish": 128,
}

embeddings = {}


def embedder_fn(loader, model_name):
    embedder = Embedder(model_name)
    input = loader.files[0]
    return embedder.get_embeddings_from_model(input)


def loader_fn():
    loader = Loader(check_if_combination_exists=False, model_name="aves", testing=True)
    assert loader.files is not None and len(loader.files) > 0
    return loader


# Define the pytest_generate_tests hook to generate test cases
def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        # Generate test cases based on the test_data list
        metafunc.parametrize("model", models)


# Define the actual test function
def test_embedding_generation(model):
    loader = loader_fn()
    result = embedder_fn(loader, model)
    embeddings[model] = result


def test_embedding_dimensions(model):
    assert embeddings[model].shape[-1] == embedding_dimensions[model]


# test_model('echopasst')
# pytest -v --disable-warnings test_embedding_creation.py
