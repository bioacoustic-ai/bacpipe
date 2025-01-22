from bacpipe.generate_embeddings import generate_embeddings, Loader, Embedder
import numpy as np
from pathlib import Path

models = [
    mod.stem for mod in Path("bacpipe/pipelines/dimensionality_reduction").glob("*.py")
]

audio_dir = Path("bacpipe/evaluation/datasets/audio_test_files")
embed_dir = Path("bacpipe/evaluation/datasets/embedding_test_files")

embeddings = {}


def embedder_fn(loader, model_name):
    embedder = Embedder("avesecho_passt", dim_reduction_model=model_name)
    for idx, file in enumerate(loader.files):
        if idx == 0:
            embeddings = loader.embed_read(idx, file)
        else:
            embeddings = np.concatenate([embeddings, loader.embed_read(idx, file)])
    return embedder.get_embeddings_from_model(embeddings)


def loader_fn():
    loader = Loader(
        audio_dir=audio_dir,
        dim_reduction_model="umap",
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
    loader = loader_fn()
    result = embedder_fn(loader, model)
    embeddings[model] = result


test_embedding_generation("umap")
