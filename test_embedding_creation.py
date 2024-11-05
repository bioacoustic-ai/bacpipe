from bacpipe.generate_embeddings import generate_embeddings, Loader, Embedder
import numpy as np
from pathlib import Path

def embedder_fn(loader, model_name):
    embedder = Embedder(model_name)
    input = loader.files[0]
    return embedder.get_embeddings_from_model(input)

def loader_fn():
    loader = Loader(check_if_combination_exists=False, 
                    model_name="aves")
    assert loader.files is not None and len(loader.files) > 0
    return loader

# def test_all_models():
models = [mod.stem for mod in Path('bacpipe/pipelines').glob('*.py') 
        if not mod.stem in ['__init__', 'utils', 'umap']]
        
def test_embedding_dimensions():
    pass        

# Define the pytest_generate_tests hook to generate test cases
def pytest_generate_tests(metafunc):
    if 'model' in metafunc.fixturenames:
        # Generate test cases based on the test_data list
        metafunc.parametrize('model', models)

# Define the actual test function
def test_model(model):
    loader = loader_fn()
    result = embedder_fn(loader, model)
    assert isinstance(result, np.ndarray) and result.size > 0

# pytest -v test_embedding_creation.py