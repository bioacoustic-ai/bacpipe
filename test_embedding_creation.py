from bacpipe.generate_embeddings import generate_embeddings, Loader, Embedder
import numpy as np
from pathlib import Path

def embedder_fn(loader, model_name):
    embedder = Embedder(model_name)
    input = loader.load(loader.files[0])
    embeds = embedder.get_embeddings_from_model(input)
    assert isinstance(embeds, np.ndarray) and embeds.size > 0

def loader_fn():
    loader = Loader(check_if_combination_exists=False, 
                    model_name="aves")
    assert loader.files is not None and len(loader.files) > 0
    return loader

def call_models(model):
    loader = loader_fn()
    print(f"Testing {model}")
    embedder_fn(loader, model)

def test_all_models():
    models = [mod.stem for mod in Path('bacpipe/pipelines').glob('*.py') 
            if not mod.stem in ['__init__', 'utils', 'umap']]

    for model in models:
        call_models(model)
        
def test_embedding_dimensions():
    pass        

test_all_models()
# # vggish
# def test_vggish():
#     call_models("vggish")

# # hbdet
# def test_hbdet():
#     call_models("hbdet")

# # animal2vec
# def test_animal2vec():
#     call_models("animal2vec")

# # perch
# def test_perch():
#     call_models("perch")

# # aves
# def test_aves():
#     call_models("aves")

# # birdaves
# def test_birdaves():
#     call_models("birdaves")

# # birdnet
# def test_birdnet():
#     call_models("birdnet")

# # biolingual
# def test_biolingual():
#     call_models("biolingual")

# # insect66
# def test_insect66():
#     call_models("insect66")

# test_insect66()