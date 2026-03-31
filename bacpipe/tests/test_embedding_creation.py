import sys
import numpy as np
import yaml
import importlib.resources as pkg_resources
from pathlib import Path

import bacpipe
from bacpipe import EMBEDDING_DIMENSIONS
from bacpipe.core.workflows import run_pipeline_for_single_model

from bacpipe.core.experiment_manager import Loader
from bacpipe.model_pipelines.runner import Embedder

from bacpipe.embedding_evaluation.label_embeddings import (
    make_set_paths_func,
    ground_truth_by_model,
)
from bacpipe.embedding_evaluation.probing.probe import probing_pipeline
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


embeddings = {}
# with pkg_resources.path(__package__ + ".test_data", "") as audio_dir:
#     audio_dir = Path(audio_dir)
with pkg_resources.path(bacpipe.tests, "test_data") as audio_dir:
    audio_dir = Path(audio_dir)
kwargs["audio_dir"] = audio_dir
get_paths = make_set_paths_func(**kwargs)
print(audio_dir)

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def embedder_fn(loader, model_name):
    """Return embeddings from a single model using the test loader."""
    embedder = Embedder(model_name, loader=loader, **kwargs)
    return embedder.get_embeddings_from_model(loader.files[0])


def loader_fn():
    """Return a Loader for the test audio directory."""
    loader = Loader(use_folder_structure=True, check_if_combination_exists=False, model_name="aves", **kwargs)
    assert loader.files, "No audio files found in test data directory"
    return loader


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------
def test_embedding_generation(model, device):
    settings['device'] = device
    embeddings[model] = run_pipeline_for_single_model(
        model_name=model,
        check_if_primary_combination_exists=False,
        check_if_secondary_combination_exists=False,
        **kwargs,
    )
    assert embeddings[model].files, f"No embeddings generated for {model}"


def test_embedding_dimensions(model):
    assert (
        embeddings[model].metadata_dict["embedding_size"] == EMBEDDING_DIMENSIONS[model]
    ), f"Embedding dimension mismatch for {model}"


def test_evaluation(model):
    embeds = embeddings[model].embeddings()
    paths = get_paths(model)

    try:
        ground_truth = ground_truth_by_model(model, paths=paths, **kwargs)
    except FileNotFoundError:
        ground_truth = None

    assert len(embeds) > 1, (
        f"Too few files to evaluate embeddings with classifier for {model}. "
        "Check that you have the right test data."
    )

    
    # generate_annotations_for_probing_task(paths, **kwargs)

    # class_embeds = embeds_array_without_noise(embeds, ground_truth, **kwargs)
    for class_config in settings["probe_configs"].values():
        if class_config["bool"]:
            probing_pipeline(
                ground_truth, embeds, 
                paths, **class_config, **kwargs
                )
            # probing_pipeline(paths, class_embeds, **class_config, **kwargs)

    embeds_array = np.concatenate(list(embeds.values()))
    clustering(paths, embeds_array, ground_truth, **kwargs)
