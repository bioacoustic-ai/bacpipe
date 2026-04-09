import sys
import numpy as np
import yaml
import importlib.resources as pkg_resources
from pathlib import Path

import bacpipe
from bacpipe import (
    make_set_paths_func,
    ground_truth_by_model,
    probing_pipeline,
    clustering_pipeline,
    EMBEDDING_DIMENSIONS,
    run_pipeline_for_single_model,
    Loader,
    Embedder
)


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
    bacpipe.ensure_models_exist(bacpipe.settings.model_base_path, model_names=[model])
    embeddings[model] = run_pipeline_for_single_model(
        model_name=model,
        check_if_already_processed=False,
        check_if_already_dim_reduced=False,
        **kwargs,
    )
    assert embeddings[model].files, f"No embeddings generated for {model}"


def test_embedding_dimensions(model):
    assert (
        embeddings[model].metadata_dict["embedding_size"] == EMBEDDING_DIMENSIONS[model]
    ), f"Embedding dimension mismatch for {model}"


def test_evaluation(model):
    embeds = embeddings[model].embeddings(return_type='array')
    paths = get_paths(model)

    try:
        ground_truth = ground_truth_by_model(model, single_label=False, **kwargs)
    except FileNotFoundError:
        ground_truth = None

    assert len(embeds) > 1, (
        f"Too few files to evaluate embeddings with classifier for {model}. "
        "Check that you have the right test data."
    )

    for class_config in settings["probe_configs"].values():
        if class_config["bool"]:
            probing_pipeline(
                model,
                ground_truth, embeds, 
                paths, single_label=False,
                **class_config,
                **kwargs
                )

    clustering_pipeline(model, ground_truth, embeds, paths, **kwargs)
