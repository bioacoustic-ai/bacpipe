import bacpipe
from bacpipe import (
    make_set_paths_func,
    ground_truth_by_model,
    probing_pipeline,
    clustering_pipeline,
    EMBEDDING_DIMENSIONS,
    run_pipeline_for_single_model,
)


embeddings = {}

# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------
# Remove all the module-level config loading and replace with:
embeddings = {}

def test_embedding_generation(
    model, 
    device,
    check_if_already_processed,
    check_if_already_dim_reduced,
    only_embed_annotations,
    kwargs
    ):
    bacpipe.ensure_models_exist(bacpipe.settings.model_base_path, model_names=[model])
    
    embeddings[model] = run_pipeline_for_single_model(
        model_name=model, 
        **kwargs
        )
    
    assert embeddings[model].files, f"No embeddings generated for {model}"

def test_embedding_dimensions(model, kwargs):
    assert (
        embeddings[model].metadata_dict["embedding_size"] == EMBEDDING_DIMENSIONS[model]
    ), f"Embedding dimension mismatch for {model}"

def test_evaluation(model, overwrite, device, only_embed_annotations, kwargs):
    embeds = embeddings[model].embeddings(return_type='array')
    get_paths = make_set_paths_func(**kwargs)
    paths = get_paths(model)
    if model in bacpipe.TF_MODELS:
        kwargs['device'] = 'cpu'
    try:
        ground_truth = ground_truth_by_model(
            model, 
            single_label=False, 
            **kwargs
            )
    except FileNotFoundError:
        ground_truth = None
    assert len(embeds) > 1
    for class_config in kwargs.get("probe_configs", {}).values():
        if class_config["bool"]:
            probing_pipeline(
                model, 
                ground_truth, 
                embeds, 
                paths, 
                single_label=False, 
                **class_config, 
                **kwargs)
    clustering_pipeline(
        model, 
        ground_truth, 
        embeds, 
        paths, 
        **kwargs
        )

def test_benchmarking(
    model,
    device,
    only_embed_annotations,
    kwargs
    ):
        
    results = bacpipe.benchmark(
        # 'birdnet',
        model,
        kwargs['audio_dir'],
        check_if_already_processed=True,
        annotations_file='annotations.csv'
    )
    assert isinstance(results, dict)