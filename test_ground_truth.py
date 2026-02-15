import bacpipe
get_paths = bacpipe.core.workflows.make_set_paths_func(**{**vars(bacpipe.settings), **vars(bacpipe.config)})

# bacpipe.core.workflows.ground_truth_by_model(
#     get_paths('birdnet'), 
#     'birdnet', 
#     **{**vars(bacpipe.settings), **vars(bacpipe.config)}
#     )
bacpipe.embedding_evaluation.label_embeddings.ground_truth_api_call(
    'birdnet', 
    min_annotation_length=0.05,
    single_label=False, 
    **{**vars(bacpipe.settings), **vars(bacpipe.config)}
    )