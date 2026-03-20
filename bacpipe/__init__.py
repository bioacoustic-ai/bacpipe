import logging
import yaml
from types import SimpleNamespace
import importlib.resources as pkg_resources

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logger = logging.getLogger("bacpipe")
if not logger.handlers:
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(c_handler)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------
# Load config & settings
# --------------------------------------------------------------------
with pkg_resources.open_text(__package__, "config.yaml") as f:
    _config_dict = yaml.load(f, Loader=yaml.CLoader)

with pkg_resources.open_text(__package__, "settings.yaml") as f:
    _settings_dict = yaml.load(f, Loader=yaml.CLoader)

# Expose as mutable namespaces
config = SimpleNamespace(**_config_dict)
settings = SimpleNamespace(**_settings_dict)



# --------------------------------------------------------------------
### EXPOSE API ENDPOINTS ### 
# --------------------------------------------------------------------

from bacpipe.core.experiment_manager import Loader
get_audio_files = Loader.get_audio_files

from bacpipe.model_pipelines.runner import Embedder

from bacpipe.core.workflows import (
    play,
    generate_embeddings,
    run_pipeline_for_single_model,
    ensure_models_exist,
    get_model_names,
    evaluation_with_settings_already_exists,
    run_pipeline_for_models,
    model_specific_evaluation,
    cross_model_evaluation,
    visualize_using_dashboard,
)

from bacpipe.embedding_evaluation.label_embeddings import (
    DefaultLabels, 
    get_default_labels,
    get_dt_filename,
    make_set_paths_func,
    create_default_labels,
    ground_truth_by_model
    )

from bacpipe.embedding_evaluation.probing.probe import probing_pipeline
from bacpipe.embedding_evaluation.probing.inference_probe import (
    run_probe_inference,
    prepare_probe_inference
)

from bacpipe.core.constants import (
    supported_models, 
    models_needing_checkpoint,
    TF_MODELS,
    EMBEDDING_DIMENSIONS,
    NEEDS_CHECKPOINT
    )

__all__ = [
    ## pipelines
    play,
    run_pipeline_for_single_model,
    run_pipeline_for_models,
    generate_embeddings,
    
    ## loader and embedder class for 
    ## loading files and computing embeddings
    Loader,
    Embedder,
    
    ## return audio files in specified dir
    get_audio_files,
    
    ## automatic creation of labels and ground truth
    DefaultLabels, 
    create_default_labels,
    ground_truth_by_model,
    get_default_labels,
    get_dt_filename,
    
    ## probing functions
    probing_pipeline,
    run_probe_inference,
    prepare_probe_inference,
    
    ## evaluation pipelines
    model_specific_evaluation,
    cross_model_evaluation,
    
    ## experiment managing functions
    ensure_models_exist,
    evaluation_with_settings_already_exists,
    get_model_names,
    make_set_paths_func,
    
    ## visualization function to start dashboard
    visualize_using_dashboard,
    
    ## constants
    supported_models, 
    models_needing_checkpoint,
    TF_MODELS,
    EMBEDDING_DIMENSIONS,
    NEEDS_CHECKPOINT
]