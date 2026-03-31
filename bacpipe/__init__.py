import logging
import yaml
from types import SimpleNamespace
import importlib.resources as pkg_resources
<<<<<<< HEAD
from huggingface_hub import hf_hub_download
import tarfile

TF_MODELS = [
    'birdnet', 
    'perch_v2',
    'perch_bird', 
    'google_whale', 
    'surfperch', 
    'vggish',
    'hbdet', 
]

EMBEDDING_DIMENSIONS = {
    "audiomae": 768,
    "audioprotopnet": 1024,
    "avesecho_passt": 768,
    "aves_especies": 768,
    "bat": 64,
    "batdetect2_mean_clip": 32,
    "batdetect2_mean_detections": 32,
    "beats": 768,
    "birdaves_especies": 1024,
    "biolingual": 512,
    "birdnet": 1024,
    "birdmae": 1280,
    "convnext_birdset": 1024,
    "hbdet": 2048,
    "insect66": 1280,
    "insect459": 1280,
    "mix2": 960,
    "naturebeats": 768,
    "perch_bird": 1280,
    "perch_v2": 1536,
    "protoclr": 384,
    "rcl_fs_bsed": 2048,
    "surfperch": 1280,
    "google_whale": 1280,
    "vggish": 128,
}

NEEDS_CHECKPOINT = [
    "audiomae",
    "avesecho_passt",
    "aves_especies",
    "bat",
    "beats",
    "birdaves_especies",
    "birdnet",
    "hbdet",
    "insect66",
    "insect459",
    "mix2",
    "naturebeats",
    "protoclr",
    "rcl_fs_bsed"
]

def ensure_models_exist(model_base_path, model_names, repo_id="vskode/bacpipe_models"):
    """
    Ensure that the model checkpoints for the selected models are
    available locally. Downloads from Hugging Face Hub if missing.

    Parameters
    ----------
    model_base_path : Path
        Local base directory where the checkpoints should be stored.
    model_names : list
        list of models to run
    repo_id : str, optional
        Hugging Face Hub repo ID, by default "vinikay/bacpipe_models"
    """
    model_base_path = Path(model_base_path)
    model_base_path.parent.mkdir(exist_ok=True, parents=True)
    
    logger.info(
        "Checking if the selected models require a checkpoint, and if so, "
        "if the checkpoint already exists.\n"
    )
    
    for model_name in model_names:
        if model_name in NEEDS_CHECKPOINT:
            if ((model_base_path / model_name).exists()
                and len(list((model_base_path / model_name).iterdir())) > 0):
                logger.info(f"{model_name} checkpoint exists.\n")    
                continue
            else:   
                if model_name == 'birdnet':
                    import tensorflow as tf
                    if tf.__version__ == '2.15.1':
                        hf_url = f"{model_name}/{model_name}_tf215.tar.xz"
                    else:
                        hf_url = f"{model_name}/{model_name}.tar.xz"
                else:
                    hf_url = f"{model_name}/{model_name}.tar.xz"
                    
                logger.info(
                    f"{model_name} checkpoint does not exists. "
                    "Downloading the model from "
                    f"https://huggingface.co/datasets/{repo_id}/blob/main/{hf_url}\n"
                    )    
                hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_url,
                    local_dir=model_base_path,
                    repo_type="dataset",
                )
                tar = tarfile.open(model_base_path / hf_url)
                tar.extractall(path=model_base_path)
                tar.close()

    return model_base_path.parent / "model_checkpoints"
=======
>>>>>>> origin/v1.3.0



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
    get_ground_truth, 
    get_dt_filename,
    make_set_paths_func,
    create_default_labels,
    ground_truth_by_model
    )

from bacpipe.core.constants import (
    supported_models, 
    models_needing_checkpoint,
    TF_MODELS,
    EMBEDDING_DIMENSIONS,
    NEEDS_CHECKPOINT
    )

__all__ = [
    play,
    Loader,
    Embedder,
    generate_embeddings,
    run_pipeline_for_single_model,
    ensure_models_exist,
    make_set_paths_func,
    create_default_labels,
    ground_truth_by_model,
    get_model_names,
    get_audio_files,
    evaluation_with_settings_already_exists,
    run_pipeline_for_models,
    model_specific_evaluation,
    cross_model_evaluation,
    visualize_using_dashboard,
    DefaultLabels, 
    get_default_labels,
    get_ground_truth, 
    get_dt_filename,
    supported_models, 
    models_needing_checkpoint,
    TF_MODELS,
    EMBEDDING_DIMENSIONS,
    NEEDS_CHECKPOINT
]