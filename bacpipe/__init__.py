import logging
import yaml
from pathlib import Path
from types import SimpleNamespace
import importlib.resources as pkg_resources
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


# --------------------------------------------------------------------
### EXPOSE API ENDPOINTS ### 
# --------------------------------------------------------------------
from bacpipe.generate_embeddings import Embedder, Loader

from bacpipe.main import (
    get_model_names,
    evaluation_with_settings_already_exists,
    model_specific_embedding_creation,
    model_specific_evaluation,
    cross_model_evaluation,
    visualize_using_dashboard,
)

from bacpipe.embedding_evaluation.label_embeddings import (
    DefaultLabels, 
    get_default_labels,
    get_ground_truth, 
    get_dt_filename
    )




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


supported_models = list(EMBEDDING_DIMENSIONS.keys())
"""list[str]: Supported embedding models available in bacpipe."""

models_needing_checkpoint = NEEDS_CHECKPOINT
"""list[str]: Models that require a checkpoint to be downloaded before use."""


def play(config=config, settings=settings, bool_save_logs=False):
    """
    Play the bacpipe! The pipeline will run using the models specified in
    bacpipe.config.models and generate results in the directory
    bacpipe.settings.results_dir. For more details see the ReadMe file on the
    repository page https://github.com/bioacoustic-ai/bacpipe.

    Parameters
    ----------
    config : dict, optional
        configurations for pipeline execution, by default config
    settings : dict, optional
        settings for pipeline execution, by default settings
    bool_save_logs : bool, optional
        Save logs, config and settings file. This is important if you get a bug,
        sharing this will be very helpful to find the source of
        the problem, by default False


    Raises
    ------
    FileNotFoundError
        If no audio files are found we can't compute any embeddings. So make
        sure the path is correct :)
    """
    settings.model_base_path = ensure_models_exist(Path(settings.model_base_path),
                                                   model_names=config.models)
    overwrite, dashboard = config.overwrite, config.dashboard

    if config.audio_dir == 'bacpipe/tests/test_data':
        with pkg_resources.path(__package__ + ".tests.test_data", "") as audio_dir:
            audio_dir = Path(audio_dir)

        if not audio_dir.exists():
            error = (
                f"\nAudio directory {config.audio_dir} does not exist. Please check the path. "
                "It should be in the format 'C:\\path\\to\\audio' on Windows or "
                "'/path/to/audio' on Linux/Mac. Use single quotes '!"
            )
            logger.exception(error)
            raise FileNotFoundError(error)
        else:
            config.audio_dir = audio_dir

        # ----------------------------------------------------------------
    # Setup logging to file if requested
    # ----------------------------------------------------------------
    if bool_save_logs:
        save_logs()

    config.models = get_model_names(**vars(config), **vars(settings))

    if overwrite or not evaluation_with_settings_already_exists(
        **vars(config), **vars(settings)
    ):

        loader_dict = model_specific_embedding_creation(
            **vars(config), **vars(settings)
        )

        model_specific_evaluation(loader_dict, **vars(config), **vars(settings))

        cross_model_evaluation(**vars(config), **vars(settings))

    if dashboard:
        visualize_using_dashboard(**vars(config), **vars(settings))



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
    remove_from_list = []
    if 'naturebeats' in model_names and not 'beats' in model_names:
        model_names.append('beats')
        remove_from_list = ['beats']
        
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
                
    [model_names.remove(l) for l in remove_from_list]
    return model_base_path.parent / "model_checkpoints"


def save_logs():
    import datetime
    import json
    
    log_dir = Path(settings.main_results_dir) / Path(config.audio_dir).stem / f"logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"bacpipe_{timestamp}.log"

    f_format = logging.Formatter(
        "%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s"
    )
    f_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(f_format)
    f_handler.flush = lambda: f_handler.stream.flush()  # optional, for clarity
    logger.addHandler(f_handler)

    # Save current config + settings snapshot
    settings_dict, config_dict = {}, {}
    for k, v in vars(settings).items():
        if '/' in str(v) or '\\' in str(v):
            settings_dict[k] = Path(v).as_posix()
        else:
            settings_dict[k] = v
    for k, v in vars(config).items():
        if '/' in str(v) or '\\' in str(v):
            config_dict[k] = Path(v).as_posix()
        else:
            config_dict[k] = v
    
    with open(
        log_dir / f"config_{timestamp}.json", "w"
    ) as f:
        json.dump(config_dict, f, indent=2)
    with open(
        log_dir / f"settings_{timestamp}.json", "w"
    ) as f:
        json.dump(settings_dict, f, indent=2)

    logger.info("Saved config, settings, and logs to %s", log_dir)