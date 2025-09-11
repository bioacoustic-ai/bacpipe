import logging
import zipfile
import yaml
from pathlib import Path
from types import SimpleNamespace
import importlib.resources as pkg_resources
from huggingface_hub import hf_hub_download

def ensure_std_models(model_base_path, repo_id="vinikay/bacpipe_models"):
    """
    Ensure that the model checkpoints for birdnetv2.4 and perchv1 are
    available locally. Downloads from Hugging Face Hub if missing.

    Parameters
    ----------
    model_base_path : Path
        Local base directory where the checkpoints should be stored.
    repo_id : str, optional
        Hugging Face Hub repo ID, by default "vinikay/bacpipe_models"
    """
    model_base_path = Path(model_base_path)
    model_base_path.parent.mkdir(exist_ok=True, parents=True)

    if model_base_path.exists():
        return model_base_path.parent / "model_checkpoints"

    print(
        "This seems to be the first call. Hi there! "
        "Downloading model checkpoints from Hugging Face Hub for "
        "BirdNet. Because let's face it, everyone needs to at least have BirdNet."
        )
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename="model_checkpoints.zip",
        repo_type="dataset",
    )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(model_base_path.parent)

    return model_base_path.parent / "model_checkpoints"


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
# Expose core API functions
# --------------------------------------------------------------------
from bacpipe.tests.test_embedding_creation import embedding_dimensions, needs_checkpoint
from bacpipe.generate_embeddings import Embedder

supported_models = list(embedding_dimensions.keys())
models_needing_checkpoint = needs_checkpoint


from bacpipe.main import (
    get_model_names,
    evaluation_with_settings_already_exists,
    model_specific_embedding_creation,
    model_specific_evaluation,
    cross_model_evaluation,
    visualize_using_dashboard,
)


def play(config=config, settings=settings, save_logs=False):
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
    save_logs : bool, optional
        Save logs, config and settings file. This is important if you get a bug,
        sharing this will be very helpful to find the source of
        the problem, by default False


    Raises
    ------
    FileNotFoundError
        If no audio files are found we can't compute any embeddings. So make
        sure the path is correct :)
    """
    settings.model_base_path = ensure_std_models(Path(settings.model_base_path))
    overwrite, dashboard = config.overwrite, config.dashboard

    with pkg_resources.path(__package__ + ".tests.test_data", "") as audio_dir:
        audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Audio directory {config.audio_dir} does not exist. Please check the path. "
            "It should be in the format 'C:\\path\\to\\audio' on Windows or "
            "'/path/to/audio' on Linux/Mac. Use single quotes '!"
        )
    else:
        config.audio_dir = audio_dir

        # ----------------------------------------------------------------
    # Setup logging to file if requested
    # ----------------------------------------------------------------
    if save_logs:
        import datetime
        import json

        Path(settings.main_results_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = Path(settings.main_results_dir) / f"bacpipe_{timestamp}.log"

        f_format = logging.Formatter(
            "%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s"
        )
        f_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(f_format)
        f_handler.flush = lambda: f_handler.stream.flush()  # optional, for clarity
        logger.addHandler(f_handler)

        # Save current config + settings snapshot
        with open(
            Path(settings.main_results_dir) / f"config_{timestamp}.json", "w"
        ) as f:
            json.dump(vars(config), f, indent=2)
        with open(
            Path(settings.main_results_dir) / f"settings_{timestamp}.json", "w"
        ) as f:
            json.dump(vars(settings), f, indent=2)

        logger.info("Saved config, settings, and logs to %s", settings.main_results_dir)

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
