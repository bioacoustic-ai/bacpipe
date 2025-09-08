import logging

# Unzip models_example.zip to initiate models dir structure
import zipfile
from pathlib import Path


# Determine package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
PACKAGE_MAIN = Path(__file__).parent

# Unzip models_example.zip if needed
models_dir = PACKAGE_MAIN / "model_checkpoints"
zip_file = PACKAGE_MAIN / "model_checkpoints.zip"

if not models_dir.exists() and zip_file.exists():
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(PACKAGE_MAIN)

# Initialize Logger
logger = logging.getLogger("bacpipe")
c_handler = logging.StreamHandler()
c_format = logging.Formatter("%(name)s::%(levelname)s:%(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.WARNING)

## Expose common functions

import yaml
from pathlib import Path
import importlib.resources as pkg_resources

import bacpipe
from bacpipe.main import (
    get_model_names,
    evaluation_with_settings_already_exists,
    model_specific_embedding_creation,
    model_specific_evaluation,
    cross_model_evaluation,
    visualize_using_dashboard,
)

with open(bacpipe.PACKAGE_ROOT / "config.yaml") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

with pkg_resources.open_text(bacpipe, "settings.yaml") as f:
    settings = yaml.load(f, Loader=yaml.CLoader)


def run(config, settings):
    overwrite, dashboard = config["overwrite"], config["dashboard"]

    if not Path(config["audio_dir"]).exists():
        raise FileNotFoundError(
            f"Audio directory {config['audio_dir']} does not exist. Please check the path. "
            "It should be in the format 'C:\\path\\to\\audio' on Windows or "
            "'/path/to/audio' on Linux/Mac. But be sure to use single quotes '!"
        )

    get_model_names(**config, **settings)

    if overwrite or not evaluation_with_settings_already_exists(**config, **settings):

        loader_dict = model_specific_embedding_creation(**config, **settings)

        model_specific_evaluation(loader_dict, **config, **settings)

        cross_model_evaluation(**config, **settings)

    if dashboard:

        visualize_using_dashboard(**config, **settings)
