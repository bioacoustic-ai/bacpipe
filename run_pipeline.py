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

with pkg_resources.open_text(bacpipe, "config.yaml") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

with pkg_resources.open_text(bacpipe, "settings.yaml") as f:
    settings = yaml.load(f, Loader=yaml.CLoader)

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
