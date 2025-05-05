import yaml
from pathlib import Path

from bacpipe.main import (
    get_model_names,
    model_specific_embedding_creation,
    model_specific_evaluation,
    cross_model_evaluation,
    visualize_using_dashboard,
)


with open("config.yaml", "rb") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

with open("bacpipe/settings.yaml", "rb") as p:
    settings = yaml.load(p, Loader=yaml.CLoader)

overwrite, main_results_dir, audio_dir = (
    settings["overwrite"],
    settings["main_results_dir"],
    Path(config["audio_dir"]).stem,
)

if __name__ == "__main__":

    get_model_names(**config, **settings)

    if overwrite or not (Path(main_results_dir) / audio_dir).exists():

        loader_dict = model_specific_embedding_creation(**config, **settings)

        model_specific_evaluation(loader_dict, **config, **settings)

        cross_model_evaluation(**config, **settings)

    if settings["dashboard"]:

        visualize_using_dashboard(**config, **settings)
