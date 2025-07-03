import yaml
from pathlib import Path

from bacpipe.main import (
    get_model_names,
    evaluation_with_settings_already_exists,
    model_specific_embedding_creation,
    model_specific_evaluation,
    cross_model_evaluation,
    visualize_using_dashboard,
)


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

with open("bacpipe/settings.yaml", "r", encoding="utf-8") as p:
    settings = yaml.load(p, Loader=yaml.CLoader)

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

    import numpy as np

    a = []
    for i, f in enumerate(loader_dict["birdnet"].files):
        a.append(np.load(f))
    embeds = np.vstack(a)

    from sklearn.neighbors import LocalOutlierFactor

    clf = LocalOutlierFactor(n_neighbors=5)
    points = clf.fit_predict(embeds)
    in_or_out = clf.negative_outlier_factor_

    model_specific_evaluation(loader_dict, **config, **settings)

    cross_model_evaluation(**config, **settings)

if dashboard:

    visualize_using_dashboard(**config, **settings)
