import sys

sys.path.insert(0, ".")
from bacpipe.generate_embeddings import Loader
import bacpipe.embedding_evaluation.label_embeddings as le
from bacpipe.main import embeds_array_without_noise
from bacpipe.embedding_evaluation.classification.classify import classification_pipeline
from bacpipe.embedding_evaluation.clustering.cluster import clustering
import numpy as np
import yaml

with open("bacpipe/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

audio_dir = "bacpipe/tests/audio_test_files/dcase_task5"


def loader_fn(model):
    loader = Loader(
        audio_dir=audio_dir,
        model_name=model,
        testing=True,
    )
    assert loader.files is not None and len(loader.files) > 0
    return loader


def prepare_evaluation():
    loader = loader_fn("avesecho_passt")

    global embeds
    global paths
    global ground_truth

    embeds = loader.embedding_dict()
    get_paths = le.make_set_paths_func(audio_dir, testing=True)
    paths = get_paths("avesecho_passt")
    ground_truth = le.ground_truth_by_model(paths, "avesecho_passt")


prepare_evaluation()


def test_classification():
    le.generate_annotations_for_classification_task(paths)

    class_embeds = embeds_array_without_noise(embeds, ground_truth)
    for class_config in settings["class_configs"].values():
        if class_config["bool"]:
            classification_pipeline(paths, class_embeds, testing=True, **class_config)


def test_clustering():
    embeds_array = np.concatenate(list(embeds.values()))
    clustering(
        paths, embeds_array, ground_truth, clust_configs=settings["clust_configs"]
    )
