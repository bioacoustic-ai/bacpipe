import json
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


from .classification_utils.embedding_dataloader import EmbeddingTaskLoader
from .classification_utils.linear_probe import (
    LinearProbe,
    train_linear_probe,
    inference_with_linear_probe,
)
from .classification_utils.evaluation_metrics import compute_task_metrics

import torch

with open("bacpipe/path_settings.yaml", "rb") as f:
    bacpipe_settings = yaml.safe_load(f)


def gen_loader_obj(
    set_name, clean_df, link_embed2wavfile, model_name, loader_object, task_config
):

    loader = EmbeddingTaskLoader(
        partition_dataframe=clean_df,
        embed2wavfile_mapper=link_embed2wavfile,
        set_name=set_name,
        pretrained_model_name=model_name,
        loader_object=loader_object,
        target_labels=task_config["label_type"],
        label2index=task_config["label_to_index"],
    )

    loader_generator = torch.utils.data.DataLoader(
        loader, batch_size=task_config["batch_size"], shuffle=False, drop_last=False
    )
    return loader_generator


def link_embeds_to_wavfiles(model_name, loader_object, data):
    return np.array(
        [
            (f, f.stem.replace(f"_{model_name}", ".wav"))
            for f in loader_object.files
            if f.stem.replace(f"_{model_name}", ".wav") in list(data.wavfilename)
        ]
    )


def define_labels_for_task(data, task_config):
    if task_config["task_name"] == "ID":
        labels = data.hierarchical_labels.unique()
    elif task_config["task_name"] == "species":
        labels = data.species.unique()
    elif task_config["task_name"] == "taxon":
        labels = data.taxon.unique()
    task_config["label_to_index"] = {x: i for i, x in enumerate(labels)}
    task_config["Num_classes"] = len(labels)
    return task_config


def load_and_clean_data(task_name, model_name, loader_object, **kwargs):
    task_config_path = (
        Path(bacpipe_settings["task_config_files"])
        .joinpath(task_name)
        .joinpath("config.json")
    )

    with open(task_config_path, "r") as f:
        task_config = json.load(f)

    # load dataset
    if "testing" in kwargs:
        dataset_path = "bacpipe/evaluation/datasets/embedding_test_files/test_task.csv"
    else:
        dataset_path = task_config["dataset_csv_path"]
    data = pd.read_csv(dataset_path)

    task_config = define_labels_for_task(data, task_config)

    data = data[~data.duplicated()]

    link_embed2wavfile = link_embeds_to_wavfiles(model_name, loader_object, data)

    # ensure that only lines are kept that have a corresponding wav file
    clean_df = data[data.wavfilename.isin(link_embed2wavfile[:, 1])]
    return clean_df, link_embed2wavfile, task_config


def evaluate_on_task(task_name, model_name, loader_object, **kwargs):
    """
    trains a linear probe and predicts on test set for the given task.

    arguments: task_name -> string, what is the task to evaluate on
    pretrained_model -> string, pretrained model name from where the embeddings were extracted. (model that is being evaluated)


    outputs: predictions on test set,
            overall and per class evaluation metrics.

    """
    clean_df, link_embed2wavfile, task_config = load_and_clean_data(
        task_name, model_name, loader_object, **kwargs
    )

    # generate the loaders
    train_gen = gen_loader_obj(
        "train", clean_df, link_embed2wavfile, model_name, loader_object, task_config
    )
    test_gen = gen_loader_obj(
        "test", clean_df, link_embed2wavfile, model_name, loader_object, task_config
    )

    embed_size = loader_object.metadata_dict["embedding_size"]

    lp = LinearProbe(in_dim=embed_size, out_dim=task_config["Num_classes"])
    lp = train_linear_probe(lp, train_gen, task_config)

    y_pred, y_true, probability_scores = inference_with_linear_probe(lp, test_gen)

    metrics = compute_task_metrics(
        y_pred, y_true, probability_scores, task_config["label_to_index"]
    )

    return metrics, task_config
