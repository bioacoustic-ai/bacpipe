import logging
import json

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

logger = logging.getLogger(__name__)

from .train_classifier import (
    LinearClassifier,
    train_linear_classifier,
    inference,
    KNN,
    train_knn_classifier,
)
from .evaluate_classifier import compute_task_metrics, evaluate_classification
from bacpipe.embedding_evaluation.visualization.visualize import (
    plot_classification_results,
)


class ClassificationLoader(Dataset):
    def __init__(self, class_df, embeds, label2index, set_name=None, **kwargs):
        """
        Class to initialize and iterate through classification dataset.

        Parameters
        ----------
        class_df : pd.DataFrame
            classification dataframe
        embeds : np.array
            embeddings
        label2index : dict
            linking labels to integers
        set_name : string, optional
            train, test or val set, by default None
        """
        if set_name is not None:
            self.dataset = class_df[class_df.predefined_set == set_name]
        else:
            self.dataset = class_df

        print(
            f"Found {len(self.dataset)} samples in the {set_name} set with "
            f"{len(self.dataset.label.unique())} unique labels."
        )
        self.embeds = embeds

        self.label2index = label2index

        self.dataset = self.dataset.sample(frac=1, random_state=42)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Iterate through dataset.

        Parameters
        ----------
        idx : int
            index of training step

        Returns
        -------
        tuple
            (embedding, true label)
        """
        X = self.embeds[self.dataset.index[idx]]
        X = X.reshape(1, -1)

        if X.shape[0] > 1:
            X = np.mean(X, axis=0)
        else:
            X = X.flatten()
        y = self.label2index[self.dataset.label.values[idx]]

        return X, y


def gen_loader_obj(
    set_name, clean_df, embeds, label2index, batch_size=64, shuffle=False, **kwargs
):
    """
    Create dataset loader object for classification.

    Parameters
    ----------
    set_name : string
        train, test of val set
    clean_df : pd.DataFrame
        classification dataframe
    embeds : np.array
        embeddings
    label2index : dict
        link labels to ints
    batch_size : int, optional
        number of embeddings per batch, by default 64
    shuffle : bool, optional
        shuffle or not, by default False

    Returns
    -------
    DataLoader obj
        dataset loader object to iterate over during training
    """
    loader = ClassificationLoader(
        class_df=clean_df,
        set_name=set_name,
        embeds=embeds,
        label2index=label2index,
        **kwargs,
    )

    loader_generator = DataLoader(
        loader, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
    return loader_generator


def classify(paths, dataset_csv_path, embeds, config="linear", **kwargs):
    """
    Classification pipeline. First the classification dataframe is loaded,
    then a dict is created to link labels to ints, then the dataset loaders
    are created to iterate over. Next depending of the specified config
    a linear or KNN classification is performed. Finally the classifiers are
    used for inference and based on that performance metrics are created.

    Parameters
    ----------
    paths : SimpleNamespace dict
        dictionary object containing paths for loading and saving
    dataset_csv_path : string
        name of classification dataframe as secified in the settings.yaml file
    embeds : np.array
        the embeddings
    config : str, optional
        type of classification, by default 'linear'

    Returns
    -------
    dict
        performance dictionary
    """
    df = pd.read_csv(paths.labels_path.joinpath(dataset_csv_path))

    label2index = {label: i for i, label in enumerate(df.label.unique())}

    # generate the loaders
    train_gen = gen_loader_obj("train", df, embeds, label2index, **kwargs)
    test_gen = gen_loader_obj("test", df, embeds, label2index, **kwargs)

    embed_size = embeds[0].shape[-1]

    if config == "linear":
        clfier = LinearClassifier(in_dim=embed_size, out_dim=len(df.label.unique()))
        clfier = train_linear_classifier(clfier, train_gen, **kwargs)

        state_dict = clfier.state_dict()
        torch.save(state_dict, paths.class_path / f"{config}_classifier.pt")
        with open(paths.class_path / "label2index.json", "w") as f:
            json.dump(label2index, f, indent=1)

    elif config == "knn":
        clfier = KNN(**kwargs)
        clfier = train_knn_classifier(clfier, train_gen, **kwargs)

    y_pred, y_true, probs = inference(clfier, test_gen, config=config, **kwargs)

    metrics = compute_task_metrics(y_pred, y_true, probs, label2index)

    return metrics


def classification_pipeline(
    paths, embeds, name, dataset_csv_path, overwrite=False, **kwargs
):
    """
    Classification pipeline consisting of building the classifier,
    evaluating it and saving metrics and plots of performance.

    Parameters
    ----------
    paths : SimpleNamespace object
        dict with attributes corresponding to paths for loading and saving
    embeds : np.array
        embeddings
    name : string
        Type of classification
    dataset_csv_path : string
        name of classification dataframe as specified in settings.yaml
    overwrite : bool
        overwrite existing classification?, defaults to False
    """
    if (
        overwrite
        or not paths.class_path.joinpath(f"class_results_{name}.json").exists()
    ):
        metrics = classify(paths, dataset_csv_path, embeds, config=name, **kwargs)

        evaluate_classification(paths, name, metrics, **kwargs)
        plot_classification_results(paths=paths, task_name=name, metrics=metrics)
    else:
        logger.info(
            f"Classification file class_results_{name}.json already exists and"
            " so is not computed. If you want to overwrite existing results, "
            "set overwrite to True in settings.yaml."
        )
