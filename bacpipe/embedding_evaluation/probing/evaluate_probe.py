import json
import torch
import torch.nn.functional as F

import sklearn.metrics as metrics
import numpy as np
from pathlib import Path
from .train_probe import probe_dataset_loader

from bacpipe.embedding_evaluation.visualization.visualize_predictions import (
    plot_classification_results,
)



#  accuracy per class
def accuracy_per_class(y_true, y_pred, label2index, items_per_class):
    """
    Accuracy per class

    Parameters
    ----------
    y_true : list
        ground truth
    y_pred : list
        predictions
    label2index : dict
        link labels to ints
    items_per_class : list
        number of items per class

    Returns
    -------
    dict
        classwise accuracy
    """
    acc_per_cls_idx = {class_idx: 0 for class_idx in label2index.values()}

    for pred_class, true_class in zip(y_pred, y_true):
        if pred_class == true_class:
            acc_per_cls_idx[true_class] += 1

    accuracy_per_class = {
        class_label: acc_per_cls_idx[class_idx] / items_per_class[class_label]
        for class_label, class_idx in label2index.items()
    }

    return accuracy_per_class


def macro_accuracy(y_true, y_pred):
    """
    Compute macro accuracy.

    Parameters
    ----------
    y_true : list
        ground truth
    y_pred : list
        predictions

    Returns
    -------
    float
        balance accuracy score
    """
    return metrics.balanced_accuracy_score(y_true, y_pred)


def micro_accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


#  Area under the ROC curve
def auc(y_true, probability_scores):
    """
    Compute the AUC
    """
    if len(np.unique(y_true)) == 2:
        probability_scores = np.array(probability_scores)[:, 1]
    return metrics.roc_auc_score(y_true, probability_scores, multi_class="ovr")


#  macro f1 score
def macro_f1(y_true, y_pred):
    """
    Compute the macro f1 score
    """
    return metrics.f1_score(y_true, y_pred, average="macro")


#  micro f1 score
def micro_f1(y_true, y_pred):
    """
    Compute the micro f1 score
    """
    return metrics.f1_score(y_true, y_pred, average="micro")


def compute_task_metrics(y_pred, y_true, probability_scores, label2index):
    """
    Compute the evaluation metrics
    """

    metrics = dict()

    metrics["overall"] = {
        "macro_accuracy": macro_accuracy(y_true, y_pred),
        "micro_accuracy": micro_accuracy(y_true, y_pred),
        "auc": auc(y_true, probability_scores) if np.unique(y_true).size > 1 else None,
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
    }
    if not metrics["overall"]["auc"]:
        metrics["overall"].pop("auc")
    metrics["items_per_class"] = {
        name: y_true.count(idx) for name, idx in label2index.items()
    }
    metrics["per_class_accuracy"] = accuracy_per_class(
        y_true, y_pred, label2index, metrics["items_per_class"]
    )

    return metrics


def save_probe_results(paths, config, metrics, **kwargs):
    """
    Save a dict with all performance metrics.

    Parameters
    ----------
    paths : SimpleNamespace object
        dict with attributs of paths for loading and saving
    config : string
        type of classification (linear or knn)
    metrics : dict
        performance
    """
    
    for k, v in kwargs.items():
        if isinstance(v, Path):
            kwargs[k] = v.as_posix()
            
    metrics["config"] = kwargs

    save_path = paths.probe_path.joinpath(f"probe_results_{config}.json")

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)



def eval_probe(
    probe, embeds, df, label2index, 
    device="cuda:0", config="linear", 
    paths=None, save_probe=False, **kwargs):
    """
    Perform inference using probe.

    Parameters
    ----------
    probe : object
        trained classification object
    test_dataloader : DataLoader object
        dataset iterator
    device : str, optional
        'cpu' or 'cuda', by default "cuda:0"
    config : str, optional
        type of classification, by default "linear"

    Returns
    -------
    list
        prediction values in ints corresponding to labels
    list
        ground truth values in ints
    np.array
        probabilities for each class and each embedding
    """
    
    test_dataloader = probe_dataset_loader("test", df, embeds, label2index, **kwargs)

    
    device = torch.device(device)
    probe = probe.to(device)

    probe.eval()
    y_pred = []
    y_true = []
    probabilities = []

    for embeddings, y in test_dataloader:
        embeddings, y = embeddings.to(device), y.to(device)

        outputs = probe(embeddings)
        if config == "linear":
            # Use softmax to get probabilities
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            _, predicted = torch.max(outputs, 1)

        elif config == "knn":
            # KNN does not require softmax
            predicted, probs = outputs
            probs = probs.cpu().numpy().tolist()

        y_pred.extend(predicted.cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
        probabilities.extend(probs)

    metrics = compute_task_metrics(y_pred, y_true, probabilities, label2index)
    
    
    if save_probe and not paths is None:
        state_dict = probe.state_dict()
        torch.save(state_dict, paths.probe_path / f"{config}_probe.pt")
        with open(paths.probe_path / "label2index.json", "w") as f:
            json.dump(label2index, f, indent=1)
        save_probe_results(paths, config, metrics, **kwargs)
        plot_classification_results(paths=paths, task_name=config, metrics=metrics)
        
    
    return metrics

        
