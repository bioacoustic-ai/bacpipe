import sklearn.metrics as metrics
import json
from pathlib import Path
import yaml
import numpy as np

with open("bacpipe/settings.yaml", "rb") as f:
    bacpipe_settings = yaml.load(f, Loader=yaml.CLoader)
#  compute the evaluation metrics for classification based predictions


#  accuracy per class
def accuracy_per_class(y_true, y_pred, label2index, items_per_class):
    """
    Compute the accuracy per class
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


#  macro accuracy
def macro_accuracy(y_true, y_pred):
    """
    Compute the macro accuracy
    """
    return metrics.balanced_accuracy_score(y_true, y_pred)


#  micro accuracy
def micro_accuracy(y_true, y_pred):
    """
    Compute the micro accuracy
    """
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
        "auc": auc(y_true, probability_scores),
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
    }
    metrics["items_per_class"] = {
        name: y_true.count(idx) for name, idx in label2index.items()
    }
    metrics["per_class_accuracy"] = accuracy_per_class(
        y_true, y_pred, label2index, metrics["items_per_class"]
    )

    return metrics


# def compute_metrics_per_level_above():
#     '''
#     aggregate the metrics as per classification level above. i.e. if classification on individual classes, then compute metrics per species classes, if classification on species, then compute metrics per taxon classes...
#     '''
#     # TODO
#     return


def build_results_report(task_name, model_name, metrics, task_config):
    """
    Build a results report
    """

    report = dict()

    report["Overall Metrics:"] = {
        "Macro Accuracy": float(metrics["overall"]["macro_accuracy"]),
        "Micro Accuracy": float(metrics["overall"]["micro_accuracy"]),
        "AUC": float(metrics["overall"]["auc"]),
        "Macro F1": float(metrics["overall"]["macro_f1"]),
        "Micro F1": float(metrics["overall"]["micro_f1"]),
    }

    report["Per Class Metrics:"] = {
        label: acc for label, acc in metrics["per_class_accuracy"].items()
    }

    report["Items per Class:"] = {
        label: items for label, items in metrics["items_per_class"].items()
    }

    report["Task Configuration:"] = task_config

    save_dir = Path(bacpipe_settings["task_results_dir"]).joinpath("metrics")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir.joinpath(f"class_results_{task_name}_{model_name}.yml")
    # save the report as json
    with open(save_path, "w") as f:
        yaml.safe_dump(report, f)
