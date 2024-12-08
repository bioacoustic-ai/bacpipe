import sklearn.metrics as metrics
import json


def map_index_to_label(index, label2index):
    """
    Map the index to the label
    """
    for label, idx in label2index.items():
        if idx == index:
            return label
    return None


#  compute the evaluation metrics for classification based predictions


#  accuracy per class
def accuracy_per_class(y_true, y_pred, label2index):
    """
    Compute the accuracy per class
    """

    accuracy_per_class_index = {}
    for i in range(len(label2index)):
        accuracy_per_class_index[i] = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            accuracy_per_class_index[y_true[i]] += 1

    accuracy_per_class = {}
    for i in range(len(label2index)):
        label = map_index_to_label(i, label2index)
        accuracy_per_class[label] = accuracy_per_class_index[i] / y_true.count(i)

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


def compute_metrics(y_pred, y_true, probability_scores, label2index):
    """
    Compute the evaluation metrics
    """

    overall_metrics = {
        "macro_accuracy": macro_accuracy(y_true, y_pred),
        "micro_accuracy": micro_accuracy(y_true, y_pred),
        "auc": auc(y_true, probability_scores),
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
    }

    per_class_accuracy = accuracy_per_class(y_true, y_pred, label2index)

    return overall_metrics, per_class_accuracy


# def compute_metrics_per_level_above():
#     '''
#     aggregate the metrics as per classification level above. i.e. if classification on individual classes, then compute metrics per species classes, if classification on species, then compute metrics per taxon classes...
#     '''
#     # TODO
#     return


def build_results_report(
    task_name,
    pretrained_model_name,
    overall_metrics,
    per_class_metrics,
    save_path="bacpipe/bacpipe/evaluation/results/metrics",
):
    """
    Build a results report
    """

    report = {}

    report["Overall Metrics:"] = {
        "Macro Accuracy": overall_metrics["macro_accuracy"],
        "Micro Accuracy": overall_metrics["micro_accuracy"],
        "AUC": overall_metrics["auc"],
        "Macro F1": overall_metrics["macro_f1"],
        "Micro F1": overall_metrics["micro_f1"],
    }

    report["Per Class Metrics:"] = {}
    for label, accuracy in per_class_metrics.items():
        report["Per Class Metrics:"][label] = accuracy

    # save the report as json
    with open(
        save_path
        + "/classsification_results_"
        + task_name
        + "_"
        + pretrained_model_name
        + ".json",
        "w",
    ) as f:
        json.dump(report, f)

    return
