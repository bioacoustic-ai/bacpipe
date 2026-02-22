import json

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import bacpipe.embedding_evaluation.label_embeddings as le
import matplotlib
import seaborn as sns

import logging

logger = logging.getLogger(__name__)



matplotlib.rcParams.update(
    {
        "figure.dpi": 600,  # High-resolution figures
        "savefig.dpi": 600,  # Exported plot DPI
        "font.size": 12,  # Better font readability
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def visualise_results_across_models(plot_path, task_name, model_list):
    """
    Create visualizations to compare models by specified tasks.

    Parameters
    ----------
    path_func : function
        return the paths when given a model name
    plot_path : pathlib.Path object
        path to overview plots
    task_name : str
        name of task
    model_list : list
        list of models
    """
    metrics = load_results(le.get_paths, task_name, model_list)
    with open(plot_path.joinpath(f"{task_name}_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if task_name == "classification":
        iterate_through_subtasks(
            plot_per_class_metrics, plot_path, task_name, model_list, metrics
        )

        iterate_through_subtasks(
            plot_overview_metrics, plot_path, task_name, model_list, metrics
        )
    else:
        plot_overview_metrics(plot_path, task_name, model_list, metrics, path_func=le.get_paths)


def iterate_through_subtasks(plot_func, plot_path, task_name, model_list, metrics):
    """
    For classification multiple subtasks exist (linear and knn). Iterate
    over each of the subtasks and call the plotting functions to create
    the visualizations.

    Parameters
    ----------
    plot_func : function
        returns model specific paths when model name is passed
    plot_path : pathlib.Path object
        path to store overview plots
    task_name : str
        name of task
    model_list : list
        list of models
    metrics : dict
        performance dictionary
    """
    subtasks = np.unique([s.split("(")[-1][:-1] for s in list(metrics.keys())])
    for subtask in subtasks:
        sub_task_metrics = {
            k.split("(")[0]: v for k, v in metrics.items() if subtask in k
        }
        plot_func(plot_path, f"{subtask} {task_name}", model_list, sub_task_metrics)


def clustering_overview(
    path_func, label_by, no_noise, model_list, label_column, **kwargs
):
    """
    Create overview plots for clustering metrics.

    Parameters
    ----------
    path_func : function
        function to return the paths when model name is given
    label_by : str
        key of default_labels dict
    no_noise : bool
        whether to plot the metrics with or without noise
    model_list : list
        list of models
    label_column : str
        label as defined in the annotations.csv file
    kwargs : dict
        additional arguments for plotting

    Returns
    -------
    plt.plot object
        figure handle
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.25, right=0.9)
    flat_metrics = dict()
    for model_name in model_list:
        with open(path_func(model_name).clust_path / "clust_results.json", "r") as f:
            metrics = json.load(f)
        if no_noise:
            no_noise = "_no_noise"
        else:
            no_noise = ""
        flat_metrics[model_name] = dict()
        if label_by == label_column:
            flat_metrics[model_name][label_column] = metrics["ARI"][
                f"{label_column}{no_noise}-kmeans"
            ]
        elif not label_by == "kmeans":
            flat_metrics[model_name]["kmeans"] = metrics["ARI"][
                f"kmeans{no_noise}-{label_by}"
            ]
        if not label_by == label_column and label_column in [
            k.split("-")[0] for k in metrics["ARI"].keys()
        ]:
            flat_metrics[model_name][label_column] = metrics["ARI"][
                f"{label_column}{no_noise}-{label_by}"
            ]

    return generate_bar_plot(flat_metrics, fig, ax, **kwargs)


def plot_clusterings(
    path_func, model_name, label_by, no_noise, fig=None, ax=None, **kwargs
):
    """
    Plot the clustering metrics for a given model and label type.

    Parameters
    ----------
    path_func : function
        function to return the paths when model name is given
    model_name : str
        name of model
    label_by : str
        key of default_labels dict
    no_noise : bool
        whether to plot the metrics with or without noise
    fig : plt.plot object, optional
        figure handle, by default None
    ax : plt.plot object, optional
        axes handle, by default None

    Returns
    -------
    plt.plot object
        figure handle
    """
    if no_noise:
        no_noise = "_no_noise"
    else:
        no_noise = ""

    clust_path = path_func(model_name).clust_path / "clust_results.json"
    if not clust_path.exists():
        error = (
            f"\nThe clustering file {clust_path} does not exist. Perhaps it was not "
            "created yet. To avoid getting this error set `overwrite=True`."
        )
        logger.exception(error)
        raise AssertionError(error)

    with open(clust_path, "r") as f:
        metrics = json.load(f)

    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.subplots_adjust(left=0.4, bottom=0.25)

    keys = [
        l
        for l in np.unique([k.split("-")[0] for k in metrics["AMI"].keys()])
        if not "no_noise" in l
    ]
    flat_metrics = {k: dict() for k in keys}
    if label_by == "ground_truth":
        return None
    for compared_to in keys:
        try:
            flat_metrics[compared_to]["AMI"] = metrics["AMI"][
                f"{compared_to+no_noise}-{label_by}"
            ]
            flat_metrics[compared_to]["ARI"] = metrics["ARI"][
                f"{compared_to+no_noise}-{label_by}"
            ]
        except KeyError:
            flat_metrics[compared_to]["AMI"] = 0
            flat_metrics[compared_to]["ARI"] = 0

    return generate_bar_plot(flat_metrics, fig, ax, **kwargs)


def generate_bar_plot(
    metrics, fig, ax, x_label="Metric value", no_legend=False, **kwargs
):
    bar_height = 1 / (len(list(metrics.values())[0].keys()) + 1)
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(list(metrics.values())[0].keys())) % cmap.N)
    metrics_sorted = dict(sorted(metrics.items()))

    for out_idx, (_, metric) in enumerate(metrics_sorted.items()):
        for inner_idx, (key, value) in enumerate(metric.items()):
            ax.barh(
                out_idx - bar_height * inner_idx,
                value,
                label=key,
                height=bar_height,
                color=colors[inner_idx],
            )

    ax.set_yticks(np.arange(len(metrics_sorted.keys())))
    ax.set_yticklabels(list(metrics_sorted.keys()))
    ax.set_xlabel(x_label)
    ax.vlines(0, -1, out_idx, linestyles="dashed", color="black", linewidth=0.3)
    hand, labl = ax.get_legend_handles_labels()
    if not no_legend:
        fig.legend(
            hand[: inner_idx + 1],
            labl[: inner_idx + 1],
            fontsize=10,
            markerscale=15,
            loc="outside lower center",
            ncol=min(len(labl), 5),
        )
    return fig


def plot_overview_metrics(
    plot_path,
    task_name,
    model_list,
    metrics,
    path_func=None,
    return_fig=False,
    sort_string="kmeans-audio_file_name",
):
    """
    Visualization of task performance by model accross all classes.
    Resulting plot is stored in the plot path.

    Parameters
    ----------
    plot_path : pathlib.Path object
        path to store overview plots
    task_name : str
        name of task
    model_list : list
        list of models
    metrics : dict
        performance dictionary
    sort_string : str
        string to sort the metrics by, defaults to "kmeans-audio_file_name"
    """
    # TODO when first ran mutliple models and then just one, metrics 
    # doesn't know the current model and this should be caught
    if not metrics:
        res_path = path_func(model_list[0]).plot_path.parent.parent.joinpath("overview")
        with open(res_path.joinpath(f"classification_results.json"), "r") as f:
            metrics = json.load(f)
        metrics = {
            k.split("(")[0]: v["overall"] for k, v in metrics.items() if task_name in k
        }

    if "classification" in task_name:
        metrics = {k: v["overall"] for k, v in metrics.items()}

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    if len(model_list) == 1 and model_list[0] not in metrics:
        error = (
            "\nIt seems like you have selected a single model in a folder where previously "
            "multiple models were computed. Try selecting at least two models, that way "
            "this error should be fixed."
        )
    elif not all([model in metrics for model in model_list]):
        raise AttributeError(
            "It seems like you have selected models for which the classification scores "
            "haven't been saved yet, but for some reason bacpipe didn't realize this. "
            "Try running bacpipe again with the setting `overwrite` set to `True`."
        )
    num_metrics = len(metrics[model_list[0]])
    bar_width = 1 / (num_metrics + 1)

    cmap = plt.cm.tab10
    cols = cmap(np.arange(num_metrics) % cmap.N)

    if task_name == "clustering":
        sort_by = lambda item: list(item[-1].values())[-1][sort_string]
    else:
        sort_by = lambda item: list(item[-1].values())[0]
    metrics = dict(sorted(metrics.items(), key=sort_by, reverse=True))
    if task_name == "clustering":
        metrics = {
            k: {
                k: v[sort_string]
                for k, v in metrics[k].items()
                if sort_string in v.keys()
            }
            for k, v in metrics.items()
        }

    for mod_idx, (model, d) in enumerate(metrics.items()):
        for i, (metric, value) in enumerate(d.items()):
            ax.bar(
                mod_idx - bar_width * i,
                value,
                label=metric,
                width=bar_width,
                color=cols[i],
            )
    ax.set_ylabel("Various Metrics")
    ax.set_xlabel("Models")
    ax.set_xticks(np.arange(len(metrics.keys())) - bar_width * (num_metrics - 1) / 2)
    ax.set_xticklabels(
        [model.upper() for model in metrics.keys()],
        rotation=45,
        horizontalalignment="right",
    )
    ax.set_title(f"Overall Metrics for {task_name} Across Models")

    fig.subplots_adjust(right=0.75, bottom=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Metrics",
        labels=d.keys(),
        fontsize=10,
    )
    if return_fig:
        return fig
    file = (
        f"overview_metrics_{task_name}_" 
        + "-".join([m[:2] for m in metrics.keys()]) 
        + ".png"
        )
    plot_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(
        plot_path.joinpath(file),
        dpi=300,
    )
    plt.close(fig)

