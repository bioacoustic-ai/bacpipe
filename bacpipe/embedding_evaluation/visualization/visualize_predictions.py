import json

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import seaborn as sns

import logging

logger = logging.getLogger(__name__)



def plot_classification_results(
    task_name,
    paths=None,
    metrics=None,
    return_fig=False,
    path_func=None,
    model_name=None,
):
    """
    Save model specific classification results in the model specific
    plot path, displayed as horizontal bars.

    Parameters
    ----------
    task_name : str
        name of task
    paths : SimpleNamespace object
        path to store plots
    metrics : dict
        classification performance
    return_fig : bool
        if True the figure will be returned, by default False
    path_func : function
        function to return the paths when model name is given
    model_name : str
        name of model, by default None

    Returns
    -------
    plt object
        figure handle
    """
    if path_func and model_name:
        paths = path_func(model_name)
    if not metrics:
        class_path = paths.class_path / f"class_results_{task_name}.json"
        if not class_path.exists():
            error = (
                f"\nThe classification file {class_path} does not exist. Perhaps it was not "
                "created yet. To avoid getting this error, make sure you have not "
                " included 'classification' in the 'evaluation_tasks'. If you want to compute "
                "classification, make sure to set `overwrite=True`."
            )
            logger.exception(error)
            raise AssertionError(error)

        with open(paths.class_path / f"class_results_{task_name}.json", "r") as f:
            metrics = json.load(f)

    # Filter overall metrics if needed
    metrics["overall"] = {
        k: v for k, v in metrics["overall"].items() if not "micro" in k
    }

    # Sort classes by accuracy for better visualization
    class_items = sorted(
        metrics["per_class_accuracy"].items(), key=lambda x: x[1], reverse=True
    )
    class_names = [item[0] for item in class_items]
    class_values = [item[1] for item in class_items]

    # Set figure size based on number of classes and return_fig
    if return_fig:
        # For dashboard, make height adapt to number of classes
        height = max(4, len(class_names) * 0.3)
        fig, ax = plt.subplots(1, 1, figsize=(5, height))
        fontsize = 10
    else:
        height = max(8, len(class_names) * 0.4)
        fig, ax = plt.subplots(1, 1, figsize=(12, height))
        fontsize = 14

    model_name = paths.labels_path.parent.stem
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(class_names)) % cmap.N)

    # Create horizontal bars
    ax.barh(
        range(len(class_names)),
        class_values,
        height=0.6,
        color=colors,
    )

    # Create metrics string
    metrics_string = "".join(
        [f"{k}: {v:.3f} | " for k, v in metrics["overall"].items()]
    )

    fig.suptitle(
        f"Classwise accuracy for {task_name} "
        f"classification with {model_name.upper()} embeddings\n"
        f"{metrics_string}",
        fontsize=fontsize,
    )

    # Adjust labels for horizontal orientation
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Classes")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)

    # Add value labels at the end of each bar
    for i, v in enumerate(class_values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    # Set x-axis limits for better visualization
    ax.set_xlim(0, min(1.0, max(class_values) * 1.15))

    # Add grid lines for easier reading
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if return_fig:
        return fig

    path = paths.plot_path
    fig.savefig(
        path.joinpath(f"class_results_{task_name}_{model_name}.png"),
        dpi=300,
    )
    plt.close(fig)


def load_results(path_func, task, model_list):
    """
    Load the task results into a dict and return them. For classification
    multiple subtasks exist, so do them seperately.

    Parameters
    ----------
    path_func : function
        returns model specific tasks when model is given
    task : str
        name of task
    model_list : list
        list of models

    Returns
    -------
    dict
        performance for different tasks and models
    """
    metrics = {}
    for model_name in model_list:
        paths = path_func(model_name)
        for file in getattr(paths, f"{task[:5]}_path").rglob("*results*.json"):
            if task == "classification":
                subtask = file.stem.split("_")[-1]
                metrics[f"{model_name}({subtask})"] = json.load(open(file, "r"))
            else:
                metrics[model_name] = json.load(open(file, "r"))
    return metrics



def plot_per_class_metrics(plot_path, task_name, model_list, metrics):
    """
    Visualization of per class results. Resulting figure is stored in
    plot path. Models are sorted by the value of the first entry.

    Parameters
    ----------
    plot_path : pathlib.Path object
        path to store plot in
    task_name : str
        name of task
    model_list : list
        list of models
    metrics : dict
        performance dictionary
    """
    per_class_metrics = {m: v["per_class_accuracy"] for m, v in metrics.items()}
    overall_metrics = {m: v["overall"] for m, v in metrics.items()}
    num_classes = len(per_class_metrics[model_list[0]].keys())
    fig_width = max(12, num_classes * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

    cmap = plt.cm.tab10
    model_colors = cmap(np.arange(len(model_list)) % cmap.N)

    d = {m: v["macro_accuracy"] for m, v in overall_metrics.items()}
    model_list = sorted(d, key=d.get, reverse=True)
    all_classes = sorted(per_class_metrics[model_list[0]].keys())

    for i, model_name in enumerate(model_list):
        class_values = per_class_metrics[model_name].values()

        ax.scatter(
            np.arange(len(class_values)),
            class_values,
            color=model_colors[i],
            label=f"{model_name.upper()} "
            + f"(accuracy: {overall_metrics[model_name]['macro_accuracy']:.3f})",
            s=100,
        )

        ax.plot(
            np.arange(len(class_values)),
            class_values,
            color=model_colors[i],
            linestyle="-",  # Solid line
            linewidth=1.5,
        )

    fig.suptitle(
        f"Per class metrics for {task_name} across models",
        fontsize=14,
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Classes")
    ax.set_xticks(np.arange(len(all_classes)))
    ax.set_xticklabels(all_classes, rotation=90)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Models", fontsize=10)

    fig.subplots_adjust(right=0.65, bottom=0.3)
    file_name = (
        f"comparison_{task_name.replace(' ', '_')}_" 
        + "-".join([m[:2] for m in model_list]) 
        + ".png"
    )
    plot_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(
        plot_path.joinpath(file_name),
        dpi=300,
    )
    plt.close(fig)

        
def plot_classification_heatmap(
    accumulated_presence, timestamps, accumulate_by, species, threshold
    ):
    
    fig = Figure(figsize=[11, 8])
    ax = fig.subplots()
    
    fig.suptitle(
        f'Presence heatmap for {species} with threshold of {threshold}',
        fontsize=10
        )
    sns.heatmap(
        accumulated_presence.T, 
        vmin=0,
        # vmax=1,
        cmap='viridis',
        cbar_kws={'label': 'Binary presence per hour'},
        ax=ax
        )
    
    y_locs, yticklabels = ax.get_yticks(), ax.get_yticklabels()
    if accumulate_by == 'day':
        labels = np.unique([ts.date() for ts in timestamps])
        ax.set_ylabel('dates')
    elif accumulate_by == 'month':
        labels = np.unique(
            [f'{date.year}-{date.month}' for date in timestamps], 
            axis=0
            )
        ax.set_ylabel('months')
    elif accumulate_by == 'week':
        labels = np.unique(
            [f'{date.year}-{date.isocalendar().week}' for date in timestamps], 
            axis=0
            )
        ax.set_ylabel('weeks')
    selected_labels = labels[[int(i.get_text()) for i in yticklabels]]
    x_locs, labels = ax.get_xticks(), ax.get_xticklabels()
    x_idxs = [0, 6, 12, 18, 23]
    ax.set_xticks(x_locs[x_idxs], np.array(labels)[x_idxs])
        
    
    ax.set_xlabel('hours')
    ax.set_yticks(y_locs)
    ax.set_yticklabels(selected_labels)
    
    # force the rotation on the axis itself
    ax.tick_params(axis='y', rotation=0)
    
    
    fig.set_size_inches(6, 5)
    fig.set_dpi(300)
    fig.tight_layout()
    return fig      
