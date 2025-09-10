import json

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import bacpipe.embedding_evaluation.label_embeddings as le

import logging

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib

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


def darken_hex_color_bitwise(hex_color):
    """
    Darkens a hex color using the bitwise operation: (color & 0xfefefe) >> 1.

    Parameters:
        hex_color (str): The hex color string (e.g., '#1f77b4').

    Returns:
        str: The darkened hex color.
    """
    # Remove '#' and convert hex color to an integer
    color_int = int(hex_color.lstrip("#"), 16)

    # Apply the bitwise operation to darken the color
    darkened_color_int = (color_int & 0xFEFEFE) >> 1

    # Convert back to a hex string and return with leading '#'
    return f"#{darkened_color_int:06x}"


def collect_dim_reduced_embeds(
    model_name, dim_reduced_embed_path, dim_reduction_model, **kwargs
):
    """
    Return the dimensionality reduced embeddings of a model.

    Parameters
    ----------
    model_name : str
        name of model
    dim_reduced_embed_path : pathlib.Path object
        path to dim reduced embeddings
    dim_reduction_model : str
        name of feature extraction model

    Returns
    -------
    dict
        dimensionality reduced embeddings
    """
    files = list(dim_reduced_embed_path.iterdir())
    if len(files) == 0:
        logger.warning(
            "No dimensionality reduced embeddings found for "
            f"{dim_reduction_model}. In fact the directory "
            f"{dim_reduced_embed_path} is empty. Deleting directory."
        )
        dim_reduced_embed_path.rmdir()
        dim_reduced_embed_path = le.get_dim_reduc_path_func(
            model_name, dim_reduction_model=dim_reduction_model, **kwargs
        )
        files = list(dim_reduced_embed_path.iterdir())
    for file in files:
        if file.suffix == ".json":  # and dim_reduction_model in file.stem:
            with open(file, "r") as f:
                embeds_dict = json.load(f)
    return embeds_dict


class EmbedAndLabelLoader:
    def __init__(self, dim_reduction_model, dashboard=False, **kwargs):
        self.labels = dict()
        self.embeds = dict()
        self.split_data = dict()
        self.bool_noise = dict()
        self.dashboard = dashboard
        self.dim_reduction_model = dim_reduction_model
        self.kwargs = kwargs

    def get_data(self, model_name, label_by, remove_noise=False, **kwargs):
        if not model_name in self.labels.keys():

            tup = get_labels_for_plot(model_name, **self.kwargs)
            self.labels[model_name], self.bool_noise[model_name] = tup

            dim_reduced_embed_path = le.get_dim_reduc_path_func(
                model_name, dim_reduction_model=self.dim_reduction_model, **kwargs
            )

            self.embeds[model_name] = collect_dim_reduced_embeds(
                model_name, dim_reduced_embed_path, self.dim_reduction_model, **kwargs
            )

        if remove_noise:
            return_labels = dict()
            return_embeds = dict()
            for key in self.labels[model_name].keys():
                if "noise" in key:
                    return_labels[key] = self.labels[model_name][key]
                else:
                    return_labels[key] = np.array(
                        self.labels[model_name][key], dtype=object
                    )[~self.bool_noise[model_name]]

            return_embeds["x"] = np.array(self.embeds[model_name]["x"])[
                ~self.bool_noise[model_name]
            ]

            return_embeds["y"] = np.array(self.embeds[model_name]["y"])[
                ~self.bool_noise[model_name]
            ]
        else:
            return_labels = self.labels[model_name]
            return_embeds = self.embeds[model_name]

        return_splits = data_split_by_labels(return_embeds, return_labels[label_by])
        return (
            return_labels[label_by],
            return_embeds,
            return_splits,
        )


def plot_embeddings(
    loader,
    model_name,
    label_by,
    paths=None,
    dim_reduction_model=None,
    axes=False,
    fig=False,
    dashboard=False,
    dashboard_idx=None,
    **kwargs,
):
    """
    Generate figures and axes to plot points corresponding to embeddings.
    This function can also be called and given figure and axes handeles.
    In that case the existing handles will be used to add the points and
    configure the axes and labels.

    Parameters
    ----------
    loader : EmbedAndLabelLoader object
        contains the labels and embeddings by model, for quicker loading
    model_name : str
        name of model
    label_by : str, optional
        key of default_labels dict, by default "audio_file_name"
    paths : SimpleNamespace object, optional
        object with path attributes, defaults to None
    dim_reduction_model : str
        name of dim reduced model
    axes : plt object, optional
        axes handle, by default False
    fig : plt object, optional
        figure handle, by default False
    dashboard : bool, optional
        whether the calls comes from the dashboard, by deafult False
    dashboard_idx : int, optional
        index of dashboard plot, relevant for legend placement

    Returns
    -------
    plt object
        axes handles is axes handles were given
    dict
        color dictionary for legend
    list
        plt point objects for legend of colorbar
    """
    labels, embeds, split_data = loader.get_data(model_name, label_by, **kwargs)

    fig, axes, return_axes = init_embed_figure(fig, axes, **kwargs)

    if label_by == 'audio_file_name':
        new_labels = [Path(l).stem+Path(l).suffix for l in labels]
        new_split_data = dict()
        for label in split_data.keys():
            new_label = Path(label).stem+Path(label).suffix
            new_split_data[new_label] = split_data[label]
        split_data = new_split_data

    c_label_dict = {lab: i for i, lab in enumerate(np.unique(labels))}
    points = plot_embedding_points(
        axes, embeds, split_data, labels, c_label_dict, **kwargs
    )

    if return_axes:
        return axes, c_label_dict, points
    elif dashboard:
        fig.set_size_inches(6, 5)
        fig.set_dpi(300)
        fig.tight_layout()
        set_colorbar_or_legend(
            fig,
            axes,
            points,
            c_label_dict,
            dashboard=dashboard,
            label_by=label_by,
            **kwargs,
        )
        return fig
    else:
        set_colorbar_or_legend(
            fig, axes, points, c_label_dict, label_by=label_by, **kwargs
        )

        axes.set_title(f"{dim_reduction_model.upper()} embeddings")
        fig.savefig(paths.plot_path.joinpath("embeddings.png"), dpi=300)
        plt.close(fig)


def init_embed_figure(fig, axes, bool_3d=False, **kwargs):
    if not fig:
        if bool_3d:
            fig, axes = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(figsize=(12, 8), dpi=400)
        return_axes = False
    else:
        return_axes = True
    axes.set_xticks([])
    axes.set_yticks([])
    return fig, axes, return_axes


def get_labels_for_plot(model_name=None, **kwargs):
    labels = dict()
    labels = le.get_default_labels(model_name, **kwargs)

    if le.get_paths(model_name).labels_path.joinpath("ground_truth.npy").exists():
        ground_truth = le.get_ground_truth(model_name)
        for label_column in [key for key in ground_truth.keys() if "label:" in key]:
            label = label_column.split("label:")[-1]
            inv = {v: k for k, v in ground_truth[f"label_dict:{label}"].items()}
            inv[-1.0] = "noise"
            inv[-2.0] = "noise"
            # technically -2.0 is not noise, but corresponds to sections
            # with multiple sources vocalizing simultaneously
            labels[label] = [inv[v] for v in ground_truth[label_column]]
            bool_noise = np.array(labels[label]) == "noise"
    else:
        bool_noise = np.array([False] * len(list(labels.values())[0]))
    if len(list(le.get_paths(model_name).clust_path.glob("*.npy"))) > 0:
        clusts = [
            np.load(f, allow_pickle=True).item()
            for f in le.get_paths(model_name).clust_path.glob("*.npy")
        ]
        for clust in clusts:
            for name, values in clust.items():
                if "kmeans" in name:
                    labels[name] = values
                else:
                    labels[name] = np.array(["noise"] * len(bool_noise), dtype=object)
                    labels[name][~bool_noise] = [inv[v] for v in values]

    return labels, bool_noise


def set_colorbar_or_legend(fig, axes, points, c_label_dict, label_by, **kwargs):
    if len(c_label_dict.keys()) > 20:
        if (
            isinstance(list(c_label_dict.keys())[0], int)
            or len(list(c_label_dict.keys())[0]) < 12
        ):
            fontsize = 9
        else:
            fontsize = 6

        # Shrink main plot area to make space for colorbar
        fig.subplots_adjust(right=0.7)

        # Add colorbar axis manually (x0, y0, width, height) in figure coords
        cbar_ax = fig.add_axes([0.72, 0.05, 0.03, 0.9])  # tweak as needed

        # Create colorbar in the custom axis
        cbar = fig.colorbar(points, cax=cbar_ax)

        locs = [*(int(len(c_label_dict) / 5) * np.arange(5)), -1]
        cbar.set_ticks([list(c_label_dict.values())[loc] for loc in locs])
        cbar.set_ticklabels(
            [list(c_label_dict.keys())[loc] for loc in locs], fontsize=fontsize
        )
        cbar.set_label(label_by.replace("_", " "), fontsize=10)
    else:
        hands, labs = axes.get_legend_handles_labels()
        fig, axes = set_legend(hands, labs, fig, axes, **kwargs)
    return fig, axes


def plot_embedding_points(
    axes, embeds, split_data, labels, c_label_dict, remove_noise=False, **kwargs
):
    """
    Plot embeddings in scatter plot.

    Parameters
    ----------
    axes : plt object
        axes handle
    embeds : dict
        embeddings
    split_data : dict
        data split by label
    labels : list
        labels of the data
    c_label_dict : dict
        linking labels to ints for coloring
    remove_noise : bool, optional
        remove noise or not, defaults to False

    Returns
    -------
    plt object
        axes points
    """
    if len(c_label_dict.keys()) > 20:
        import matplotlib.cm as cm

        cmap = cm.viridis  # or 'plasma', 'inferno', 'magma', etc.
        # if remove_noise:
        #     bool_labels = np.array(labels) != "noise"
        #     labels = np.array(labels)[bool_labels]
        # else:
        #     bool_labels = [True] * len(labels)

        num_labels = np.array([c_label_dict[lab] for lab in labels])
        points = axes.scatter(
            # np.array(embeds["x"])[bool_labels],
            # np.array(embeds["y"])[bool_labels],
            np.array(embeds["x"]),
            np.array(embeds["y"]),
            c=num_labels,
            label=labels,
            s=1,
            cmap=cmap,
        )
    else:
        cmap = plt.cm.tab20
        colors = cmap(np.arange(len(c_label_dict.keys())) % cmap.N)
        for idx, (label, data) in enumerate(split_data.items()):
            if remove_noise and label == "noise":
                continue
            points = axes.scatter(
                data[0],
                data[1],
                label=label,
                s=1,
                color=colors[idx],
            )
    return points


def set_legend(
    handles, labels, fig, axes, bool_plot_centroids=False, dashboard=False, **kwargs
):
    """
    Create the legend for embeddings visualization plots.

    Parameters
    ----------
    handles : list
        list of legend handles
    labels : list
        list of labels for legend
    fig : plt.fig object
        figure handle
    axes : plt.axes object
        axes handle
    bool_plot_centroids : bool, optional
        if True centroids of each class will be plotted, by default True
    dashboard : bool
        if dashboard called this function or not

    Returns
    -------
    plt.fig object
        figure handle
    plt.axes object
        axes handle
    """

    # Calculate number of columns dynamically based on the number of labels
    num_labels = len(labels)  # Number of labels in the legend
    ncol = min(num_labels, 5)  # Use 6 columns or fewer if there are fewer labels

    if bool_plot_centroids:
        custom_marker = plt.scatter(
            [], [], marker="x", color="black", s=10
        )  # Empty scatter, only for the legend
        new_handles = handles[::2] + [custom_marker]
        new_labels = labels[::2] + ["centroids"]
    else:
        new_handles = handles
        new_labels = labels
    if dashboard:
        fig.subplots_adjust(right=0.7)

        fig.legend(
            new_handles,
            new_labels,
            loc="outside right",
            markerscale=4 if dashboard else 6,
            fontsize=7,
            frameon=False,
        )
    else:

        fig.subplots_adjust(bottom=0.2)
        fig.legend(
            new_handles,
            new_labels,  # Use the handles and labels from the plot
            loc="outside lower center",  # Center the legend
            ncol=ncol,  # Number of columns
            markerscale=6,
        )
    return fig, axes


def data_split_by_labels(embeds_dict, labels):
    """
    Split data by labels for scatterplots.

    Parameters
    ----------
    embeds_dict : dict
        embeddings by model
    labels : list
        list of labels

    Returns
    -------
    dict
        x and y data corresponding to labels
    """
    split_data = {}
    uni_labels = np.unique(labels)
    if len(uni_labels) > 20:
        split_data["all"] = np.array(
            [
                np.array(embeds_dict["x"]),
                np.array(embeds_dict["y"]),
            ]
        )
    else:
        for label in uni_labels:  # TODO don't do this for more than 20 categories
            split_data[str(label)] = np.array(
                [
                    np.array(embeds_dict["x"])[np.array(labels) == label],
                    np.array(embeds_dict["y"])[np.array(labels) == label],
                ]
            )

    return split_data


def return_rows_cols(num):
    if num <= 3:
        return 1, 3
    elif num > 3 and num <= 6:
        return 2, 3
    elif num > 6 and num <= 9:
        return 3, 3
    elif num > 9 and num <= 12:
        return 3, 4
    elif num > 12 and num <= 16:
        return 4, 4
    elif num > 16 and num <= 20:
        return 4, 5
    else:
        return 5, num // 5


def set_figsize_for_comparison(rows, cols):
    if rows == 1:
        return (12, 5)
    elif rows == 2:
        return (12, 7)
    elif rows == 3:
        return (12, 8)
    elif rows > 3:
        return (12, 10)


def plot_comparison(
    plot_path,
    models,
    dim_reduction_model,
    bool_spherical=False,
    dashboard=False,
    loader=None,
    evaluation_task=[],
    **kwargs,
):
    """
    Create big overview visualization of all embeddings spaces. Labels
    are chosen from ground_truth and if that does not exist, default
    lables are used.

    Parameters
    ----------
    plot_path : pathlib.Path object
        path to store overview plots
    models : list
        list of models
    dim_reduction_model : str
        name of dimensionality reduction model
    bool_spherical : bool, optional
        if True 3d embeddings will be plotted, by default False
    dashboard : bool, optional
        if dashboard called this function or not
    loader : EmbedAndLabelLoader object
        object containing embeds and labels by model for quicker loading
    evaluation_task : list, optional
        list of tasks to evaluate, by default []

    Returns
    -------
    plt object
        figure handle
    """
    rows, cols = return_rows_cols(len(models))

    if not bool_spherical:
        fig, axes = plt.subplots(
            rows, cols, figsize=set_figsize_for_comparison(rows, cols)
        )
    else:
        fig, axes = plt.subplots(
            rows,
            cols,
            subplot_kw={"projection": "3d"},
            figsize=set_figsize_for_comparison(rows, cols),
        )
    if not dashboard:
        vis_loader = EmbedAndLabelLoader(dim_reduction_model, **kwargs)
    else:
        vis_loader = loader

    c_label_dict, points = {}, {}
    for idx, model in enumerate(models):
        paths = le.get_paths(model)

        axes.flatten()[idx], c_label_dict[idx], points[idx] = plot_embeddings(
            vis_loader,
            model,
            paths=paths,
            dim_reduction_model=dim_reduction_model,
            axes=axes.flatten()[idx],
            fig=fig,
            bool_plot_centroids=False,
            dashboard=dashboard,
            **kwargs,
        )
        axes.flatten()[idx].set_title(f"{model.upper()}")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.2)
    colorbar_idx = np.argmax([len(d) for d in c_label_dict.values()])

    fig, _ = set_colorbar_or_legend(
        fig,
        axes.flatten()[colorbar_idx],
        points[colorbar_idx],
        c_label_dict[colorbar_idx],
        dashboard=dashboard,
        **kwargs,
    )
    [ax.remove() for ax in axes.flatten()[idx + 1 :]]
    if "clustering" in evaluation_task:
        reorder_embeddings_by_clustering_performance(plot_path, axes, models)

    fig.suptitle(f"Comparison of {dim_reduction_model} embeddings", fontweight="bold")
    if not dashboard:
        fig.savefig(plot_path.joinpath("comp_fig.png"), dpi=300)
        plt.close(fig)
    else:
        return fig


def reorder_embeddings_by_clustering_performance(
    plot_path, axes, models, order_metric="ground_truth-kmeans"
):
    """
    Reorder the embedding overview plot by clustering performance.

    Parameters
    ----------
    plot_path : pathlib.Path object
        path to store plots and results comparing all models
    axes : plt.axes object
        handle for figures axes
    models : list
        list of models
    order_metric : str
        key corresponding to a metric in the clustering_results.json file.
        Defaults to "ARI(kmeans)"
    """
    clust_dict = json.load(open(plot_path.joinpath("clustering_results.json"), "r"))
    new_order = dict(
        sorted(
            clust_dict.items(), key=lambda kv: kv[1]["ARI"][order_metric], reverse=True
        )
    )
    positions = {mod: ax.get_position() for mod, ax in zip(new_order, axes.flatten())}
    for model, ax in zip(models, axes.flatten()):
        if not model in positions.keys():
            continue
        ax.set_position(positions[model])


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
            raise AssertionError(
                f"The classification file {class_path} does not exist. Perhaps it was not "
                "created yet. To avoid getting this error, make sure you have not "
                " included 'classificaion' in the 'evaluation_tasks'. If you want to compute "
                "classification, make sure to set `overwrite=True`."
            )

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
        plot_overview_metrics(plot_path, task_name, model_list, metrics)


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
        raise AssertionError(
            f"The clustering file {clust_path} does not exist. Perhaps it was not "
            "created yet. To avoid getting this error set `overwrite=True`."
        )

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
    file = f"overview_metrics_{task_name}_" + "-".join(metrics.keys()) + ".png"
    plot_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(
        plot_path.joinpath(file),
        dpi=300,
    )
    plt.close(fig)


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
        f"comparison_{task_name.replace(' ', '_')}_" + "-".join(model_list) + ".png"
    )
    plot_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(
        plot_path.joinpath(file_name),
        dpi=300,
    )
    plt.close(fig)


#################################################################


def plot_violins(left, right):
    import pandas as pd
    import seaborn as sns

    val = []
    typ = []
    cat = []
    for idx, (intra, inter) in enumerate(zip(left, right)):
        val.append(intra.tolist())
        val.append(inter.tolist())
        typ.extend(["Intra"] * len(intra))
        typ.extend(["Inter"] * len(inter))
        cat.extend([f"Group {idx}"] * len(intra))
        cat.extend([f"Group {idx}"] * len(inter))

    # Convert to long-form format
    data_long = pd.DataFrame(
        {"Value": np.concatenate(val), "Type": typ, "Category": cat}
    )

    # Create the violin plot
    plt.figure(figsize=(14, 8))
    sns.violinplot(
        x="Category",
        y="Value",
        hue="Type",
        data=data_long,
        split=True,
        inner="quartile",
    )

    plt.show()
