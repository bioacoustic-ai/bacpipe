import json

import matplotlib.pyplot as plt
import numpy as np

import bacpipe.embedding_evaluation.label_embeddings as le


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


def collect_dim_reduced_embeds(dim_reduced_embed_path, dim_reduction_model):
    """
    Return the dimensionality reduced embeddings of a model.

    Parameters
    ----------
    dim_reduced_embed_path : pathlib.Path object
        path to dim reduced embeddings
    dim_reduction_model : str
        name of feature extraction model

    Returns
    -------
    dict
        dimensionality reduced embeddings
    """
    files = dim_reduced_embed_path.iterdir()
    for file in files:
        if file.suffix == ".json" and dim_reduction_model in file.stem:
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

            dim_reduced_embed_path = le.get_dim_reduc_path_func(model_name)

            self.embeds[model_name] = collect_dim_reduced_embeds(
                dim_reduced_embed_path, self.dim_reduction_model
            )

        if remove_noise:
            return_labels = dict()
            return_embeds = dict()
            for key in self.labels[model_name].keys():

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

    c_label_dict = {lab: i for i, lab in enumerate(np.unique(labels))}
    points = plot_embedding_points(
        axes, embeds, split_data, labels, c_label_dict, **kwargs
    )

    if return_axes:
        return axes, c_label_dict, points
    elif dashboard:
        if dashboard_idx == 1:
            fig.set_size_inches(8, 6)
            set_colorbar_or_legend(
                fig, axes, points, c_label_dict, dashboard=dashboard, **kwargs
            )
        else:
            fig.set_size_inches(7, 6)

        return fig
    else:
        set_colorbar_or_legend(fig, axes, points, c_label_dict, **kwargs)

        axes.set_title(f"{dim_reduction_model.upper()} embeddings")
        fig.savefig(paths.plot_path.joinpath("embeddings.png"), dpi=300)
        plt.close(fig)


def init_embed_figure(fig, axes, bool_3d=False, **kwargs):
    if not fig:
        if bool_3d:
            fig, axes = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(figsize=(12, 8))
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
        inv = {v: k for k, v in ground_truth["label_dict"].items()}
        inv[-1.0] = "noise"
        # TODO -2 still in data
        labels["ground_truth"] = [inv[v] for v in ground_truth["labels"]]
        bool_noise = np.array(labels["ground_truth"]) == "noise"
    else:
        bool_noise = np.array([False] * len(list(labels.values())[0]))
    if len(list(le.get_paths(model_name).clust_path.glob("*.npy"))) > 0:
        clusts = [
            np.load(f, allow_pickle=True).item()
            for f in le.get_paths(model_name).clust_path.glob("*.npy")
        ]
        for clust in clusts:
            for name, values in clust.items():
                labels[name] = np.array(["noise"] * len(bool_noise), dtype=object)
                labels[name][~bool_noise] = [inv[v] for v in values]

    return labels, bool_noise


def set_colorbar_or_legend(fig, axes, points, c_label_dict, **kwargs):
    if len(c_label_dict.keys()) > 20:
        # Shrink main plot area to make space for colorbar
        fig.subplots_adjust(right=0.85)

        # Add colorbar axis manually (x0, y0, width, height) in figure coords
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # tweak as needed

        # Create colorbar in the custom axis
        cbar = fig.colorbar(points, cax=cbar_ax)

        locs = [*(int(len(c_label_dict) / 5) * np.arange(5)), -1]
        cbar.set_ticks([list(c_label_dict.values())[loc] for loc in locs])
        cbar.set_ticklabels([list(c_label_dict.keys())[loc] for loc in locs])
        cbar.set_label("Label")
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
        for label, data in split_data.items():
            if remove_noise and label == "noise":
                continue
            points = axes.scatter(
                data[0],
                data[1],
                label=label,
                s=1,
                cmap=cmap,
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

    # Update the legend
    if dashboard:
        fig.subplots_adjust(right=0.8)
        ncol = 1
        fig.legend(
            new_handles,
            new_labels,  # Use the handles and labels from the plot
            loc="outside right",  # Center the legend
            bbox_to_anchor=(1.12, 0.5),
            ncol=ncol,  # Number of columns
            markerscale=4,
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

    for idx, model in enumerate(models):
        paths = le.get_paths(model)

        axes.flatten()[idx], c_label_dict, points = plot_embeddings(
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

    fig, _ = set_colorbar_or_legend(
        fig, axes.flatten()[0], points, c_label_dict, dashboard=dashboard, **kwargs
    )
    [ax.remove() for ax in axes.flatten()[idx + 1 :]]
    reorder_embeddings_by_clustering_performance(plot_path, axes, models)

    fig.suptitle(f"Comparison of {dim_reduction_model} embeddings", fontweight="bold")
    if not dashboard:
        fig.savefig(plot_path.joinpath("comp_fig.png"), dpi=300)
    else:
        return fig


def reorder_embeddings_by_clustering_performance(
    plot_path, axes, models, order_metric="ARI(kmeans)"
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
        sorted(clust_dict.items(), key=lambda kv: kv[1][order_metric], reverse=True)
    )
    positions = {mod: ax.get_position() for mod, ax in zip(new_order, axes.flatten())}
    for model, ax in zip(models, axes.flatten()):
        if not model in positions.keys():
            continue
        ax.set_position(positions[model])


def plot_classification_results(paths, task_name, metrics):
    """
    Save model specific classification results in the model specific
    plot path.

    Parameters
    ----------
    paths : SimpleNamespace object
        dictlike object with path attributes to save and load data
    task_name : str
        name of task
    metrics : dict
        performance dictionary
    """
    model_name = paths.labels_path.parent.stem
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(metrics["per_class_accuracy"].values())) % cmap.N)
    ax.bar(
        metrics["per_class_accuracy"].keys(),
        metrics["per_class_accuracy"].values(),
        width=0.5,
        color=colors,
    )
    metrics_string = "".join(
        [f"{k}: {v:.3f} | " for k, v in metrics["overall"].items()]
    )
    fig.suptitle(
        f"Per Class Metrics for {task_name} "
        f"classification with {model_name.upper()} embeddings\n"
        f"{metrics_string}"
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Classes")
    ax.set_xticks(range(len(metrics["per_class_accuracy"])))
    ax.set_xticklabels(metrics["per_class_accuracy"].keys(), rotation=90)
    fig.subplots_adjust(bottom=0.3)
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
        for file in getattr(paths, f"{task[:5]}_path").rglob("*.json"):
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
        json.dump(metrics, f)

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


def plot_overview_metrics(plot_path, task_name, model_list, metrics):
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
    """
    if "classification" in task_name:
        metrics = {k: v["Overall Metrics"] for k, v in metrics.items()}

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    num_metrics = len(metrics[model_list[0]])
    bar_width = 1 / (num_metrics + 1)

    cmap = plt.cm.tab10
    cols = cmap(np.arange(num_metrics) % cmap.N)

    metrics = dict(
        sorted(
            metrics.items(), key=lambda item: list(item[-1].values())[0], reverse=True
        )
    )

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
    per_class_metrics = {m: v["Per Class Metrics"] for m, v in metrics.items()}
    overall_metrics = {m: v["Overall Metrics"] for m, v in metrics.items()}
    num_classes = len(per_class_metrics[model_list[0]].keys())
    fig_width = max(12, num_classes * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

    cmap = plt.cm.tab10
    model_colors = cmap(np.arange(len(model_list)) % cmap.N)

    d = {m: v["Macro Accuracy"] for m, v in overall_metrics.items()}
    model_list = sorted(d, key=d.get, reverse=True)
    all_classes = sorted(per_class_metrics[model_list[0]].keys())

    for i, model_name in enumerate(model_list):
        class_values = per_class_metrics[model_name].values()

        ax.scatter(
            np.arange(len(all_classes)),
            class_values,
            color=model_colors[i],
            label=f"{model_name.upper()} "
            + f"(accuracy: {overall_metrics[model_name]['Macro Accuracy']:.3f})",
            s=100,
        )

        ax.plot(
            np.arange(len(all_classes)),
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
