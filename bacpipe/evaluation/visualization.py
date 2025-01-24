from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from bacpipe.generate_embeddings import Loader
import yaml
from .clustering import get_centroid, get_clustering_scores

with open("bacpipe/path_settings.yaml", "rb") as f:
    bacpipe_settings = yaml.safe_load(f)

SPLIT_BY_FOLDER = True


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


def plot_embeddings(umap_embed_path, dim_reduction_model, axes=False, fig=False):
    files = umap_embed_path.iterdir()
    centroids = {}
    for file in files:
        if file.suffix == ".json" and dim_reduction_model in file.stem:
            with open(file, "r") as f:
                embeds_dict = json.load(f)
    split_data = data_split_by_labels(embeds_dict)

    if not fig:
        fig, axes = plt.subplots(figsize=(12, 8))
        return_axes = False
    else:
        return_axes = True

    # for embed in embeds_ar:
    if split_data is not None:
        for label in split_data:
            points = axes.plot(
                split_data[label]["x"],
                split_data[label]["y"],
                "o",
                label=label,
                markersize=0.5,
            )
            centroids[label] = get_centroid(split_data[label])
            c = darken_hex_color_bitwise(points[0]._color)
            axes.plot(
                centroids[label][0],
                centroids[label][1],
                "x",
                color=c,
                label=f"{label} centroid",
                markersize=12,
            )
        clustering_dict = get_clustering_scores(split_data, centroids)
        with open(umap_embed_path.joinpath("clustering_metrics.json"), "w") as f:
            json.dump(clustering_dict, f)
    else:
        embed_x = embeds_dict["x"]
        embed_y = embeds_dict["y"]
        file_stem = Path(embeds_dict["metadata"]["audio_files"]).stem
        axes.plot(embed_x, embed_y, "o", label=file_stem, markersize=0.5)

    if return_axes:
        return axes, clustering_dict
    else:
        fig, axes = set_legend(fig, axes)

        axes.set_title(f"{dim_reduction_model.upper()} embeddings")
        fig.savefig(umap_embed_path.joinpath("embed.png"), dpi=300)
        plt.close(fig)


def set_legend(fig, axes):
    # Calculate number of columns dynamically based on the number of labels
    num_labels = len(
        axes.get_legend_handles_labels()[1]
    )  # Number of labels in the legend
    ncol = min(num_labels, 6)  # Use 6 columns or fewer if there are fewer labels

    handles, labels = axes.get_legend_handles_labels()

    custom_marker = plt.scatter(
        [], [], marker="x", color="black", s=10
    )  # Empty scatter, only for the legend

    # Update the legend
    axes.legend(
        handles[::2] + [custom_marker],
        labels[::2] + ["centroids"],  # Use the handles and labels from the plot
        loc="upper center",  # Center the legend
        bbox_to_anchor=(0.5, -0.19),  # Position below the plot
        ncol=ncol,  # Number of columns
        markerscale=3,
    )

    # Adjust the layout so the legend doesn't overlap with the figure
    fig.subplots_adjust(bottom=0.25)
    return fig, axes


def data_split_by_labels(embeds_dict):
    meta = SimpleNamespace(**embeds_dict["metadata"])
    if SPLIT_BY_FOLDER:
        index = 0
        split_data = {}
        idx_increment = lambda x: meta.embedding_dimensions[x][0]
        concat = lambda x, x1: np.concatenate([x, x1])
        for idx, file in enumerate(meta.embedding_files):
            parent_dir = Path(file).parent.stem
            if not parent_dir in split_data:
                split_data[parent_dir] = {"x": [], "y": []}

            for k in split_data[parent_dir].keys():
                split_data[parent_dir][k] = concat(
                    split_data[parent_dir][k],
                    embeds_dict[k][index : index + idx_increment(idx)],
                )
            index += idx_increment(idx)

        return split_data


def return_rows_cols(num):
    if num <= 3:
        return 1, 3
    elif num > 3 and num <= 6:
        return 2, 3
    elif num > 6 and num <= 9:
        return 3, 3
    elif num > 9 and num <= 16:
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


def plot_comparison(audio_dir, embedding_models, dim_reduction_model):
    rows, cols = return_rows_cols(len(embedding_models))
    clust_dict = {}
    fig, axes = plt.subplots(rows, cols, figsize=set_figsize_for_comparison(rows, cols))
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.85, wspace=0.4, hspace=0.9
    )
    for idx, model in enumerate(embedding_models):
        ld = Loader(
            audio_dir, model_name=model, dim_reduction_model=dim_reduction_model
        )
        axes.flatten()[idx], clust_dict[model] = plot_embeddings(
            ld.embed_dir, dim_reduction_model, axes=axes.flatten()[idx], fig=fig
        )
        # metric_str = ", ".join([f"{k}={v:.3f}" for k, v in clust_dict[model].items()])
        metric_str = f"Silhouette Score= {clust_dict[model]['SS']:.3f}"
        axes.flatten()[idx].set_title(f"{model.upper()}\n{metric_str}")
    # fig.tight_layout()
    new_order = [
        k[0] for k in sorted(clust_dict.items(), key=lambda kv: kv[1]["SS"])[::-1]
    ]
    positions = {mod: ax.get_position() for mod, ax in zip(new_order, axes.flatten())}
    for model, ax in zip(embedding_models, axes.flatten()):
        ax.set_position(positions[model])
    # plt.legend(loc='lower left', ncol=6, bbox_to_anchor=(0, 0, 1, 1))
    set_legend(fig, axes.flatten()[-int(cols / 2) - 1])
    fig.suptitle(f"Comparison of {dim_reduction_model} embeddings", fontweight="bold")
    fig.savefig(ld.embed_dir.joinpath("comp_fig.png"), dpi=300)


def visualize_task_results(task_name, model_name, metrics):
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
    fig.savefig(
        Path(bacpipe_settings["task_results_dir"])
        .joinpath("plots")
        .joinpath(f"classsification_results_{task_name}_{model_name}.png"),
        dpi=300,
    )
