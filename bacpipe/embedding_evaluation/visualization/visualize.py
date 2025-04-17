import yaml
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import bacpipe.embedding_evaluation.label_embeddings as le
from bacpipe.generate_embeddings import Loader
from bacpipe.embedding_evaluation.clustering.cluster_embeddings import (
    get_centroid,
    get_clustering_scores,
)

with open("bacpipe/settings.yaml", "rb") as f:
    bacpipe_settings = yaml.load(f, Loader=yaml.CLoader)

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


def collect_embeddings(dim_reduced_embed_path, dim_reduction_model):
    files = dim_reduced_embed_path.iterdir()
    for file in files:
        if file.suffix == ".json" and dim_reduction_model in file.stem:
            with open(file, "r") as f:
                embeds_dict = json.load(f)
    # split_data = data_split_by_labels(embeds_dict)
    return embeds_dict


def plot_centroids(axes, centroids, label, split_data, points):
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
    return centroids


def plot_points(axes, split_data, label, bool_spherical):
    if bool_spherical:
        x = np.sin(split_data[label]["x"]) * np.cos(split_data[label]["y"])
        y = np.sin(split_data[label]["x"]) * np.sin(split_data[label]["y"])
        z = np.cos(split_data[label]["x"])
        points = axes.plot(
            x,
            y,
            z,
            "o",
            label=label,
            markersize=0.5,
        )
    else:
        points = axes.plot(
            split_data[label]["x"],
            split_data[label]["y"],
            "o",
            label=label,
            markersize=0.5,
        )
    return points


def plot_embeddings(
    dim_reduced_embed_path,
    dim_reduction_model,
    default_labels,
    ground_truth=None,
    axes=False,
    fig=False,
    bool_plot_centroids=True,
    bool_spherical=False,
    label_by="audio_file_name",
):
    embeds = collect_embeddings(dim_reduced_embed_path, dim_reduction_model)
    split_data = data_split_by_labels(embeds)
    if not fig:
        if bool_spherical:
            fig, axes = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(figsize=(12, 8))
        return_axes = False
    else:
        return_axes = True

    if not ground_truth:
        labels = default_labels[label_by]
        remove_noise = False
    else:
        labels = ground_truth["labels"]
        remove_noise = True

    centroids = {}

    c_label_dict = {lab: i for i, lab in enumerate(np.unique(labels))}
    # Normalize and pick a continuous colormap

    import matplotlib.cm as cm

    cmap = cm.viridis  # or 'plasma', 'inferno', 'magma', etc.
    if remove_noise:
        num_labels = np.array([c_label_dict[lab] for lab in labels[labels != -1]])
        sss = axes.scatter(
            np.array(embeds["x"])[labels != -1],
            np.array(embeds["y"])[labels != -1],
            c=num_labels,
            s=1,
            cmap=cmap,
        )
    else:
        num_labels = np.array([c_label_dict[lab] for lab in labels])
        sss = axes.scatter(
            embeds["x"],
            embeds["y"],
            c=num_labels,
            s=1,
            cmap=cmap,
        )
    if bool_plot_centroids:
        centroids = plot_centroids(axes, centroids, label, split_data, points)
    clustering_dict = get_clustering_scores(split_data, centroids)
    with open(dim_reduced_embed_path.joinpath("clustering_metrics.json"), "w") as f:
        json.dump(clustering_dict, f)

    if return_axes:
        return axes, clustering_dict
    else:
        # handles, labels = axes.get_legend_handles_labels()
        # handl = []
        # for label in np.unique(default_labels[label_by]):
        #     handl.append(handles[np.where(np.array(labels) == label)[0][0]])
        cbar = plt.colorbar(sss, ax=axes)
        locs = [0, int(len(c_label_dict) / 3), int(len(c_label_dict) / 3) * 2, -1]
        cbar.set_ticks([list(c_label_dict.values())[loc] for loc in locs])
        cbar.set_ticklabels([list(c_label_dict.keys())[loc] for loc in locs])
        cbar.set_label("Label")
        # fig, axes = set_legend(handl,
        #                        np.unique(default_labels[label_by]),
        #                        fig,
        #                        axes,
        #                        bool_plot_centroids=bool_plot_centroids)

        axes.set_title(f"{dim_reduction_model.upper()} embeddings")
        fig.savefig(dim_reduced_embed_path.joinpath("embed.png"), dpi=300)
        plt.close(fig)


def set_legend(handles, labels, fig, axes, bool_plot_centroids=True):
    # handles, labels = axes.get_legend_handles_labels()

    # Calculate number of columns dynamically based on the number of labels
    num_labels = len(labels)  # Number of labels in the legend
    ncol = min(num_labels, 6)  # Use 6 columns or fewer if there are fewer labels

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
    fig.legend(
        new_handles,
        new_labels,  # Use the handles and labels from the plot
        loc="outside lower center",  # Center the legend
        # bbox_to_anchor=(0.5, 1.05),  # Position below the plot
        ncol=ncol,  # Number of columns
        markerscale=6,
    )
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
    paths, audio_dir, embedding_models, dim_reduction_model, bool_spherical=False
):
    rows, cols = return_rows_cols(len(embedding_models))
    clust_dict = {}
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

    fig.subplots_adjust(
        left=0.1, bottom=0.15, right=0.9, top=0.85, wspace=0.4, hspace=0.9
    )
    for idx, model in enumerate(embedding_models):
        ld = Loader(
            audio_dir, model_name=model, dim_reduction_model=dim_reduction_model
        )
        default_labels = le.create_default_labels(paths, model, audio_dir)
        if le.labels_path.joinpath("ground_truth.npy").exists():
            ground_truth_dict = np.load(
                le.labels_path.joinpath("ground_truth.npy"), allow_pickle=True
            ).item()
        else:
            ground_truth_dict = None

        axes.flatten()[idx], clust_dict[model] = plot_embeddings(
            ld.embed_dir,
            dim_reduction_model,
            default_labels,
            ground_truth=ground_truth_dict,
            axes=axes.flatten()[idx],
            fig=fig,
            bool_plot_centroids=False,
        )

        # metric_str = f"Silhouette Score= {clust_dict[model]['SS']:.3f}"
        # axes.flatten()[idx].set_title(f"{model.upper()}\n{metric_str}")
        axes.flatten()[idx].set_title(f"{model.upper()}")
    # [ax.remove() for ax in axes.flatten()[idx + 1 :]]
    # new_order = [
    #     k[0] for k in sorted(clust_dict.items(), key=lambda kv: kv[1]["SS"])[::-1]
    # ]
    # positions = {mod: ax.get_position() for mod, ax in zip(new_order, axes.flatten())}
    # for model, ax in zip(embedding_models, axes.flatten()):
    #     ax.set_position(positions[model])

    # set_legend(fig, axes.flatten()[0], bool_plot_centroids=False)
    fig.suptitle(f"Comparison of {dim_reduction_model} embeddings", fontweight="bold")
    fig.savefig(ld.embed_dir.joinpath("comp_fig.png"), dpi=300)


def plot_classification_results(paths, task_name, metrics):
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
    path = paths.class_path
    fig.savefig(
        path.joinpath(f"class_results_{task_name}_{model_name}.png"),
        dpi=300,
    )
    plt.close(fig)


def load_classification_results(task_name, model_list):
    per_class_metrics = {}
    overall_metrics = {}
    for model_name in model_list:
        with open(
            Path(bacpipe_settings["task_results_dir"])
            .joinpath("metrics")
            .joinpath(f"class_results_{task_name}_{model_name}.yml"),
            "r",
        ) as f:
            metrics = yaml.load(f, Loader=yaml.CLoader)
            per_class_metrics[model_name] = metrics["Per Class Metrics:"]
            overall_metrics[model_name] = metrics["Overall Metrics:"]
    return per_class_metrics, overall_metrics


def visualise_classification_results_across_models(task_name, model_list):
    per_class_metrics, overall_metrics = load_classification_results(
        task_name, model_list
    )
    plot_per_class_metrics(task_name, model_list, per_class_metrics, overall_metrics)

    plot_overview_metrics(task_name, model_list, overall_metrics)


def plot_overview_metrics(task_name, model_list, overall_metrics):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    num_metrics = len(overall_metrics[model_list[0]])
    bar_width = 1 / (num_metrics + 1)

    cmap = plt.cm.tab10
    cols = cmap(np.arange(num_metrics) % cmap.N)

    d = {m: v["Macro Accuracy"] for m, v in overall_metrics.items()}
    overall_metrics = dict(
        sorted(overall_metrics.items(), key=lambda x: d[x[0]], reverse=True)
    )
    model_list = list(overall_metrics.keys())

    for mod_idx, (model, d) in enumerate(overall_metrics.items()):
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
    ax.set_xticks(np.arange(len(model_list)) - bar_width * (num_metrics - 1) / 2)
    ax.set_xticklabels(
        [model.upper() for model in model_list],
        rotation=45,
        horizontalalignment="right",
    )
    ax.set_title(f"Overall Metrics for {task_name} Classification Across Models")

    fig.subplots_adjust(right=0.75, bottom=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Metrics",
        labels=d.keys(),
        fontsize=10,
    )

    fig.savefig(
        Path(bacpipe_settings["task_results_dir"])
        .joinpath("plots")
        .joinpath(f"overview_metrics_{task_name}_" + "-".join(model_list) + ".png"),
        dpi=300,
    )
    plt.close(fig)


def plot_per_class_metrics(task_name, model_list, per_class_metrics, overall_metrics):
    num_classes = len(per_class_metrics[model_list[0]].keys())
    fig_width = max(12, num_classes * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

    cmap = plt.cm.tab10
    model_colors = cmap(np.arange(len(model_list)) % cmap.N)

    base_markers = [
        "o",
        "s",
        "^",
        "D",
        "P",
        "X",
        "*",
        "h",
        "H",
        "<",
        ">",
        "v",
        "|",
        "_",
    ]
    extended_markers = base_markers * ((len(model_list) // len(base_markers)) + 1)

    d = {m: v["Macro Accuracy"] for m, v in overall_metrics.items()}
    model_list = sorted(d, key=d.get, reverse=True)
    all_classes = sorted(per_class_metrics[model_list[0]].keys())

    for i, model_name in enumerate(model_list):
        per_class_values = per_class_metrics[model_name].values()

        ax.scatter(
            np.arange(len(all_classes)),
            per_class_values,
            color=model_colors[i],
            label=f"{model_name.upper()} "
            + f"(accuracy: {overall_metrics[model_name]['Macro Accuracy']:.3f})",
            marker=extended_markers[i],
            s=100,
        )

        ax.plot(
            np.arange(len(all_classes)),
            per_class_values,
            color=model_colors[i],
            linestyle="-",  # Solid line
            linewidth=1.5,
        )

    fig.suptitle(
        f"Per Class Metrics for {task_name} Classification Across Models",
        fontsize=14,
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Classes")
    ax.set_xticks(np.arange(len(all_classes)))
    ax.set_xticklabels(all_classes, rotation=90)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Models", fontsize=10)

    fig.subplots_adjust(right=0.65, bottom=0.3)
    file_name = f"comparison_{task_name}_" + "-".join(model_list) + ".png"
    fig.savefig(
        Path(bacpipe_settings["task_results_dir"])
        .joinpath("plots")
        .joinpath(file_name),
        dpi=300,
    )
    plt.close(fig)


#################################################################


def plot_violins(left, right):
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


def plot_bars(clusterings, run, model, ax, outer_idx):
    metrics_list = ["SS", "AMI", "ARI"]
    bar_width = 1 / (len(metrics_list) + 1)
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(metrics_list)) % cmap.N)
    for inner_idx, (metric, color) in enumerate(zip(metrics_list, colors)):
        if metric == "SS":
            val = clusterings[run][model][metric]
            label = metric
        else:
            val = clusterings[run][model][metric]["KMeans"]
            label = metric + "(KMeans)"

        ax.bar(
            outer_idx - bar_width * inner_idx,
            val,
            label=label,
            width=bar_width,
            color=color,
        )


def plot_comparison(
    models,
    label_keys,
    reduced_embeds,
    name,
    metrics_embed,
    metrics_reduc,
    clust_embed,
    clust_reduc,
    reducer_conf,
    label_file=None,
    **kwargs,
):
    # fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    if label_file:  # TODO
        label_keys = list(label_keys) + ["Unlabeled"]
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    for ax, model in zip(axes.flatten(), models):
        for key, embed_split in reduced_embeds[model]["split"].items():
            ax.plot(
                embed_split[:, 0],
                embed_split[:, 1],
                "o",
                label=key,
                markersize=0.5,
            )
        # ss_string = f"\nSS_orig: {metrics_embed[model]['SS']:.3f} " + \
        #             f"SS_reduc: {metrics_reduc[model]['SS']:.3f}"
        ss_string = (
            f"\nSilhouette Score: {metrics_embed[model]['SS']:.3f} | "
            + f"AMI: {metrics_embed[model]['AMI']['kmeans']:.3f}"
        )
        ax.set_title(model.upper() + ss_string)

    fig.subplots_adjust(hspace=0.5)
    orig_sorted = dict(
        sorted(metrics_embed.items(), key=lambda x: x[-1]["SS"])[::-1]
    ).keys()
    # orig_sorted = dict(sorted(metrics_embed.items(),
    #                           key=lambda x: x[-1]['AMI']['kmeans'])[::-1]).keys()
    reduc_sorted = dict(
        sorted(metrics_reduc.items(), key=lambda x: x[-1]["SS"])[::-1]
    ).keys()
    positions = {
        mod: ax.get_position() for mod, ax in zip(list(orig_sorted), axes.flatten())
    }
    for model, ax in zip(list(metrics_embed.keys()), axes.flatten()):
        ax.set_position(positions[model])

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    # labels = [
    #     "Chiffchaff & Cuckoo",
    #     "Coati",
    #     "Dawn Chorus (Cuckoo, Wren & Robin)",
    #     "Chick calls",
    #     "Manx Shearwater",
    #     "Dolphin Quacks",
    # ]
    fig.legend(handles, labels, fontsize=12, markerscale=15, ncol=3, loc="lower center")
    # fig.legend(*ax.get_legend_handles_labels(),
    #            markerscale=6,
    #            loc="outside right")
    path = plot_path.joinpath(f"default_{list(reducer_conf.keys())[-1]}")
    path.mkdir(exist_ok=True)

    fig.savefig(path.joinpath(f"{name}_ss.png"))

    plot_clusterings(models, clust_embed, metrics_embed, reduced_embeds, name, **kwargs)
    plot_clusterings(
        models, clust_reduc, metrics_reduc, reduced_embeds, name + "_reduced", **kwargs
    )


def clust_bar_plot(runs):
    clusterings = {}
    for run in runs:
        with open(clust_metrics_path.joinpath(f"{run}_cluster_metrics.json"), "r") as f:
            clusterings[run] = json.load(f)
        with open(
            clust_metrics_path.joinpath(f"{run}_reduced_cluster_metrics.json"), "r"
        ) as f:
            clusterings[run + "_reduced"] = json.load(f)

    perc_clust = get_percentage_change(runs, clusterings)

    bar_width = 1 / (len(runs) * 2 - 1 + 1)
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(runs) * 2 - 1 + 3) % cmap.N)
    fig_tot, axes_tot = plt.subplots(4, 1, figsize=(10, 10))
    met_idx = 0
    run_label = []
    for met in ["SS", "AMI", "ARI"]:
        for mod_idx, model in enumerate(clusterings[run].keys()):
            # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
            run_idx = 0
            for reduc_label in ["", "_reduced"]:
                for run in runs:
                    run = run + reduc_label

                    if run == "normal":
                        if met_idx == 0:
                            for idx, orig_met in enumerate(["SS", "AMI", "ARI"]):
                                if not isinstance(
                                    clusterings[run][model][orig_met], float
                                ):
                                    axes_tot[0].bar(
                                        mod_idx - bar_width * idx,
                                        clusterings[run][model][orig_met]["kmeans"],
                                        label=orig_met,
                                        width=bar_width,
                                        color=colors[5 + idx],
                                    )
                                else:
                                    axes_tot[0].bar(
                                        mod_idx - bar_width * idx,
                                        clusterings[run][model][orig_met],
                                        label=orig_met,
                                        width=bar_width,
                                        color=colors[5 + idx],
                                    )
                        continue

                    if not isinstance(perc_clust[run][model][met], float):
                        axes_tot[met_idx + 1].bar(
                            mod_idx - bar_width * run_idx,
                            perc_clust[run][model][met]["kmeans"],
                            label=run,
                            width=bar_width,
                            color=colors[run_idx],
                        )
                    else:
                        axes_tot[met_idx + 1].bar(
                            mod_idx - bar_width * run_idx,
                            perc_clust[run][model][met],
                            label=run,
                            width=bar_width,
                            color=colors[run_idx],
                        )
                    run_label.append(run)
                    # plot_bars(perc_clust, run+reduc_label, model, ax, run_idx)
                    # axes_tot[run_idx].bar(mod_idx - bar_width * run_idx,
                    #                 perc_clust[run][model][met],
                    #                 label=run,
                    #                 width=bar_width,
                    #                 color=colors[run_idx])
                    run_idx += 1
        axes_tot[met_idx].set_xticks([])
        axes_tot[met_idx + 1].set_ylabel(met)
        axes_tot[met_idx + 1].hlines(
            0, -1, mod_idx, linestyles="dashed", color="black", linewidth=0.3
        )
        met_idx += 1

        # ax.set_title(model)
        # ax.set_xticks([])

        # axes[0].set_ylabel('original embedding')
        # axes[1].set_ylabel('reduced 2d UMAP embedding')
        # axes[-1].set_xticks(np.arange(len(runs)) - 0.3)
        # axes[-1].set_xticklabels(runs, rotation=45, ha='right')
        # hand, labl = ax.get_legend_handles_labels()
        # hand, labl = hand[:3], labl[:3]

        # fig.legend(hand, labl, loc="lower right")
        path = plot_path.joinpath(f"compare_runs").joinpath(model)
        path.mkdir(exist_ok=True, parents=True)
        # fig.savefig(path.joinpath(f'{run}_barplot.png'))

    axes_tot[0].set_ylabel("original embeddings")
    axes_tot[0 + 1].set_ylim(-50, 1400)
    axes_tot[1 + 1].set_ylim(-30, 150)
    axes_tot[2 + 1].set_ylim(-30, 150)
    fig_tot.subplots_adjust(right=0.8)
    models = list(clusterings[run].keys())
    # axes_tot[0].set_ylabel('original embedding')
    # axes_tot[1].set_ylabel('reduced 2d UMAP embedding')
    axes_tot[-1].set_xticks(np.arange(len(models)) - 0.3)
    axes_tot[-1].set_xticklabels(models, rotation=45, ha="right")

    hand0, labl0 = axes_tot[0].get_legend_handles_labels()
    hand0, labl0 = hand0[:3], labl0[:3]
    # fig_tot.legend(hand0, labl0, loc="outside right")

    hand1, labl1 = axes_tot[-1].get_legend_handles_labels()
    hand1, labl1 = hand1[:5][::-1], labl1[:5][::-1]
    hand, labl = hand0 + hand1, labl0 + labl1
    fig_tot.legend(hand, labl, loc="outside right")

    axes_tot[0].set_title("Clustering scores of original embeddings")
    axes_tot[1].set_title(
        "Percentage change of clustering scores from original embeddings"
    )
    # fig_tot.suptitle('Deviations from original values in percent')
    path = plot_path.joinpath(f"compare_runs")
    path.mkdir(exist_ok=True)
    fig_tot.savefig(path.joinpath(f"barplot.png"), dpi=300)


def plot_clusterings(
    models, clust_values, metrics, reduced_embeds, name, clust_conf, **kwargs
):
    for clust_name in list(clust_values.values())[0].keys():
        fig, axes = plt.subplots(4, 4, figsize=(20, 10))
        for ax, model in zip(axes.flatten(), models):
            clust_vals = clust_values[model][clust_name]
            for cluster in np.unique(clust_vals):
                ems = reduced_embeds[model]["all"][clust_vals == cluster]
                ax.plot(
                    ems[:, 0],
                    ems[:, 1],
                    "o",
                    markersize=0.5,
                )

            metric_string = (
                f"\nARI: {metrics[model]['ARI'][clust_name]:.3f} "
                + f"AMI: {metrics[model]['AMI'][clust_name]:.3f}"
            )
            ax.set_title(model.upper() + metric_string)

        conf_label = [
            list(clust.keys())[-1]
            for clust in clust_conf
            if clust["name"] == clust_name.lower()
        ][0]
        if "_reduced" in name:
            path = plot_path.joinpath(f"reduced_{clust_name}_{conf_label}")
        else:
            path = plot_path.joinpath(f"{clust_name}_{conf_label}")

        path.mkdir(exist_ok=True)

        fig.savefig(path.joinpath(f"{name}_{clust_name}.png"))


def plot_clustering_by_metric_new(all_clusts_reordered, models):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    plt.subplots_adjust(right=0.9)

    sort_dict = {
        m: v["normal_no_noise"] for m, v in all_clusts_reordered["AMI"].items()
    }
    sort_dict = dict(sorted(sort_dict.items(), key=lambda x: x[-1], reverse=True))
    models = list(sort_dict.keys())

    cmap = plt.cm.tab20
    colors = cmap(np.arange(len(models)) % cmap.N)
    for axes_row, metric in zip(axes, ["AMI", "ARI"]):
        for axes, met in zip(axes_row, ["normal", "pca", "umap"]):
            # for axes, met in zip(axes_row, ["normal", "pca", "spca", "umap", "umap_bars"]):
            for idx, model in enumerate(sort_dict.keys()):
                # order = [
                #     "normal",
                #     "pca_50",
                #     "pca_100",
                #     "spca_50",
                #     "spca_100",
                #     "umap_50",
                #     "umap_100",
                # ]
                if met == "normal":
                    values = all_clusts_reordered[metric][model][met + "_no_noise"]
                    axes.bar(
                        idx / len(models),
                        values,
                        width=1 / len(models),
                        color=colors[idx],
                        label=model,
                    )
                    if metric == "AMI":
                        axes.set_title("Orig. Embedding")
                        axes.set_xticks([])
                    else:
                        axes.set_xticks([])
                    axes.set_ylabel(metric)
                # elif met == 'umap_bars':
                else:
                    # values = all_clusts_reordered[metric][model]['umap_100_no_noise']
                    values = all_clusts_reordered[metric][model][met + "_300_no_noise"]
                    axes.bar(
                        idx / len(models),
                        values,
                        width=1 / len(models),
                        color=colors[idx],
                        label=model,
                    )
                    if metric == "AMI":
                        axes.set_title(met.upper() + " 300")
                        axes.set_xticks([])
                    else:
                        axes.set_xticks([])
                # else:
                #     # values = [all_clusts_reordered[metric][model][met+ii+'_no_noise'] for ii in ["_50", "_100"]]
                #     values = [all_clusts_reordered[metric][model][met+'_'+ii+'_no_noise'] for ii in ["300"]]
                #     values = np.array(values)/all_clusts_reordered[metric][model]['normal_no_noise']
                #     axes.plot(values, "-x", color=colors[idx], label=model)
                #     axes.set_ylim([0.45, 2.5])
                #     axes.hlines(1, 0, 1, linestyles="dashed", color="black", linewidth=1.5)
                #     if metric == 'AMI':
                #         axes.set_title(met.upper() + ' perc. change')
                #         axes.set_xticks([])
                #     else:
                #         axes.set_xticks(np.arange(1))
                #         # axes.set_xticklabels([met+'_50', met+'_100'], rotation=45, ha="right")
                #         axes.set_xticklabels([met+'_300'], rotation=45, ha="right")

    plt.show()
    fig.legend(models, loc="outside right")
    fig.savefig(plot_path.joinpath(f"clustering_by_metriccc.png"), dpi=300)


def plot_clustering_by_metric(all_clusts_reordered, models):
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    plt.subplots_adjust(right=0.8)

    cmap = plt.cm.tab20
    colors = cmap(np.arange(len(models)) % cmap.N)[::-1]
    for axes, metric in zip(axes.flatten(), ["SS", "AMI", "ARI"]):
        for idx, model in enumerate(models):
            order = [
                "normal",
                "pca_50",
                "pca_100",
                "spca_50",
                "spca_100",
                "umap_50",
                "umap_100",
            ]
            values = [
                all_clusts_reordered[metric][model][run + "_no_noise"] for run in order
            ]
            # values = np.array(values)/values[0]
            axes.plot(values, "-x", color=colors[idx], label=model)
        axes.set_ylabel(metric)
        axes.set_xticks(np.arange(len(order)))
        axes.set_xticklabels(order, rotation=45, ha="right")
    plt.show()
    fig.legend(models, loc="outside right")
    fig.savefig(plot_path.joinpath(f"clustering_by_metric.png"), dpi=300)
