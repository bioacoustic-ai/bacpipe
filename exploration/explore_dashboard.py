import panel as pn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import copy
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

matplotlib.use("agg")

from exploration.explore_embeds import set_paths

# data_path = "Evaluation_set_5shots"
data_path = "neotropic_dawn_chorus"
# data_path = "id_task_data"
# data_path = "colombia_soundscape"
set_paths(data_path)
from exploration.explore_embeds import (
    get_original_embeds,
    reduce_to_2d,
    clustering,
    main_embeds_path,
    distances_path,
    clust_metrics_path,
    np_embeds_path,
    np_clust_path,
)

# Enable Panel
pn.extension()

# Create a shared frequency slider

MODELS = [
    d.stem.split("___")[-1].split("-")[0]
    for d in list(main_embeds_path.rglob("*"))
    if d.is_dir()
]
REDUCERS = [
    d.stem
    for d in list(np_embeds_path.glob("*.npy"))
    if d.stem not in ["distances", "embed_dict", "normal_distances"]
]
METRICS = ["SS", "AMI_hdbscan", "AMI_kmeans", "ARI_hdbscan", "ARI_kmeans"]

ALL_DISTS = {}
for file in distances_path.glob("*.npy"):
    ALL_DISTS[file.stem.split("_distances")[0]] = np.load(
        file, allow_pickle=True
    ).item()


conf_2d_reduction = [
    {
        "name": "2dumap",
        "conf_1": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
        },
    }
]


def load_distances(reducer=None, model=None, label=None, metric="euclidean"):
    if reducer is not None:
        distances = np.load(
            distances_path.joinpath(f"{reducer}_distances.npy"), allow_pickle=True
        ).item()
    else:
        distances = ALL_DISTS
    if reducer is None:
        distances_left = {
            k: v[model][metric][label]["intra"] for k, v in distances.items()
        }
        distances_right = {
            k: v[model][metric][label]["inter"] for k, v in distances.items()
        }
    elif model is not None:
        distances_left = {k: v["intra"] for k, v in distances[model][metric].items()}
        distances_right = {k: v["inter"] for k, v in distances[model][metric].items()}
    elif label is not None:
        distances_left = {
            k: v[metric][label]["intra"]
            for k, v in distances.items()
            if not k == "rcl_fs_bsed"
        }
        distances_right = {
            k: v[metric][label]["inter"]
            for k, v in distances.items()
            if not k == "rcl_fs_bsed"
        }

    return plot_violins(distances_left, distances_right)


def plot_violins(left, right):
    val = []
    typ = []
    cat = []
    for idx, label in enumerate(dict(sorted(left.items())).keys()):
        val.append(left[label])
        val.append(right[label])
        typ.extend(["Intra"] * len(left[label]))
        typ.extend(["Inter"] * len(right[label]))
        cat.extend([label] * len(left[label]))
        cat.extend([label] * len(right[label]))

    # Convert to long-form format
    data_long = pd.DataFrame(
        {"Value": np.concatenate(val), "Type": typ, "Category": cat}
    )

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(
        x="Category",
        y="Value",
        hue="Type",
        data=data_long,
        split=True,
        inner="quartile",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return fig


def plot_overview(orig_embeds, reducer, **kwargs):
    fig, ax = plt.subplots(4, 4, figsize=(10, 7))
    fig.subplots_adjust(
        # right=0.8,
        bottom=0.1,
    )
    embeds = reduce_to_2d(
        orig_embeds, name=reducer, reducer_2d_conf=conf_2d_reduction[0]
    )

    if True:
        met_clust = np.load(
            clust_metrics_path.joinpath("all_clusts_reordered.npy"), allow_pickle=True
        ).item()
        amis = {k: v["normal_no_noise"] for k, v in met_clust["AMI"].items()}
        new_order = dict(sorted(amis.items(), key=lambda item: item[1], reverse=True))

    for idx, model in enumerate(new_order.keys()):
        label_keys = list(orig_embeds[model]["label_dict"].keys())
        plot_embeds(
            embeds,
            model,
            reducer,
            label_keys,
            fig=fig,
            ax=ax[idx // 4, idx % 4],
            **kwargs,
        )
    ax.flatten()[-1].axis("off")
    if "label_by" in kwargs and kwargs["label_by"] == "hdbscan":
        num_labels = []
        for a in ax.flatten():
            hand, labl = a.get_legend_handles_labels()
            num_labels.append(len(labl))
        hand, labl = ax.flatten()[
            num_labels.index(max(num_labels))
        ].get_legend_handles_labels()
    else:
        hand, labl = ax[idx // 4, idx % 4].get_legend_handles_labels()
    # fig.legend(hand, labl, fontsize=12, markerscale=15, loc="outside right")
    fig.legend(hand, labl, fontsize=9.5, markerscale=30, loc="lower center", ncol=4)
    for axes, model in zip(ax.flatten(), new_order.keys()):
        axes.set_title(f"{model.upper()} ({amis[model]:.2f}) ", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    return fig


def plot_clust_overview(
    select_clustering, label_by, no_noise=False, plot_percentages=False, **kwargs
):
    metrics = {}
    for reducer in REDUCERS:
        if no_noise:
            reducer = f"{reducer}_no_noise"
        with open(
            clust_metrics_path.joinpath(f"{reducer}_cluster_metrics.json"), "r"
        ) as f:
            metrics[reducer] = flatten_metric_dict(json.load(f))

    metrics_by_clustering = {}
    for model in metrics[reducer].keys():
        metrics_by_clustering[model] = {}
        for reducer, metric in metrics.items():
            metrics_by_clustering[model][reducer] = metric[model][select_clustering]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.8)

    if plot_percentages:
        perc_metrics = get_percentage_change(metrics_by_clustering, no_noise=no_noise)
        return plot_bars_dicka(
            perc_metrics, fig, ax, y_label="Percentage change", **kwargs
        )
    else:
        return plot_bars_dicka(metrics_by_clustering, fig, ax, **kwargs)


def plot_embeds(
    orig_embeds,
    model,
    reducer,
    label_keys,
    label_by="ground truth",
    fig=None,
    ax=None,
    no_legend=False,
    no_noise=False,
    **kwargs,
):
    embeds = reduce_to_2d(
        orig_embeds, name=reducer, reducer_2d_conf=conf_2d_reduction[0]
    )

    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.subplots_adjust(right=0.6)
        ax.set_title(f"Embeddings of {model}", fontsize=8)
    else:
        ax.set_title(model.upper(), fontsize=12)
        no_legend = True

    if label_by in ["hdbscan", "kmeans"]:
        clusterings = np.load(
            np_clust_path.joinpath(f"{reducer}_no_noise_clusterings.npy"),
            allow_pickle=True,
        ).item()
        clusterings = clusterings[model][label_by]
        label_keys = list(set(clusterings))
        data = {}
        for label in label_keys:
            data[label] = embeds[model]["all"][clusterings == label]
    elif label_by == "ground truth":
        data = embeds[model]["split"]

    cmap = plt.cm.tab20
    colors = cmap(np.arange(len(data.keys())) % cmap.N)[::-1]

    if no_noise:
        label_keys = [key for key in label_keys if key != "unknown" and key != -1]
    elif "unknown" in label_keys:
        # reorder dict so that unknown is first
        label_keys = ["unknown"] + [key for key in label_keys if key != "unknown"]
    elif -1 in label_keys:
        label_keys = [-1] + [key for key in label_keys if key != -1]

    for idx, label in enumerate(label_keys):
        ax.plot(
            data[label][:, 0],
            data[label][:, 1],
            "o",
            label=label,
            markersize=0.2,
            color=colors[idx],
        )
        ax.set_xticks([])
        ax.set_yticks([])
    if not no_legend:
        fig.legend(fontsize=8, markerscale=15, loc="outside right")
    return fig


def get_percentage_change(metrics, no_noise=False):
    if not no_noise:
        relative_to = "normal"
    else:
        relative_to = "normal_no_noise"
    clust_percentages = {}
    for reducer, metric in [
        metric for metric in metrics.items() if not metric[0] == relative_to
    ]:
        clust_percentages[reducer] = {}
        for key, value in metric.items():
            if relative_to in metrics.keys():
                denominator = metrics[relative_to][key]
            elif relative_to in metrics[reducer].keys():
                denominator = metrics[reducer][relative_to]

            try:
                clust_percentages[reducer][key] = (value / denominator - 1) * 100
            except ZeroDivisionError:
                clust_percentages[reducer][key] = 0
    return clust_percentages


def plot_bars_dicka(
    metrics, fig, ax, y_label="Metric value", no_legend=False, **kwargs
):
    bar_width = 1 / (len(list(metrics.values())[0].keys()) + 1)
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(list(metrics.values())[0].keys())) % cmap.N)
    metrics_sorted = dict(sorted(metrics.items()))

    for out_idx, (reducer, metric) in enumerate(metrics_sorted.items()):
        for inner_idx, (key, value) in enumerate(metric.items()):
            ax.bar(
                out_idx - bar_width * inner_idx,
                value,
                label=key,
                width=bar_width,
                color=colors[inner_idx],
            )

    ax.set_xticks(np.arange(len(metrics_sorted.keys())) - 0.33)
    ax.set_xticklabels(list(metrics_sorted.keys()), rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.hlines(0, -1, out_idx, linestyles="dashed", color="black", linewidth=0.3)
    hand, labl = ax.get_legend_handles_labels()
    if not no_legend:
        fig.legend(
            hand[: inner_idx + 1],
            labl[: inner_idx + 1],
            fontsize=10,
            markerscale=15,
            loc="outside right",
        )
    return fig


def flatten_metric_dict(metrics):
    # flatten
    flat_dict = {}

    for key_red, met in metrics.items():
        flat_dict[key_red] = {}
        for key_met, value in met.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_dict[key_red][f"{key_met}_{sub_key}"] = sub_value
            else:
                flat_dict[key_red][key_met] = value
    return flat_dict


def plot_clusterings(
    model, fig=None, ax=None, plot_percentages=False, no_noise=False, **kwargs
):
    metrics = {}
    for reducer in REDUCERS:
        if no_noise:
            reducer = f"{reducer}_no_noise"
        with open(
            clust_metrics_path.joinpath(f"{reducer}_cluster_metrics.json"), "r"
        ) as f:
            metrics[reducer] = json.load(f)[model]

    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.subplots_adjust(right=0.7)

    flat_dict = flatten_metric_dict(metrics)

    if plot_percentages:
        perc_metrics = get_percentage_change(flat_dict, no_noise=no_noise)
        return plot_bars_dicka(
            perc_metrics, fig, ax, y_label="Percentage change", **kwargs
        )
    else:
        return plot_bars_dicka(flat_dict, fig, ax, **kwargs)


if False:
    embed_dict = get_original_embeds()

    ############## LAYOUT ##############

    model1 = pn.widgets.Select(name="Model", options=MODELS, width=120)
    reducer1 = pn.widgets.Select(name="Reducer", options=REDUCERS, width=120)
    label_by1 = pn.widgets.Select(
        name="label_by", options=["ground truth", "hdbscan", "kmeans"], width=120
    )
    percentages1 = pn.widgets.RadioBoxGroup(
        name="Show percentages", options=[True, False], value=False
    )
    no_noise1 = pn.widgets.Checkbox(name="Remove noise", value=False)
    select_clustering = pn.widgets.Select(name="Reducer", options=METRICS, width=120)
    label1 = pn.widgets.Select(
        name="Label",
        options=list(embed_dict[model1.value]["label_dict"].keys()),
        width=120,
    )

    model2 = copy.deepcopy(model1)
    reducer2 = copy.deepcopy(reducer1)
    label_by2 = copy.deepcopy(label_by1)
    percentages2 = copy.deepcopy(percentages1)
    no_noise2 = copy.deepcopy(no_noise1)

    reducer3 = pn.widgets.Select(name="Reducer", options=[None] + REDUCERS, width=120)
    label3 = pn.widgets.Select(
        name="Label",
        options=list(embed_dict[model1.value]["label_dict"].keys()),
        width=120,
    )
    metric1 = pn.widgets.Select(
        name="Metric", options=["euclidean", "cosine"], width=120
    )

    # Bind the slider to both plots
    Overview = pn.bind(
        plot_overview,
        orig_embeds=embed_dict,
        reducer=reducer1,
        label_by=label_by1,
        no_noise=no_noise1,
    )

    Clusterings_overview = pn.bind(
        plot_clust_overview,
        select_clustering=select_clustering,
        label_by=label_by1,
        no_noise=no_noise1,
        plot_percentages=percentages1,
    )

    Embeddings1 = pn.bind(
        plot_embeds,
        orig_embeds=embed_dict,
        reducer=reducer1,
        model=model1,
        label_keys=list(embed_dict.values())[0]["label_dict"].keys(),
        label_by=label_by1,
        no_noise=no_noise1,
    )
    Embeddings2 = pn.bind(
        plot_embeds,
        orig_embeds=embed_dict,
        reducer=reducer2,
        model=model2,
        label_keys=list(embed_dict.values())[0]["label_dict"].keys(),
        label_by=label_by2,
        no_noise=no_noise2,
        no_legend=True,
    )

    Clusterings1 = pn.bind(
        plot_clusterings,
        model=model1,
        plot_percentages=percentages1,
        no_noise=no_noise1,
    )
    Clusterings2 = pn.bind(
        plot_clusterings,
        model=model2,
        plot_percentages=percentages2,
        no_noise=no_noise2,
        no_legend=True,
    )

    Violins = pn.bind(
        load_distances, reducer=reducer3, model=model1, label=label3, metric=metric1
    )

    Violins2 = pn.bind(load_distances, reducer=reducer1, label=label1)

    # Layout: Dashboard with tabs
    dashboard = pn.Tabs(
        (
            "Single model",
            pn.Column(
                pn.Row(
                    model1,
                    reducer1,
                    label_by1,
                    no_noise1,
                ),
                pn.panel(Embeddings1, tight=True),
                pn.Row(
                    "Display percentage relativ to original embeddngs?",
                    percentages1,
                ),
                pn.panel(Clusterings1, tight=True),
                pn.Row(label3, reducer3, metric1),
                pn.panel(Violins, tight=True),
            ),
        ),
        (
            "Two models",
            pn.Row(
                pn.Column(
                    pn.Row(
                        model2,
                        reducer2,
                        label_by2,
                        no_noise2,
                    ),
                    pn.panel(Embeddings2, tight=True),
                    pn.Row(
                        "Display percentage relativ to original embeddngs?",
                        percentages2,
                    ),
                    pn.panel(Clusterings2, tight=True),
                ),
                pn.Column(
                    pn.Row(
                        model1,
                        reducer1,
                        label_by1,
                        no_noise1,
                    ),
                    pn.panel(Embeddings1, tight=True),
                    pn.Row(
                        "Display percentage relativ to original embeddngs?",
                        percentages1,
                    ),
                    pn.panel(Clusterings1, tight=True),
                ),
            ),
        ),
        (
            "All models",
            pn.Column(
                pn.Row(
                    reducer1,
                    label_by1,
                    no_noise1,
                ),
                pn.panel(Overview, tight=True),
                pn.Row(
                    "Display percentage relativ to original embeddngs?",
                    percentages1,
                    select_clustering,
                ),
                pn.panel(Clusterings_overview, tight=True),
                label1,
                pn.panel(Violins2, tight=True),
            ),
        ),
    )

    # Serve the dashboard
    dashboard.servable()
