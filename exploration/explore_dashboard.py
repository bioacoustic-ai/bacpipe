import panel as pn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import copy

matplotlib.use("agg")

from exploration.explore_embeds import set_paths

data_path = "colombia_soundscape"
set_paths(data_path)
from exploration.explore_embeds import (
    get_original_embeds,
    reduce_dimensions,
    clustering,
    main_embeds_path,
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
    for d in list(np_embeds_path.rglob("*.npy"))
    if d.stem not in ["distances", "embed_dict"]
]
METRICS = ["SS", "AMI_hdbscan", "AMI_kmeans", "ARI_hdbscan", "ARI_kmeans"]
# freq_slider = pn.widgets.IntSlider(name="Frequency", start=1, end=10, value=5)

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


def plot_overview(orig_embeds, reducer, **kwargs):
    fig, ax = plt.subplots(4, 4, figsize=(12, 8))
    fig.subplots_adjust(
        right=0.8,
    )
    embeds = reduce_dimensions(
        orig_embeds, name=reducer, reducer_2d_conf=conf_2d_reduction[0]
    )
    for idx, model in enumerate(embeds.keys()):
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
    if kwargs["label_by"] == "hdbscan":
        num_labels = []
        for a in ax.flatten():
            hand, labl = a.get_legend_handles_labels()
            num_labels.append(len(labl))
        hand, labl = ax.flatten()[
            num_labels.index(max(num_labels))
        ].get_legend_handles_labels()
    else:
        hand, labl = ax[idx // 4, idx % 4].get_legend_handles_labels()
    fig.legend(hand, labl, fontsize=8, markerscale=15, loc="outside right")
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
    label_by,
    fig=None,
    ax=None,
    no_legend=False,
    no_noise=False,
    **kwargs,
):
    embeds = reduce_dimensions(
        orig_embeds, name=reducer, reducer_2d_conf=conf_2d_reduction[0]
    )

    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.subplots_adjust(right=0.6)
        ax.set_title(f"Embeddings of {model}", fontsize=8)
    else:
        ax.set_title(model.upper(), fontsize=8)
        no_legend = True

    if label_by in ["hdbscan", "kmeans"]:
        clusterings = np.load(
            np_clust_path.joinpath(f"{reducer}_clusterings.npy"), allow_pickle=True
        ).item()
        clusterings = clusterings[model][label_by]
        label_keys = list(set(clusterings))
        data = {}
        for label in label_keys:
            data[label] = embeds[model]["all"][clusterings == label]
    elif label_by == "ground truth":
        data = embeds[model]["split"]

    cmap = plt.cm.tab20
    colors = cmap(np.arange(len(data.keys())) % cmap.N)

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
            markersize=0.5,
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

model2 = copy.deepcopy(model1)
reducer2 = copy.deepcopy(reducer1)
label_by2 = copy.deepcopy(label_by1)
percentages2 = copy.deepcopy(percentages1)
no_noise2 = copy.deepcopy(no_noise1)

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
    plot_clusterings, model=model1, plot_percentages=percentages1, no_noise=no_noise1
)
Clusterings2 = pn.bind(
    plot_clusterings,
    model=model2,
    plot_percentages=percentages2,
    no_noise=no_noise2,
    no_legend=True,
)


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
        ),
    ),
)

# Serve the dashboard
dashboard.servable()
