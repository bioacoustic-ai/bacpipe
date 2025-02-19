import numpy as np
import sklearn.decomposition
import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt
import umap
import sklearn
from sklearn.metrics import silhouette_score as SS
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI
import hdbscan
from sklearn.cluster import KMeans
from tqdm import tqdm
from types import SimpleNamespace
from matplotlib import cm

# data_path = "id_task_data"
data_path = "Evaluation_set_5shots"


main_embeds_path = Path(f"exploration/{data_path}/embeddings")
np_embeds_path = Path(f"exploration/{data_path}/numpy_embeddings")
np_clust_path = Path(f"exploration/{data_path}/numpy_clusterings")
clust_metrics_path = Path(f"exploration/{data_path}/cluster_metrics")
plot_path = Path(f"exploration/{data_path}/plots")

main_embeds_path.mkdir(exist_ok=True)
np_embeds_path.mkdir(exist_ok=True)
np_clust_path.mkdir(exist_ok=True)
clust_metrics_path.mkdir(exist_ok=True)
plot_path.mkdir(exist_ok=True)


def get_original_embeds(models):
    if not np_embeds_path.joinpath("embed_dict.npy").exists():

        paths = [
            [d for d in list(main_embeds_path.rglob(f"*{model}*"))[::-1] if d.is_dir()][
                0
            ]
            for model in models
        ]

        embed_dict = {}
        for model, path in tqdm(
            zip(models, paths), desc="loading models", position=0, leave=False
        ):
            files = list(path.rglob("*.npy"))
            files.sort()

            embed_dict[model] = {"label_dict": {}, "all": []}
            num_embeds = 0
            for ind, file in tqdm(
                enumerate(files), desc="loading embeddings", position=1, leave=True
            ):
                embed = np.load(file)
                if ind == 0:
                    e = embed
                else:
                    e = np.concatenate((e, embed))
                if not file.parent.stem in embed_dict[model]["label_dict"].keys():
                    embed_dict[model]["label_dict"][file.parent.stem] = num_embeds
                num_embeds += embed.shape[0]

            embed_dict[model]["all"] = e

            embed_dict[model]["split"] = np.split(
                embed_dict[model]["all"],
                list(embed_dict[model]["label_dict"].values())[1:],
            )

            embed_dict[model]["labels"] = np.concatenate(
                [
                    np.ones(len(data)) * i
                    for i, data in enumerate(embed_dict[model]["split"])
                ]
            )
        np.save(np_embeds_path.joinpath("embed_dict.npy"), embed_dict)
    else:
        embed_dict = np.load(
            np_embeds_path.joinpath("embed_dict.npy"), allow_pickle=True
        ).item()
    return embed_dict


def reduce_dimensions(embeds, name, reducer_conf=None, **kwargs):
    if not np_embeds_path.joinpath(f"{name}.npy").exists():
        reduc_embeds = {}
        for model, embed in tqdm(
            embeds.items(),
            desc="calculating dimensionality reduction",
            position=0,
            leave=False,
        ):
            reduc_embeds[model] = {}
            if reducer_conf["name"] == "umap":
                reducer = umap.UMAP(**list(reducer_conf.values())[-1])
            else:
                assert False, "Reducer not implemented"

            reduc_embeds[model]["all"] = reducer.fit_transform(embed["all"])

            reduc_embeds[model]["split"] = np.split(
                reduc_embeds[model]["all"], list(embed["label_dict"].values())[1:]
            )

            reduc_embeds[model]["labels"] = np.concatenate(
                [
                    np.ones(len(data)) * i
                    for i, data in enumerate(reduc_embeds[model]["split"])
                ]
            )

        np.save(np_embeds_path.joinpath(f"{name}.npy"), reduc_embeds)
    else:
        reduc_embeds = np.load(
            np_embeds_path.joinpath(f"{name}.npy"), allow_pickle=True
        ).item()
    return reduc_embeds


def convert_numpy_types(obj):
    if isinstance(obj, np.float32):  # Convert np.float32 to Python float
        return float(obj)
    if isinstance(obj, np.ndarray):  # Convert NumPy array to list
        return obj.tolist()
    return obj  # Return unchanged for other types


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
    **kwargs,
):
    # fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    for ax, model in zip(axes.flatten(), models):
        for key, embed_split in zip(label_keys, reduced_embeds[model]["split"]):
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
    labels = [
        "Chiffchaff & Cuckoo",
        "Coati",
        "Dawn Chorus (Cuckoo, Wren & Robin)",
        "Chick calls",
        "Manx Shearwater",
        "Dolphin Quacks",
    ]
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


def plot_clusterings(
    models, clust_values, metrics, reduced_embeds, name, clust_conf, **kwargs
):
    for clust_name in list(clust_values.values())[0].keys():
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
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


def clustering(embeds, name, clust_conf=None, **kwargs):
    if clust_metrics_path.joinpath(f"{name}_cluster_metrics.json").exists():
        metrics = {}
        clusterings = {}
        for model in tqdm(
            embeds.keys(),
            desc="calculating clustering metrics",
            position=0,
            leave=False,
        ):
            labels = embeds[model]["labels"]
            metrics[model] = {}
            clusterings[model] = {}
            metrics[model]["SS"] = SS(embeds[model]["all"], labels)

            for clust in clust_conf:
                if clust["name"] == "kmeans":
                    clusterer = KMeans(**list(clust.values())[-1])
                elif clust["name"] == "hdbscan":
                    clusterer = hdbscan.HDBSCAN(**list(clust.values())[-1])
                clusterings[model][clust["name"]] = clusterer.fit_predict(
                    embeds[model]["all"]
                )
            metrics[model]["AMI"] = {}
            metrics[model]["ARI"] = {}
            for clust_name, clustering in clusterings[model].items():
                metrics[model]["AMI"][clust_name] = AMI(labels, clustering)
                metrics[model]["ARI"][clust_name] = ARI(labels, clustering)

        np.save(np_clust_path.joinpath(f"{name}_clusterings.npy"), clusterings)

        with open(
            clust_metrics_path.joinpath(f"{name}_cluster_metrics.json"), "w"
        ) as f:
            json.dump(metrics, f, default=convert_numpy_types)

    else:
        clusterings = np.load(
            np_clust_path.joinpath(f"{name}_clusterings.npy"), allow_pickle=True
        ).item()
        with open(
            clust_metrics_path.joinpath(f"{name}_cluster_metrics.json"), "r"
        ) as f:
            metrics = json.load(f)
    return metrics, clusterings


def compare(embeds, name, **kwargs):

    reduc_embeds = reduce_dimensions(embeds, name, **kwargs)

    metrics_embed, clust_embed = clustering(embeds, name, **kwargs)
    metrics_reduc, clust_reduc = clustering(reduc_embeds, name + "_reduced", **kwargs)

    plot_comparison(
        embeds.keys(),
        list(embeds.values())[0]["label_dict"].keys(),
        reduc_embeds,
        name,
        metrics_embed,
        metrics_reduc,
        clust_embed,
        clust_reduc,
        **kwargs,
    )


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


def get_percentage_change(runs, clusterings):
    clust_percentages = {}
    for run in runs:
        for reduc_label in ["", "_reduced"]:
            run = run + reduc_label
            if run == "normal":
                continue
            clust_percentages[run] = {}
            for model, metrics in clusterings[run].items():
                clust_percentages[run][model] = {}
                for metric, vals in metrics.items():
                    clust_percentages[run][model][metric] = {}
                    if isinstance(vals, float):
                        clust_percentages[run][model][metric] = (
                            vals / clusterings["normal"][model][metric] - 1
                        ) * 100
                    else:
                        for clust, val in vals.items():
                            clust_percentages[run][model][metric][clust] = (
                                val / clusterings["normal"][model][metric][clust] - 1
                            ) * 100
    return clust_percentages


def old_clust_bar_plot(runs):
    clusterings = {}
    for run in runs:
        with open(clust_metrics_path.joinpath(f"{run}_cluster_metrics.json"), "r") as f:
            clusterings[run] = json.load(f)
        with open(
            clust_metrics_path.joinpath(f"{run}_reduced_cluster_metrics.json"), "r"
        ) as f:
            clusterings[run + "_reduced"] = json.load(f)

    perc_clust = get_percentage_change(runs, clusterings)

    bar_width = 1 / (len(runs) + 1)
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(runs)) % cmap.N)
    fig_tot, axes_tot = plt.subplots(2, 1, figsize=(10, 10))
    for mod_idx, model in enumerate(clusterings[run].keys()):
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        for reduc_label, ax in zip(["", "_reduced"], axes):
            for run_idx, run in enumerate(runs):
                if run == "normal":
                    continue
                plot_bars(perc_clust, run, model, ax, reduc_label, run_idx)

                if reduc_label == "":
                    axes_tot[0].bar(
                        mod_idx - bar_width * run_idx,
                        clusterings[run][model]["SS"],
                        label=run,
                        width=bar_width,
                        color=colors[run_idx],
                    )
                    axes_tot[1].bar(
                        mod_idx - bar_width * run_idx,
                        clusterings[run + "_reduced"][model]["SS"],
                        label=run,
                        width=bar_width,
                        color=colors[run_idx],
                    )

            ax.set_title(model)
            ax.set_xticks([])
        axes[0].set_ylabel("original embedding")
        axes[1].set_ylabel("reduced 2d UMAP embedding")
        axes[-1].set_xticks(np.arange(len(runs)) - 0.3)
        axes[-1].set_xticklabels(runs, rotation=45, ha="right")
        hand, labl = ax.get_legend_handles_labels()
        hand, labl = hand[:3], labl[:3]

        fig.legend(hand, labl, loc="lower right")
        path = plot_path.joinpath(f"compare_runs").joinpath(model)
        path.mkdir(exist_ok=True, parents=True)
        fig.savefig(path.joinpath(f"{run}_barplot.png"))

    models = list(clusterings[run].keys())
    axes_tot[0].set_ylabel("original embedding")
    axes_tot[1].set_ylabel("reduced 2d UMAP embedding")
    axes_tot[-1].set_xticks(np.arange(len(models)) - 0.3)
    axes_tot[-1].set_xticklabels(models, rotation=45, ha="right")
    hand, labl = axes_tot[-1].get_legend_handles_labels()
    hand, labl = hand[:5], labl[:5]

    fig_tot.legend(hand, labl, loc="lower right")
    fig_tot.suptitle("Comparison of different runs by Silhouette Scores")
    path = plot_path.joinpath(f"compare_runs")
    path.mkdir(exist_ok=True)
    fig_tot.savefig(path.joinpath(f"barplot.png"))


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
