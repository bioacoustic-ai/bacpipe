import yaml
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import umap
import hdbscan

from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score as SS
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

import seaborn as sns
import matplotlib.pyplot as plt


def set_paths(data_path):
    global main_embeds_path
    global np_embeds_path
    global np_clust_path
    global clust_metrics_path
    global plot_path

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


def get_models_in_dir():
    return [
        d.stem.split("___")[-1].split("-")[0]
        for d in list(main_embeds_path.rglob("*"))
        if d.is_dir()
    ]


def ensure_model_paths_match_models(models):
    return [
        [
            d
            for d in main_embeds_path.iterdir()
            if d.is_dir() and d.stem.split("___")[-1].split("-")[0] == model
        ][0]
        for model in models
    ]


def load_labels_and_build_dict(label_file):
    label_df = pd.read_csv(label_file)
    label_idx_dict = {label: idx for idx, label in enumerate(label_df.label.unique())}
    with open(np_embeds_path.joinpath("label_idx_dict.json"), "w") as f:
        json.dump(label_idx_dict, f)
    return label_df, label_idx_dict


def concat_embeddings(ind, cumulative_embeddings, file_embeddings):
    if ind == 0:
        cumulative_embeddings = file_embeddings
    else:
        cumulative_embeddings = np.concatenate((cumulative_embeddings, file_embeddings))
    return cumulative_embeddings


def apply_labels_to_embeddings(df, label_idx_dict, embed, segment_s, audio_file):
    file_labels = np.ones(embed.shape[0]) * -1
    embed_timestamps = np.arange(embed.shape[0]) * segment_s

    assert (
        df.end.values[-1] < embed_timestamps[-1] + segment_s
    ), f"Timestamps do not match for {audio_file}"

    for _, row in df.iterrows():
        em_start = np.argmin(np.abs(embed_timestamps - row.start))
        em_end = np.argmin(np.abs(embed_timestamps - row.end))
        file_labels[em_start:em_end] = label_idx_dict[row.label]
    return file_labels


def build_split_array_by_labels(
    ind,
    file,
    embed,
    model,
    embed_dict,
    num_embeds,
    segment_s,
    metadata,
    all_labels,
    label_file,
    label_df=None,
    label_idx_dict=None,
):
    if label_file:
        audio_file = metadata["files"]["audio_files"][ind]
        df = label_df[label_df.Filename == audio_file]
        file_labels = apply_labels_to_embeddings(
            df, label_idx_dict, embed, segment_s, audio_file
        )

        all_labels = np.concatenate((all_labels, file_labels))
        embed_dict[model]["label_dict"] = label_idx_dict

    else:  # if not file then the split is done by the parent folder name
        if not file.parent.stem in embed_dict[model]["label_dict"].keys():
            embed_dict[model]["label_dict"][file.parent.stem] = num_embeds
    return embed_dict, all_labels


def concat_embeddings_and_split_by_label(
    files, model, embed_dict, segment_s, metadata, label_file, label_df, label_idx_dict
):
    num_embeds = 0
    embeddings, all_labels = np.array([]), np.array([])

    for ind, file in tqdm(
        enumerate(files),
        desc="Loading embeddings and split by labels",
        position=1,
        leave=True,
    ):
        file_embeds = np.load(file)
        embed_dict, all_labels = build_split_array_by_labels(
            ind,
            file,
            file_embeds,
            model,
            embed_dict,
            num_embeds,
            segment_s,
            metadata,
            all_labels,
            label_file,
            label_df,
            label_idx_dict,
        )
        embeddings = concat_embeddings(ind, embeddings, file_embeds)

        num_embeds += file_embeds.shape[0]
    return embeddings, all_labels, embed_dict


def finalize_split_arrays(label_file, embed_dict, model, all_labels, label_idx_dict):
    if label_file:
        embed_dict[model]["split"] = {
            k: embed_dict[model]["all"][all_labels == v]
            for k, v in label_idx_dict.items()
        }
        embed_dict[model]["split"].update(
            {"unknown": embed_dict[model]["all"][all_labels == -1]}
        )
        embed_dict[model]["label_dict"].update({"unknown": -1})
        embed_dict[model]["labels"] = all_labels

    else:  # if not file then the split is done by the parent folder name
        embed_dict[model]["split"] = {  # TODO check that this works
            k: v
            for k, v in zip(
                embed_dict[model]["label_dict"].keys(),
                np.split(
                    embed_dict[model]["all"],
                    list(embed_dict[model]["label_dict"].values())[1:],
                ),
            )
        }

        embed_dict[model]["labels"] = np.concatenate(
            [
                np.ones(len(data)) * i
                for i, data in enumerate(embed_dict[model]["split"])
            ]
        )
    return embed_dict


def get_original_embeds(models=None, label_file=None):
    if not models:
        models = get_models_in_dir()

    if not np_embeds_path.joinpath("embed_dict.npy").exists():

        paths = ensure_model_paths_match_models(models)

        if label_file:
            label_df, label_idx_dict = load_labels_and_build_dict(label_file)

        embed_dict = {}
        for model, path in tqdm(
            zip(models, paths),
            desc="Organize embeddings and concatenate them",
            position=0,
            leave=False,
        ):
            files = list(path.rglob("*.npy"))
            files.sort()

            metadata = yaml.safe_load(open(path.joinpath("metadata.yml"), "r"))
            segment_s = (
                metadata["segment_length (samples)"] / metadata["sample_rate (Hz)"]
            )

            embed_dict[model] = {"label_dict": {}, "all": []}

            embeddings, all_labels, embed_dict = concat_embeddings_and_split_by_label(
                files,
                model,
                embed_dict,
                segment_s,
                metadata,
                label_file,
                label_df,
                label_idx_dict,
            )

            embed_dict[model]["all"] = embeddings

            embed_dict = finalize_split_arrays(
                label_file, embed_dict, model, all_labels, label_idx_dict
            )

        np.save(np_embeds_path.joinpath("embed_dict.npy"), embed_dict)
    else:
        embed_dict = np.load(
            np_embeds_path.joinpath("embed_dict.npy"), allow_pickle=True
        ).item()
    return embed_dict


def define_2d_reducer(reducer_2d_conf, verbose=True):
    if reducer_2d_conf["name"] == "2dumap":
        reducer = umap.UMAP(**list(reducer_2d_conf.values())[-1], verbose=verbose)
    else:
        assert False, "Reducer not implemented"
    return reducer


def get_reduced_embeddings_by_label(embed, model, reduc_embeds, label_file=None):
    if label_file:
        reduc_embeds[model]["split"] = {
            k: reduc_embeds[model]["all"][embed["labels"] == v]
            for k, v in embed["label_dict"].items()
        }
    else:
        reduc_embeds[model]["split"] = {  # TODO test if this works
            k: np.split(
                reduc_embeds[model]["all"], list(embed["label_dict"].values())[1:]
            )
            for k in embed["label_dict"].keys()
        }
    return reduc_embeds


def reduce_dimensions(embeds, name, reducer_2d_conf=None, label_file=None, **kwargs):
    if not np_embeds_path.joinpath(f"{name}.npy").exists():
        reduc_embeds = {}
        for model, embed in tqdm(
            embeds.items(),
            desc="calculating dimensionality reduction",
            position=0,
            leave=False,
        ):
            reduc_embeds[model] = {}
            reducer = define_2d_reducer(reducer_2d_conf)

            reduc_embeds[model]["all"] = reducer.fit_transform(embed["all"])

            reduc_embeds = get_reduced_embeddings_by_label(
                embed, model, reduc_embeds, label_file
            )

            reduc_embeds[model]["labels"] = embed["labels"]

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


def clustering(embeds, name, clust_conf=None, remove_noise=False, **kwargs):
    if not clust_metrics_path.joinpath(f"{name}_cluster_metrics.json").exists():
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
            if remove_noise:
                if -1 in labels:
                    embeds[model]["all"] = embeds[model]["all"][labels != -1]
                    labels = labels[labels != -1]
            metrics[model]["SS"] = SS(embeds[model]["all"], labels)

            for clust in clust_conf:
                if clust["name"] == "kmeans":
                    clusterer = KMeans(**list(clust.values())[-1])
                elif clust["name"] == "hdbscan":
                    clusterer = hdbscan.HDBSCAN(
                        **list(clust.values())[-1], core_dist_n_jobs=-1
                    )
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


def comppute_reduction(orig_embeddings, name, reducer, **kwargs):
    """
    Compute the dimensionality reduction for the embeddings.

    Parameters
    ----------
    orig_embeddings : dict
        dictionary containing all the embeddings for each model and labels
    name : string
        name of the dimensionality reduction method
    reducer : class
        class of the dimensionality reduction method
    **kwargs : dict
        dictionary containing the configuration of the dimensionality reduction method

    Returns
    -------
    processed_embeds : dict
        dictionary containing the processed embeddings
    """
    processed_embeds = {}
    for model, embed in tqdm(
        orig_embeddings.items(),
        desc=f"calculating dimensionality reduction for {name}",
        position=0,
        leave=False,
    ):
        processed_embeds[model] = {}
        processed_embeds[model]["all"] = reducer.fit_transform(embed["all"])

        processed_embeds[model]["labels"] = embed["labels"]
        if "label_dict" in embed.keys():
            processed_embeds[model]["label_dict"] = embed["label_dict"]
    return processed_embeds


def compare(orig_embeddings, remove_noise=False, **kwargs):

    if "reducer_conf" in kwargs:
        configs = ["normal"] + [conf["name"] for conf in kwargs["reducer_conf"]]
    else:
        configs = ["normal"]

    for config_idx, name in enumerate(configs):
        if not name == "normal":
            conf = [a for a in kwargs["reducer_conf"] if a["name"] == name][0]
            if name.split("_")[0] == "pca":
                reducer = PCA(**conf["conf_1"])
            elif name.split("_")[0] == "spca":
                reducer = SparsePCA(**conf["conf_1"])
            elif name.split("_")[0] == "umap":
                reducer = umap.UMAP(**conf["conf_1"])

            processed_embeds = comppute_reduction(
                orig_embeddings, name, reducer, **kwargs
            )
        else:
            processed_embeds = orig_embeddings

        reduc_2d_embeds = reduce_dimensions(processed_embeds, name, **kwargs)

        # calc_distances(embeds)

        if remove_noise:
            name += "_no_noise"
        metrics_embed, clust_embed = clustering(
            processed_embeds, name, remove_noise=remove_noise, **kwargs
        )
        metrics_reduc, clust_reduc = clustering(
            reduc_2d_embeds, name + "_reduced", remove_noise=remove_noise, **kwargs
        )

    # plot_comparison(
    #     embeds.keys(),
    #     list(embeds.values())[0]["label_dict"].keys(),
    #     reduc_embeds,
    #     name,
    #     metrics_embed,
    #     metrics_reduc,
    #     clust_embed,
    #     clust_reduc,
    #     **kwargs,
    # )


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


def calc_distances(all_embeds):
    if np_embeds_path.joinpath("distances.npy").exists():
        distances = {}
        for model, embeds in tqdm(
            all_embeds.items(), desc="calculating distances", position=0, leave=False
        ):
            distances[model] = {}
            d_all = []
            d_intra = []
            d_inter = []
            for metric in ["cosine", "euclidean"]:
                d_all.append(pairwise_distances(embeds["all"], metric=metric).flatten())
                for ind, (k, v) in tqdm(
                    enumerate(embeds["split"].items()),
                    desc="calculating intra and inter distances",
                    position=1,
                    leave=False,
                ):
                    cluster = v
                    if len(cluster) == 0:
                        continue
                    all_without_cluster = []
                    # dict_copy = embeds['split']
                    [
                        all_without_cluster.extend(em)
                        for label, em in embeds["split"].items()
                        if not label == k
                    ]

                    d_intra.append(pairwise_distances(cluster).flatten())
                    # d_inter.append(
                    #     pairwise_distances(cluster, np.array(all_without_cluster)).flatten()
                    # )

                ratios = [
                    float(np.mean(intr) / np.mean(inte))
                    for intr, inte in zip(d_intra, d_inter)
                ]

            distances[model] = {
                "all": d_all,
                "intra": d_intra,
                "inter": d_inter,
                "ratios": ratios,
            }

        np.save(np_embeds_path.joinpath("distances.npy"), distances)
    else:
        distances = np.load(
            np_embeds_path.joinpath("distances.npy"), allow_pickle=True
        ).item()
    return distances

    # plt.figure()
    # plt.plot(ratios)
    # plt.show()

    # plt.figure()
    # sns.violinplot(data=d)
    # plt.show

    #########################################################


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
    **kwargs,
):
    # fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    if label_file:  # TODO
        label_keys = list(label_keys) + ["Unlabeled"]
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
