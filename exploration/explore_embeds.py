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

# sns.set_theme(style="white")


def set_paths(data_path):
    global main_embeds_path
    global np_embeds_path
    global distances_path
    global np_clust_path
    global clust_metrics_path
    global plot_path

    main_embeds_path = Path(f"results/by_dataset/{data_path}/embeddings")
    np_embeds_path = Path(f"results/by_dataset/{data_path}/numpy_embeddings")
    distances_path = Path(f"results/by_dataset/{data_path}/distances")
    np_clust_path = Path(f"results/by_dataset/{data_path}/numpy_clusterings")
    clust_metrics_path = Path(f"results/by_dataset/{data_path}/cluster_metrics")
    plot_path = Path(f"results/by_dataset/{data_path}/plots")

    main_embeds_path.mkdir(exist_ok=True)
    np_embeds_path.mkdir(exist_ok=True)
    distances_path.mkdir(exist_ok=True)
    np_clust_path.mkdir(exist_ok=True)
    clust_metrics_path.mkdir(exist_ok=True)
    plot_path.mkdir(exist_ok=True)


def get_models_in_dir():
    return [
        d.stem.split("___")[-1].split("-")[0]
        for d in list(main_embeds_path.glob("*"))
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


def apply_labels_to_embeddings(
    df, label_idx_dict, embed, segment_s, audio_file, single_label=True
):
    file_labels = np.ones(embed.shape[0]) * -1
    embed_timestamps = np.arange(embed.shape[0]) * segment_s
    if single_label:
        single_label_arr = [True] * len(embed_timestamps)

    # assert (
    #     df.end.values[-1] <= embed_timestamps[-1] + segment_s*2
    # ), f"Timestamps do not match for {audio_file}"

    for _, row in df.iterrows():
        em_start = np.argmin(np.abs(embed_timestamps - row.start))
        em_end = np.argmin(np.abs(embed_timestamps - row.end))
        if single_label:
            if (
                not all(file_labels[em_start:em_end] == -1)
                and not label_idx_dict[row.label] in file_labels[em_start:em_end]
            ):
                single_label_arr[em_start:em_end] = [False] * (em_end - em_start)
        if row.end - row.start > 0.65:  # 0.33*segment_s:
            file_labels[em_start:em_end] = label_idx_dict[row.label]
    if single_label:
        file_labels[~np.array(single_label_arr)] = -2
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
        if "/" in audio_file:
            audio_file = Path(audio_file).stem + Path(audio_file).suffix
        df = label_df[label_df.audiofilename == audio_file]

        if df.empty:
            all_labels = np.concatenate((all_labels, np.ones(embed.shape[0]) * -1))
            return embed_dict, all_labels

        file_labels = apply_labels_to_embeddings(
            df, label_idx_dict, embed, segment_s, audio_file
        )
        all_labels = np.concatenate((all_labels, file_labels))

        embed_dict[model]["label_dict"] = label_idx_dict

        if np.unique(file_labels).shape[0] > 3:
            embed_timestamps = np.arange(embed.shape[0]) * segment_s
            path = (
                np_embeds_path.parent.joinpath("annotations")
                .joinpath("raven_tables_for_sanity_check")
                .joinpath(model)
            )
            path.mkdir(exist_ok=True, parents=True)
            if (
                len(list(path.iterdir())) < 10
            ):  # make sure to only do this a handful of times
                df_file_gt = label_df[label_df.audiofilename == audio_file]
                df_file_fit = pd.DataFrame()
                df_file_fit["start"] = embed_timestamps[file_labels > -1]
                df_file_fit["end"] = embed_timestamps[file_labels > -1] + segment_s
                inv = {v: k for k, v in label_idx_dict.items()}
                df_file_fit["label"] = [inv[i] for i in file_labels[file_labels > -1]]
                raven_gt = create_Raven_annotation_table(df_file_gt)
                raven_fit = create_Raven_annotation_table(df_file_fit)
                raven_fit["Low Freq (Hz)"] = 1500
                raven_fit["High Freq (Hz)"] = 2000
                raven_gt.to_csv(
                    path.joinpath(f"{audio_file}_gt.txt"), sep="\t", index=False
                )
                raven_fit.to_csv(
                    path.joinpath(f"{audio_file}_fit.txt"), sep="\t", index=False
                )

    else:  # if not file then the split is done by the parent folder name
        # if not file.parent.stem in embed_dict[model]["label_dict"].keys():
        embed_dict[model]["label_dict"][file.parent.stem] = label_idx_dict[
            file.parent.parent.stem
        ]
        all_labels = np.concatenate(
            (
                all_labels,
                np.ones(embed.shape[0]) * label_idx_dict[file.parent.parent.stem],
            )
        )
    return embed_dict, all_labels


def create_Raven_annotation_table(df):
    df.index = np.arange(1, len(df) + 1)
    raven_df = pd.DataFrame()
    raven_df["Selection"] = df.index
    raven_df.index = np.arange(1, len(df) + 1)
    raven_df["Begin Time (s)"] = df.start
    raven_df["End Time (s)"] = df.end
    raven_df["High Freq (Hz)"] = 1000
    raven_df["Low Freq (Hz)"] = 0
    raven_df["Label"] = df.label
    return raven_df


def concat_embeddings_and_split_by_label(
    files, model, embed_dict, segment_s, metadata, label_file, label_df, label_idx_dict
):
    num_embeds = 0
    embeddings, all_labels = np.array([]), np.array([])

    for ind, file in tqdm(
        enumerate(files),
        desc=f"Loading {model} embeddings and split by labels",
        leave=False,
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


def finalize_split_arrays(
    label_file,
    embed_dict,
    model,
    all_labels,
    label_idx_dict,
    remove_noise=False,
    single_label=True,
):
    embed_dict[model]["split"] = {
        k: embed_dict[model]["all"][all_labels == v] for k, v in label_idx_dict.items()
    }
    embed_dict[model]["labels"] = all_labels
    if label_file:
        if remove_noise == False:
            embed_dict[model]["split"].update(
                {"unknown": embed_dict[model]["all"][all_labels == -1]}
            )
            embed_dict[model]["label_dict"].update({"unknown": -1})
        else:
            if single_label:
                embed_dict[model]["all"] = embed_dict[model]["all"][all_labels > -1]
                embed_dict[model]["labels"] = all_labels[all_labels > -1]
            else:
                embed_dict[model]["all"] = embed_dict[model]["all"][all_labels != -1]
                embed_dict[model]["labels"] = all_labels[all_labels != -1]

    return embed_dict


def get_original_embeds(models=None, label_file=None, **kwargs):
    if not models:
        models = get_models_in_dir()

    if not np_embeds_path.joinpath("embed_dict.npy").exists():

        paths = ensure_model_paths_match_models(models)

        if label_file:
            label_df, label_idx_dict = load_labels_and_build_dict(label_file)
        else:
            labels = [d.stem for d in paths[0].iterdir() if d.is_dir()]
            label_idx_dict = {label: idx for idx, label in enumerate(labels)}
            label_df = None

        embed_dict = {}
        for model, path in tqdm(
            zip(models, paths),
            desc="Organize embeddings and concatenate them",
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
                label_file, embed_dict, model, all_labels, label_idx_dict, **kwargs
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
    # if label_file:
    reduc_embeds[model]["split"] = {
        k: reduc_embeds[model]["all"][embed["labels"] == v]
        for k, v in embed["label_dict"].items()
    }
    # else:
    #     reduc_embeds[model]["split"] = {  # TODO test if this works
    #         k: np.split(
    #             reduc_embeds[model]["all"], list(embed["label_dict"].values())[1:]
    #         )
    #         for k in embed["label_dict"].keys()
    #     }
    return reduc_embeds


def reduce_to_2d(embeds, name, reducer_2d_conf=None, label_file=None, **kwargs):
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
    folder = np_embeds_path.joinpath(f"{name}")
    folder.mkdir(exist_ok=True, parents=True)
    processed_embeds = {}
    for model, embed in tqdm(
        orig_embeddings.items(),
        desc=f"calculating dimensionality reduction for {name}",
        position=0,
        leave=False,
    ):
        file = folder.joinpath(f"{name}_{model}.npy")
        if not file.exists():
            processed_embeds[model] = {}
            processed_embeds[model]["all"] = reducer.fit_transform(embed["all"])

            processed_embeds[model]["labels"] = embed["labels"]
            if "label_dict" in embed.keys():
                processed_embeds[model]["label_dict"] = embed["label_dict"]
            np.save(file, processed_embeds[model])
        else:
            processed_embeds[model] = np.load(
                file,
                allow_pickle=True,
            ).item()
    return processed_embeds


def load_task_results(path=None):
    import yaml

    task_dict = {}
    if not path:
        path = main_embeds_path
    path = path.parent.joinpath("task_results")
    path.mkdir(exist_ok=True, parents=True)
    for fold in path.iterdir():
        if fold.is_dir():
            space = fold.stem.split("__")[-1]
            cl_type = fold.stem.split("__")[1]
            if not cl_type in task_dict.keys():
                task_dict[cl_type] = {}
            task_dict[cl_type][space] = {}
            for file in fold.joinpath("metrics").glob("*.yml"):
                model = file.stem.split("species_")[-1]
                task_dict[cl_type][space][model] = yaml.load(
                    open(file, "r"), Loader=yaml.CLoader
                )["Overall Metrics:"]["Macro Accuracy"]
    return task_dict


def compare(orig_embeddings, remove_noise=False, distances=False, **kwargs):

    if "reducer_conf" in kwargs:
        configs = ["normal"] + [conf["name"] for conf in kwargs["reducer_conf"]]
    else:
        configs = ["normal"]
    all_clusts = {}
    all_clusts_reordere = {}
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

        reduc_2d_embeds = reduce_to_2d(processed_embeds, name, **kwargs)

        save_2d_embeds_by_model(reduc_2d_embeds, name)

        if distances:
            distances = calc_distances(processed_embeds, name)

        from exploration.explore_dashboard import plot_overview
        import exploration.explore_dashboard
        import importlib

        importlib.reload(exploration.explore_dashboard)
        from exploration.explore_dashboard import plot_overview

        fig = plot_overview(processed_embeds, name, no_noise=True)
        fig.savefig(plot_path.joinpath(f"{name}_overview_kmeans.png"), dpi=300)

        if remove_noise:
            name += "_no_noise"
        metrics_embed, clust_embed = clustering(
            processed_embeds, name, remove_noise=remove_noise, **kwargs
        )
        # metrics_reduc, clust_reduc = clustering(
        #     reduc_2d_embeds, name + "_reduced", remove_noise=remove_noise, **kwargs
        # )

        # plot_comparison(
        #     reduc_2d_embeds.keys(),
        #     list(reduc_2d_embeds.values())[0]["split"].keys(),
        #     reduc_2d_embeds,
        #     name,
        #     metrics_embed,
        #     metrics_reduc,
        #     clust_embed,
        #     clust_reduc,
        #     **kwargs,
        # )
        all_clusts.update({name: metrics_embed})

    if not clust_metrics_path.joinpath(f"all_clusts_reordered.npy").exists():
        all_clusts_reordered = {"SS": {}, "AMI": {}, "ARI": {}}
        for model in reduc_2d_embeds.keys():
            all_clusts_reordered["SS"][model] = {
                run: all_clusts[run][model]["SS"] for run in all_clusts.keys()
            }
            all_clusts_reordered["AMI"][model] = {
                run: all_clusts[run][model]["AMI"]["kmeans"]
                for run in all_clusts.keys()
            }
            all_clusts_reordered["ARI"][model] = {
                run: all_clusts[run][model]["ARI"]["kmeans"]
                for run in all_clusts.keys()
            }
        np.save(
            clust_metrics_path.joinpath(f"all_clusts_reordered.npy"),
            all_clusts_reordered,
        )
    else:
        all_clusts_reordered = np.load(
            clust_metrics_path.joinpath(f"all_clusts_reordered.npy"), allow_pickle=True
        ).item()
    plot_clustering_by_metric_new(all_clusts_reordered, reduc_2d_embeds.keys())
    scatterplot_clust_vs_class()


def scatterplot_clust_vs_class():
    sns.set_theme(style="white")
    met_clust = np.load(
        clust_metrics_path.joinpath("all_clusts_reordered.npy"), allow_pickle=True
    ).item()
    met_clust_nt = np.load(
        clust_metrics_path.parent.parent.joinpath(
            "neotropic_dawn_chorus/cluster_metrics"
        ).joinpath("all_clusts_reordered.npy"),
        allow_pickle=True,
    ).item()
    color = ["#1E25E5", "#009A2E", "#D81B60", "#4F1EE5"]
    met_class = load_task_results()
    met_class_nt = load_task_results(
        path=main_embeds_path.parent.parent.joinpath("neotropic_dawn_chorus/embeddings")
    )
    symbols = {"birds": "x", "nonbirds": "o"}
    colors = {
        "bird": {"supl": color[0], "ssl": color[1]},
        "frog": {"supl": color[2], "ssl": color[3]},
    }
    # colors = {"supl": color[0], "ssl": color[1]}
    mod_type = {
        "birdnet": "supl-birds",
        "animal2vec_xc": "ssl-birds",
        "animal2vec_mk": "ssl-nonbirds",
        "audiomae": "ssl-nonbirds",
        "aves_especies": "ssl-nonbirds",
        "biolingual": "supl-birds",
        "birdaves_especies": "ssl-birds",
        "avesecho_passt": "supl-birds",
        "insect66": "supl-nonbirds",
        "insect459": "supl-nonbirds",
        "perch_bird": "supl-birds",
        "protoclr": "supl-birds",
        "surfperch": "supl-birds",
        "google_whale": "supl-nonbirds",
        "nonbioaves_especies": "ssl-nonbirds",
    }

    mod_short = {
        "birdnet": "brdnet",
        "animal2vec_xc": "a2v_xc",
        "animal2vec_mk": "a2v_mk",
        "audiomae": "aud_mae",
        "aves_especies": "aves",
        "biolingual": "bioling",
        "birdaves_especies": "birdaves",
        "avesecho_passt": "aecho",
        "insect66": "i66",
        "insect459": "i459",
        "perch_bird": "perch",
        "protoclr": "p_clr",
        "surfperch": "s_perch",
        "google_whale": "g_whale",
        "nonbioaves_especies": "nonbioaves",
    }

    loc_text = {
        "birdnet": {"ha": "left", "va": "top"},
        "animal2vec_xc": {"ha": "left", "va": "center"},
        "animal2vec_mk": {"ha": "left", "va": "center"},
        "audiomae": {"ha": "left", "va": "bottom"},
        "aves_especies": {"ha": "left", "va": "center"},
        "biolingual": {"ha": "left", "va": "top"},
        "birdaves_especies": {"ha": "left", "va": "top"},
        "avesecho_passt": {"ha": "right", "va": "top"},
        "insect66": {"ha": "right", "va": "top"},
        "insect459": {"ha": "left", "va": "top"},
        "perch_bird": {"ha": "left", "va": "center"},
        "protoclr": {"ha": "left", "va": "center"},
        "surfperch": {"ha": "left", "va": "bottom"},
        "google_whale": {"ha": "left", "va": "bottom"},
        "nonbioaves_especies": {"ha": "left", "va": "top"},
    }

    loc = {
        "birdnet": [+0.01, -0.01],
        "animal2vec_xc": [0.01, -0.01],
        "animal2vec_mk": [0.01, 0],
        "audiomae": [+0.01, +0.01],
        "aves_especies": [+0.01, 0],
        "avesecho_passt": [-0.01, 0],
        "biolingual": [+0, -0.01],
        "birdaves_especies": [0.01, -0.01],
        "insect66": [-0.01, 0],
        "insect459": [+0.01, -0],
        "perch_bird": [+0.01, -0],
        "protoclr": [0.01, -0.01],
        "surfperch": [+0.01, +0],
        "google_whale": [+0.01, +0],
        "nonbioaves_especies": [+0.01, -0.01],
    }

    fig, axes = plt.subplots(figsize=(5, 4))
    plt.subplots_adjust(bottom=0.3)
    for model in met_clust_nt["AMI"].keys():
        axes.scatter(
            [met_class_nt["knn"]["normal"][model]][0],
            [met_clust_nt["AMI"][model]["normal_no_noise"]][0],
            label=model,
            marker=symbols[mod_type[model].split("-")[-1]],
            color=colors["bird"][mod_type[model].split("-")[0]],
        )
        plt.text(
            met_class_nt["knn"]["normal"][model] + loc[model][0],
            met_clust_nt["AMI"][model]["normal_no_noise"] + loc[model][1],
            mod_short[model],
            fontsize=10,
            **loc_text[model],
        )
    axes.scatter([], [], label="supl(birds)", marker="x", color=color[0])
    axes.scatter([], [], label="ssl(birds)", marker="x", color=color[1])
    axes.scatter([], [], label="supl(non-birds)", marker="o", color=color[0])
    axes.scatter([], [], label="ssl(non-birds)", marker="o", color=color[1])
    axes.set_xlabel("kNN. Class. Macro Accuracy")
    axes.set_ylabel("Clustering AMI")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    hand, label = axes.get_legend_handles_labels()
    # fig.legend(["supl-birds", "ssl-birds", "supl-nonbirds", "ssl-nonbirds"], loc="upper right")
    fig.legend(hand[-4:], label[-4:], ncol=2, loc="lower center")
    # fig.legend(hand, label, loc="upper right")
    fig.savefig(
        plot_path.joinpath(f"scatterplot_clust_vs_class_nt_knn_normal.png"), dpi=600
    )

    reduc = "normal"
    fig, axes = plt.subplots(figsize=(5, 4))
    plt.subplots_adjust(bottom=0.3)
    for model in met_clust["AMI"].keys():
        axes.scatter(
            [met_class_nt["knn"][f"{reduc}"][model]][0],
            [met_clust_nt["AMI"][model][f"{reduc}_no_noise"]][0],
            label=model,
            marker=symbols[mod_type[model].split("-")[-1]],
            color=colors["bird"][mod_type[model].split("-")[0]],
            # s=40,
            # color="k",
        )
        axes.scatter(
            [met_class["knn"][f"{reduc}"][model]][0],
            [met_clust["AMI"][model][f"{reduc}_no_noise"]][0],
            label=model,
            marker=symbols[mod_type[model].split("-")[-1]],
            # color=colors['bird'][mod_type[model].split("-")[0]],
            color="k",
            # s=12,
        )
        axes.arrow(
            met_class_nt["knn"][f"{reduc}"][model],
            met_clust_nt["AMI"][model][f"{reduc}_no_noise"],
            met_class["knn"][f"{reduc}"][model]
            - met_class_nt["knn"][f"{reduc}"][model],
            met_clust["AMI"][model][f"{reduc}_no_noise"]
            - met_clust_nt["AMI"][model][f"{reduc}_no_noise"],
            color="gray",
            length_includes_head=True,
            head_width=0.007,
            head_length=0.011,
            linewidth=0.5,
        )
        plt.text(
            met_class_nt["knn"][f"{reduc}"][model] + loc[model][0],
            met_clust_nt["AMI"][model][f"{reduc}_no_noise"] + loc[model][1],
            mod_short[model],
            fontsize=10,
            **loc_text[model],
        )
    axes.scatter([], [], label="supl(bird)", marker="x", color=color[0])
    axes.scatter([], [], label="ssl(bird)", marker="x", color=color[1])
    axes.scatter([], [], label="supl(non-bird)", marker="o", color=color[0])
    axes.scatter([], [], label="ssl(non-bird)", marker="o", color=color[1])
    axes.scatter([], [], marker="x", label="frog data(bird)", color="black")
    axes.scatter([], [], marker="o", label="frog data(non-bird)", color="black")
    axes.set_xlabel("kNN. Class. Macro Accuracy")
    axes.set_ylabel("Clustering AMI")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    hand, label = axes.get_legend_handles_labels()
    # fig.legend(["supl-birds", "ssl-birds", "supl-nonbirds", "ssl-nonbirds"], loc="upper right")
    fig.legend(hand[-6:], label[-6:], ncol=3, loc="lower center", columnspacing=0.7)
    # fig.legend(hand, label, loc="upper right")
    fig.savefig(
        plot_path.joinpath(f"scatterplot_clust_vs_class_neotrop_anuran_{reduc}.png"),
        dpi=600,
    )

    #### subplots with lines
    model_order = [
        "birdnet",  # "supl-birds"
        "perch_bird",  # "supl-birds"
        "avesecho_passt",  # "supl-birds"
        "surfperch",  # "supl-birds"
        "biolingual",  # "supl-birds"
        "protoclr",  # "supl-birds"
        "insect66",  # "supl-nonbirds"
        "insect459",  # "supl-nonbirds"
        "google_whale",  # "supl-nonbirds"
        "animal2vec_xc",  # "ssl-birds"
        "birdaves_especies",  # "ssl-birds"
        "aves_especies",  # "ssl-nonbirds"
        "nonbioaves_especies",  # "ssl-nonbirds"
        "animal2vec_mk",  # "ssl-nonbirds"
        "audiomae",  # "ssl-nonbirds"
    ]
    mods_xticks = [mod_short[model] for model in model_order]
    clust = {
        "nt": [
            met_clust_nt["AMI"][model][f"{reduc}_no_noise"] for model in model_order
        ],
        "fr": [met_clust["AMI"][model][f"{reduc}_no_noise"] for model in model_order],
    }
    clas = {
        "nt": [met_class_nt["knn"][f"{reduc}"][model] for model in model_order],
        "fr": [met_class["knn"][f"{reduc}"][model] for model in model_order],
    }
    fig, axes = plt.subplots(2, 1, figsize=(6, 5))
    plt.subplots_adjust(bottom=0.3)
    for idx, (species, name) in enumerate(zip(["bird", "frog"], ["nt", "fr"])):
        for section in [[0, 1, 2, 3, 4, 5], [6, 7, 8], [9, 10], [11, 12, 13, 14]]:
            axes[0].plot(
                np.array(model_order)[section],
                np.array(clust[name])[section],
                c=color[idx],
                marker="x",
                label=f"{species} data",
            )

    for idx, (species, name) in enumerate(zip(["bird", "frog"], ["nt", "fr"])):
        for section in [[0, 1, 2, 3, 4, 5], [6, 7, 8], [9, 10], [11, 12, 13, 14]]:
            axes[1].plot(
                np.array(model_order)[section],
                np.array(clas[name])[section],
                c=color[idx],
                marker="x",
                label=f"{species} data",
            )

    axes[0].vlines(
        [5.5, 8.5, 10.5], ymin=0.15, ymax=0.82, color="gray", linestyle="--", lw=2
    )
    axes[1].vlines(
        [5.5, 8.5, 10.5], ymin=0.4, ymax=0.9, color="gray", linestyle="--", lw=2
    )
    axes[0].text(
        2, 0.7, "supl \nbird", ha="center", fontsize=12, color="black", alpha=0.85
    )
    axes[0].text(
        7, 0.7, "supl \nnon-bird", ha="center", fontsize=12, color="black", alpha=0.85
    )
    axes[0].text(
        9.5, 0.7, "ssl \nbird", ha="center", fontsize=12, color="black", alpha=0.85
    )
    axes[0].text(
        13, 0.7, "ssl \nnon-bird", ha="center", fontsize=12, color="black", alpha=0.85
    )

    axes[0].set_ylabel("AMI")
    axes[1].set_ylabel("Macro accuracy")
    axes[1].set_xlabel("Models")
    axes[1].set_xticks(np.arange(len(model_order)) + 0.2)
    axes[1].set_xticklabels(mods_xticks, rotation=45, ha="right", fontsize=12)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[0].set_xticklabels([])

    axes[0].plot([], [], label="bird data", color=color[0])
    axes[0].plot([], [], label="frog data", color=color[1])
    axes[0].plot([], [], label="bird data", color=color[0])
    axes[0].plot([], [], label="frog data", color=color[1])
    hand, label = axes[0].get_legend_handles_labels()
    fig.legend(hand[-2:], label[-2:], loc="upper center", ncol=2, columnspacing=0.7)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(plot_path.joinpath("clust_and_class_lineplots.png"), dpi=800)

    ### get values for table
    d, d_std = {}, {}
    reduc = "pca_300"
    d[reduc], d_std[reduc] = get_table_metrics(
        reduc, met_class, met_clust, met_class_nt, met_clust_nt
    )
    mets = {"means": d, "stds": d_std}
    with open(plot_path.joinpath("clustering_summary.yml"), "w") as f:
        yaml.dump(d, f)

    table = f"""
        ssl & {d['normal']['nt']['ssl']['class']}& {d['umap_300']['nt']['ssl']['class']}& {d['pca_300']['nt']['ssl']['class']} \\
        supl & {d['normal']['nt']['subl']['class']}& {d['umap_300']['nt']['subl']['class']}& {d['pca_300']['nt']['subl']['class']} \\
        bird & {d['normal']['nt']['bird']['class']}& {d['umap_300']['nt']['bird']['class']}& {d['pca_300']['nt']['bird']['class']} \\
        non-bird & {d['normal']['nt']['nonbird']['class']}& {d['umap_300']['nt']['nonbird']['class']}& {d['pca_300']['nt']['nonbird']['class']} \\
        \hline
            
        \hline
        ssl & {d['normal']['nt']['ssl']['clust']}& {d['umap_300']['nt']['ssl']['clust']}& {d['pca_300']['nt']['ssl']['clust']} \\
        supl & {d['normal']['nt']['subl']['clust']}& {d['umap_300']['nt']['subl']['clust']}& {d['pca_300']['nt']['subl']['clust']} \\
        bird & {d['normal']['nt']['bird']['clust']}& {d['umap_300']['nt']['bird']['clust']}& {d['pca_300']['nt']['bird']['clust']} \\
        non-bird & {d['normal']['nt']['nonbird']['clust']}& {d['umap_300']['nt']['nonbird']['clust']}& {d['pca_300']['nt']['nonbird']['clust']} \\
        \hline
            
        \hline
        category &
        original &
        umap &
        pca \\
        \hline
        ssl & {d['normal']['frog']['ssl']['class']}& {d['umap_300']['frog']['ssl']['class']}& {d['pca_300']['frog']['ssl']['class']} \\
        supl & {d['normal']['frog']['subl']['class']}& {d['umap_300']['frog']['subl']['class']}& {d['pca_300']['frog']['subl']['class']} \\
        bird & {d['normal']['frog']['bird']['class']}& {d['umap_300']['frog']['bird']['class']}& {d['pca_300']['frog']['bird']['class']} \\
        non-bird & {d['normal']['frog']['nonbird']['class']}& {d['umap_300']['frog']['nonbird']['class']}& {d['pca_300']['frog']['nonbird']['class']} \\
        \hline
            
        \hline
        ssl & {d['normal']['frog']['ssl']['clust']}& {d['umap_300']['frog']['ssl']['clust']}& {d['pca_300']['frog']['ssl']['clust']} \\
        supl & {d['normal']['frog']['subl']['clust']}& {d['umap_300']['frog']['subl']['clust']}& {d['pca_300']['frog']['subl']['clust']} \\
        bird & {d['normal']['frog']['bird']['clust']}& {d['umap_300']['frog']['bird']['clust']}& {d['pca_300']['frog']['bird']['clust']} \\
        non-bird & {d['normal']['frog']['nonbird']['clust']}& {d['umap_300']['frog']['nonbird']['clust']}& {d['pca_300']['frog']['nonbird']['clust']} \\
    """


def get_table_metrics(reduc, met_class, met_clust, met_class_nt, met_clust_nt):
    class_nt = met_class_nt["linear"][f"{reduc}"]
    clust_nt = {
        k: float(v[f"{reduc}_no_noise"]) for k, v in met_clust_nt["AMI"].items()
    }
    class_frog = met_class["linear"][f"{reduc}"]
    clust_frog = {k: float(v[f"{reduc}_no_noise"]) for k, v in met_clust["AMI"].items()}

    ssl_mods = [
        "animal2vec_mk",
        "audiomae",
        "aves_especies",
        "animal2vec_xc",
        "birdaves_especies",
        "nonbioaves_especies",
    ]
    supl_mods = [
        "birdnet",
        "biolingual",
        "avesecho_passt",
        "insect66",
        "insect459",
        "perch_bird",
        "protoclr",
        "surfperch",
        "google_whale",
    ]

    bird = [
        "birdnet",
        "biolingual",
        "birdaves_especies",
        "avesecho_passt",
        "perch_bird",
        "protoclr",
        "surfperch",
        "animal2vec_xc",
    ]

    nonbird = [
        "animal2vec_mk",
        "audiomae",
        "aves_especies",
        "google_whale",
        "nonbioaves_especies",
        "insect66",
        "insect459",
    ]
    d = {
        "frog": {
            "ssl": {
                "clust": float(np.round(np.mean([clust_frog[m] for m in ssl_mods]), 3)),
                "class": float(np.round(np.mean([class_frog[m] for m in ssl_mods]), 3)),
            },
            "subl": {
                "clust": float(
                    np.round(np.mean([clust_frog[m] for m in supl_mods]), 3)
                ),
                "class": float(
                    np.round(np.mean([class_frog[m] for m in supl_mods]), 3)
                ),
            },
            "bird": {
                "clust": float(np.round(np.mean([clust_frog[m] for m in bird]), 3)),
                "class": float(np.round(np.mean([class_frog[m] for m in bird]), 3)),
            },
            "nonbird": {
                "clust": float(np.round(np.mean([clust_frog[m] for m in nonbird]), 3)),
                "class": float(np.round(np.mean([class_frog[m] for m in nonbird]), 3)),
            },
        },
        "nt": {
            "ssl": {
                "clust": float(np.round(np.mean([clust_nt[m] for m in ssl_mods]), 3)),
                "class": float(np.round(np.mean([class_nt[m] for m in ssl_mods]), 3)),
            },
            "subl": {
                "clust": float(np.round(np.mean([clust_nt[m] for m in supl_mods]), 3)),
                "class": float(np.round(np.mean([class_nt[m] for m in supl_mods]), 3)),
            },
            "bird": {
                "clust": float(np.round(np.mean([clust_nt[m] for m in bird]), 3)),
                "class": float(np.round(np.mean([class_nt[m] for m in bird]), 3)),
            },
            "nonbird": {
                "clust": float(np.round(np.mean([clust_nt[m] for m in nonbird]), 3)),
                "class": float(np.round(np.mean([class_nt[m] for m in nonbird]), 3)),
            },
        },
    }

    d_std = {
        "frog": {
            "ssl": {
                "clust": float(np.round(np.std([clust_frog[m] for m in ssl_mods]), 3)),
                "class": float(np.round(np.std([class_frog[m] for m in ssl_mods]), 3)),
            },
            "subl": {
                "clust": float(np.round(np.std([clust_frog[m] for m in supl_mods]), 3)),
                "class": float(np.round(np.std([class_frog[m] for m in supl_mods]), 3)),
            },
            "bird": {
                "clust": float(np.round(np.std([clust_frog[m] for m in bird]), 3)),
                "class": float(np.round(np.std([class_frog[m] for m in bird]), 3)),
            },
            "nonbird": {
                "clust": float(np.round(np.std([clust_frog[m] for m in nonbird]), 3)),
                "class": float(np.round(np.std([class_frog[m] for m in nonbird]), 3)),
            },
        },
        "nt": {
            "ssl": {
                "clust": float(np.round(np.std([clust_nt[m] for m in ssl_mods]), 3)),
                "class": float(np.round(np.std([class_nt[m] for m in ssl_mods]), 3)),
            },
            "subl": {
                "clust": float(np.round(np.std([clust_nt[m] for m in supl_mods]), 3)),
                "class": float(np.round(np.std([class_nt[m] for m in supl_mods]), 3)),
            },
            "bird": {
                "clust": float(np.round(np.std([clust_nt[m] for m in bird]), 3)),
                "class": float(np.round(np.std([class_nt[m] for m in bird]), 3)),
            },
            "nonbird": {
                "clust": float(np.round(np.std([clust_nt[m] for m in nonbird]), 3)),
                "class": float(np.round(np.std([class_nt[m] for m in nonbird]), 3)),
            },
        },
    }
    return d, d_std


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


def save_2d_embeds_by_model(reduc_embeds, name):
    for model, embed in reduc_embeds.items():
        folder = np_embeds_path.joinpath(name)
        if not folder.joinpath(f"{name}_{model}_2d.npy").exists():
            embed["x"] = embed["all"][:, 0]
            embed["y"] = embed["all"][:, 1]

            for k, v in embed.items():
                embed[k] = convert_numpy_types(v)
                for kk, vv in embed["split"].items():
                    embed["split"][kk] = convert_numpy_types(vv)

            folder.mkdir(exist_ok=True, parents=True)
            np.save(
                folder.joinpath(f"{name}_{model}_2d.npy"),
                embed,
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


def calc_distances(all_embeds, name):
    if not distances_path.joinpath(f"{name}_distances.npy").exists():
        distances = {}
        for model, embeds in tqdm(
            all_embeds.items(), desc="calculating distances", position=0, leave=False
        ):
            distances[model] = {}
            for metric in ["euclidean"]:  # "cosine",
                # d_all.append(pairwise_distances(embeds["all"], metric=metric).flatten())
                distances[model][metric] = {}
                for ind, lab in tqdm(
                    enumerate(np.unique(embeds["labels"])),
                    desc="calculating intra and inter distances",
                    position=1,
                    leave=False,
                ):
                    if lab == -1:
                        continue
                    cluster = embeds["all"][embeds["labels"] == lab]
                    if len(cluster) == 0:
                        continue
                    label_key = [
                        k for k, v in embeds["label_dict"].items() if v == lab
                    ][0]
                    distances[model][metric].update({label_key: {}})
                    all_without_cluster = embeds["all"][embeds["labels"] != lab]

                    distances[model][metric][label_key].update(
                        {
                            "intra": pairwise_distances(cluster, metric=metric)
                            .flatten()
                            .tolist()
                        }
                    )

                    distances[model][metric][label_key].update(
                        {
                            "inter": np.sort(
                                pairwise_distances(
                                    cluster, all_without_cluster, metric=metric
                                )
                            )[:, :15]
                            .flatten()
                            .tolist()
                        }
                    )

                    distances[model][metric][label_key].update(
                        {
                            "ratio": np.mean(
                                distances[model][metric][label_key]["intra"]
                            )
                            / np.mean(distances[model][metric][label_key]["inter"])
                        }
                    )

        np.save(distances_path.joinpath(f"{name}_distances.npy"), distances)
    else:
        distances = np.load(
            distances_path.joinpath(f"{name}_distances.npy"), allow_pickle=True
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
