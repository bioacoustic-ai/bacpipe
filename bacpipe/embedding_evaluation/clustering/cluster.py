import numpy as np
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import adjusted_mutual_info_score as ami_score

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as SS
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

import json


def convert_numpy_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


def save_clustering_performance(paths, clusterings, metrics, remove_noise):
    """
    Save the clustering performance. A json file for the performance
    metrics and a npy file with the cluster labels for visualizations.

    Parameters
    ----------
    paths : SimpleNamespace object
        dict with path attributes
    clusterings : np.array
        clustering labels
    metrics : dict
        clustering performance
    remove_noise : bool
        whether to remove the not annotated segments or not
    """
    if remove_noise:
        appendix = "_no_noise"
    else:
        appendix = "_with_noise"
    clust_path = lambda a, b, c: paths.clust_path.joinpath(f"clust_{a}_{b}.{c}")

    np.save(clust_path("label", appendix, "npy"), clusterings)

    with open(clust_path("metrics", appendix, "json"), "w") as f:
        json.dump(metrics, f, default=convert_numpy_types)


def compute_clusterings(embeds, labels, cluster_configs):
    """
    Run clustering algorithms.

    Parameters
    ----------
    embeds : np.array
        embeddings
    labels : list
        ground truth labels
    cluster_configs : dict
        clustering algorithm objects

    Returns
    -------
    dict
        performance metrics
    dict
        labels accordings to clustering algorithms
    """
    metrics = {}
    clusterings = {}
    for name, clusterer in cluster_configs.items():
        clusterings[name] = clusterer.fit_predict(embeds)

    metrics["SS"] = SS(embeds, labels)
    metrics["AMI"] = {}
    metrics["ARI"] = {}
    for clust_name, cluster_labels in clusterings.items():
        metrics["AMI"][clust_name] = AMI(labels, cluster_labels)
        metrics["ARI"][clust_name] = ARI(labels, cluster_labels)
    return metrics, clusterings


def get_clustering_models(clust_params):
    """
    Initialize the clustering models specified in settings.yaml

    Parameters
    ----------
    clust_params : dict
        clusterings specified in settings.yaml

    Returns
    -------
    dict
        clustering objects to run the data on
    """
    cluster_configs = {}
    for name, params in clust_params.items():
        if name == "kmeans":
            cluster_configs[name] = KMeans(**params)

        if False:  # TODO name == "hdbscan":
            from hdbscan import hdbscan

            cluster_configs[name] = hdbscan.HDBSCAN(**params, core_dist_n_jobs=-1)
    return cluster_configs


def get_nr_of_clusters(ground_truth, labels, clust_configs, **kwargs):
    """
    Get number of clusters either from ground truth or if doesn't exist
    from settings.yaml

    Parameters
    ----------
    labels : list
        ground truth labels
    clust_configs : dict
        clusterings specified in settings.yaml

    Returns
    -------
    dict
        clustering dict with correct number of clusters
    """
    clust_params = {}
    for config in clust_configs.values():
        if config["name"] == "kmeans":
            if ground_truth is not None:
                nr_of_classes = len(np.unique(labels))
                clust_params[config["name"]] = {
                    "n_clusters": nr_of_classes,
                }
            else:
                clust_params[config["name"]] = config["params"]
        else:
            if config["bool"]:
                clust_params[config["name"]] = config["params"]
    return clust_params


def clustering(
    paths, embeds, ground_truth, overwrite=False, remove_noise=False, **kwargs
):
    """
    Clustering pipeline.

    Parameters
    ----------
    paths : SimpleNamespace object
        dict with path attributs for saving and loading
    embeds : np.array
        embeddings
    ground_truth : dict
        ground truth labels and a label2dict dictionary
    overwrite : bool, optional
        whether to overwrite exisiting clustering files, by default False
    remove_noise : bool, optional
        remove embeddings corresponding to non-annotated segments, by default False
    """
    if overwrite or not len(list(paths.clust_path.glob("*.json"))) > 0:

        labels = ground_truth["labels"]

        if remove_noise:
            if -1 in labels:
                embeds = embeds[labels != -1]
                labels = labels[labels != -1]

        clust_params = get_nr_of_clusters(labels, **kwargs)

        cluster_configs = get_clustering_models(clust_params)

        metrics, clusterings = compute_clusterings(embeds, labels, cluster_configs)

        save_clustering_performance(paths, clusterings, metrics, remove_noise)
    else:
        print(
            "Clustering file cluster_metrics.json already exists and"
            " so is not computed. If you want to overwrite existing results, "
            "set overwrite to True in settings.yaml."
        )
