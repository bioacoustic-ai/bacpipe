import numpy as np

import json
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as SS
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

import bacpipe.embedding_evaluation.label_embeddings as le

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


def save_clustering_performance(paths, clusterings, metrics, label_column):
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
    label_column : str
        label as defined in annotation.csv file
    """
    clusterings = {k: v for k, v in clusterings.items() if not label_column in k}
    np.save(paths.clust_path.joinpath(f"clust_labels.npy"), clusterings)

    if metrics:
        with open(paths.clust_path.joinpath(f"clust_results.json"), "w") as f:
            json.dump(metrics, f, default=convert_numpy_types, indent=2)


def compute_clusterings(
    embeds,
    labels,
    cluster_configs,
    default_labels,
    label_column,
    evaluate_with_silhouette=False,
    **kwargs,
):
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
    default_labels : dict
        default labels for the dataset
    label_column : string
        label type defined in annotations.csv file
    evaluate_with_silhouette : bool, optional
        whether to evaluate with silhouette score, by default False

    Returns
    -------
    dict
        performance metrics
    dict
        labels accordings to clustering algorithms
    """
    metrics = {"SS": dict(), "AMI": dict(), "ARI": dict()}
    clusterings = {}
    for name, clusterer in cluster_configs.items():
        clusterings[name] = clusterer.fit_predict(embeds)
        if len(labels) > 0:
            clusterings[name + "_no_noise"] = clusterer.fit_predict(
                embeds[labels != -1]
            )
    if len(labels) > 0:
        clusterings[label_column] = labels
        clusterings[f"{label_column}_no_noise"] = labels[labels != -1]
        default_labels["kmeans"] = clusterings["kmeans"]

    for cl_name, cl_labels in clusterings.items():
        if cl_name == f"{label_column}_no_noise":
            if -1 in labels:
                embeds = embeds[labels != -1]
                cl_labels = labels[labels != -1]

        if evaluate_with_silhouette:
            metrics["SS"][cl_name] = SS(embeds, cl_labels)
        for def_name, def_labels in default_labels.items():
            if "no_noise" in cl_name:
                def_labels = np.array(def_labels)[labels != -1]
            metrics[f"AMI"][f"{cl_name}-{def_name}"] = AMI(def_labels, cl_labels)
            metrics[f"ARI"][f"{cl_name}-{def_name}"] = ARI(def_labels, cl_labels)

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


def get_nr_of_clusters(labels, clust_configs, **kwargs):
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
            if len(labels) > 0:
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


def clustering(paths, embeds, ground_truth, label_column, overwrite=False, **kwargs):
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
    """
    if overwrite or not len(list(paths.clust_path.glob("*.json"))) > 0:

        if ground_truth:
            labels = ground_truth[f"label:{label_column}"]
        else:
            labels = []

        clust_params = get_nr_of_clusters(labels, **kwargs)

        cluster_configs = get_clustering_models(clust_params)

        default_labels = le.create_default_labels(
            paths, paths.clust_path.parent.stem, **kwargs
        )

        metrics, clusterings = compute_clusterings(
            embeds, labels, cluster_configs, default_labels, label_column
        )

        save_clustering_performance(paths, clusterings, metrics, label_column)
    else:
        logger.info(
            "Clustering file cluster_metrics.json already exists and"
            " so is not computed. If you want to overwrite existing results, "
            "set overwrite to True in settings.yaml."
        )
