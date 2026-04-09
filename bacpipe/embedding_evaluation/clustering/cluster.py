import numpy as np

import json
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as SS
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

import bacpipe.embedding_evaluation.label_embeddings as le
import bacpipe

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

def run_clustering(
    embeds, cluster_configs, 
    label_column=None, ground_truth=[]
    ):
    """
    Fit clustering algorithms to embeddings.

    Parameters
    ----------
    embeds : np.array
        embeddings
    cluster_configs : dict
        clustering algorithm objects
    label_column : string
        label type defined in annotations.csv file
    ground_truth : list
        ground truth labels

    Returns
    -------
    dict
        labels accordings to clustering algorithms
    """
    clusterings = {}
    for name, clusterer in cluster_configs.items():
        clusterings[name] = clusterer.fit_predict(embeds)
        if len(ground_truth) > 0:
            clusterings[name + "_no_noise"] = clusterer.fit_predict(
                embeds[ground_truth != -1]
            )
    if len(ground_truth) > 0 and label_column:
        clusterings[label_column] = ground_truth
        clusterings[f"{label_column}_no_noise"] = ground_truth[ground_truth != -1]
    return clusterings

def eval_clustering(
    clusterings, ground_truth=[],
    default_labels=None, 
    label_column=None, 
    **kwargs
    ):
    """
    Evaluate clustering performance.

    Parameters
    ----------
    clusterings : dict
        dictionary with clusterings
    ground_truth : list
        ground truth labels
    default_labels : dict
        default labels for the dataset
    label_column : string
        label type defined in annotations.csv file

    Returns
    -------
    dict
        performance metrics
    """
    metrics = {"AMI": dict(), "ARI": dict()}
    for cl_name, cl_labels in clusterings.items():
        if cl_name == f"{label_column}_no_noise":
            if -1 in ground_truth:
                embeds = embeds[ground_truth != -1]
                cl_labels = ground_truth[ground_truth != -1]
            
        if hasattr(default_labels, 'kmeans'):
            default_labels["kmeans"] = clusterings["kmeans"]
        if not default_labels:
            metrics[f"AMI"][f"{cl_name}-ground_truth"] = AMI(ground_truth, cl_labels)
            metrics[f"ARI"][f"{cl_name}-ground_truth"] = ARI(ground_truth, cl_labels)
        else:
            for def_name, def_labels in default_labels.items():
                if "no_noise" in cl_name:
                    def_labels = np.array(def_labels)[ground_truth != -1]
                metrics[f"AMI"][f"{cl_name}-{def_name}"] = AMI(def_labels, cl_labels)
                metrics[f"ARI"][f"{cl_name}-{def_name}"] = ARI(def_labels, cl_labels)
    return metrics

def eval_with_silhouette(embeds, ground_truth, metrics=None):
    if not metrics:
        metrics = dict()
    metrics["SS"] = SS(embeds, ground_truth)
    return metrics

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


def clustering_pipeline(
    model_name,
    ground_truth, embeds, 
    paths=None, 
    overwrite=True, 
    label_column=bacpipe.settings.label_column, 
    **kwargs
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
    """
    if not kwargs:
        kwargs = {**vars(bacpipe.settings)}
        kwargs.pop('label_column')
    if not paths:
        get_paths_func = bacpipe.make_set_paths_func(
            bacpipe.config.audio_dir, bacpipe.settings.main_results_dir
        )
        paths = get_paths_func(model_name)
    if (
        overwrite
        or not len(list(paths.clust_path.glob("*.json"))) > 0
    ):
        
        if "audio_dir" in kwargs: kwargs.pop("audio_dir")
        
        if ground_truth:
            ground_truth = ground_truth[f"label:{label_column}"]
        else:
            ground_truth = []

        clust_params = get_nr_of_clusters(ground_truth, **kwargs)

        cluster_configs = get_clustering_models(clust_params)

        default_labels = le.create_default_labels(
            paths.audio_dir, paths.clust_path.parent.stem, paths, **kwargs
        )
        
        clusterings = run_clustering(
            embeds, cluster_configs, label_column, ground_truth
            )
        metrics = eval_clustering(
            clusterings, ground_truth, default_labels, label_column, **kwargs
        )
        if kwargs.get('evaluate_with_silhouette'):
            metrics = eval_with_silhouette(embeds, clusterings, metrics)

        save_clustering_performance(paths, clusterings, metrics, label_column)
        
    else:
        logger.info(
            "Clustering file cluster_metrics.json already exists and"
            " so is not computed. If you want to overwrite existing results, "
            "set overwrite to True in settings.yaml."
        )
        clusterings = np.load(
            paths.clust_path.joinpath(f"clust_labels.npy"), 
            allow_pickle=True
            ).item()
        with open(paths.clust_path.joinpath(f"clust_results.json"), "r") as f:
            metrics = json.load(f)
            
    return clusterings, metrics
