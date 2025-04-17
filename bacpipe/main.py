import time
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

import bacpipe.generate_embeddings as ge

from bacpipe.embedding_evaluation.visualization.visualize import (
    plot_comparison,
    plot_embeddings,
    visualise_classification_results_across_models,
)
from bacpipe.embedding_evaluation.label_embeddings import (
    create_default_labels,
    generate_annotations_for_classification_task,
    ground_truth_by_model,
)

from bacpipe.embedding_evaluation.classification.classify import classification_pipeline

from bacpipe.embedding_evaluation.clustering.cluster_embeddings import clustering

from bacpipe.embedding_evaluation.distance_evalutation.distances import (
    calc_distances,
)

logger = logging.getLogger("bacpipe")


def set_paths(audio_dir, model_name, main_results_dir, **kwargs):
    """
    Generate model specific paths for the results of the embedding evaluation.
    This includes paths for the embeddings, labels, clustering, classification,
    distances, and plots. The paths are created based on the audio directory,
    and model name.

    Parameters
    ----------
    audio_dir : string
        full path to audio files
    model_name : string
        name of the model used for embedding
    main_results_dir : string
        top level directory for the results of the embedding evaluation

    Returns
    -------
    paths : SimpleNamespace
        object containing the paths for the results of the embedding evaluation
    """
    dataset_path = Path(main_results_dir).joinpath(Path(audio_dir).stem)
    task_path = dataset_path.joinpath("task_results").joinpath(model_name)

    paths = {
        "main_embeds_path": dataset_path.joinpath("embeddings"),
        "labels_path": task_path.joinpath("labels"),
        "clust_path": task_path.joinpath("clustering"),
        "class_path": task_path.joinpath("classification"),
        "distances_path": task_path.joinpath("distances"),
        "plot_path": task_path.joinpath("plots"),
    }

    paths = SimpleNamespace(**paths)

    paths.main_embeds_path.mkdir(exist_ok=True, parents=True)
    paths.labels_path.mkdir(exist_ok=True, parents=True)
    paths.clust_path.mkdir(exist_ok=True)
    paths.class_path.mkdir(exist_ok=True)
    paths.distances_path.mkdir(exist_ok=True)
    paths.plot_path.mkdir(exist_ok=True)
    return paths


def get_model_names(
    embedding_model, audio_dir, main_results_dir, embed_parent_dir, **kwargs
):
    """
    Get the names of the models used for embedding. This is either done
    by using already computed embeddings or by using the selected models
    from the config file. If already computed embeddings are used, the
    model names are extracted from the directory structure.

    Parameters
    ----------
    embedding_model : dict
        dict containing the configuration for the embedding models
    audio_dir : string
        full path to audio files
    main_results_dir : string
        top level directory for the results of the embedding evaluation
    embed_parent_dir : string
        parent directory for the embeddings

    Raises
    ------
    ValueError
        If already computed embeddings are used, but no embeddings
        are found in the specified directory.
    """
    global model_names
    if embedding_model["already_computed"]:

        dataset_name = Path(audio_dir).stem
        main_results_path = (
            Path(main_results_dir).joinpath(dataset_name).joinpath(embed_parent_dir)
        )
        model_names = [
            d.stem.split("___")[-1].split("-")[0]
            for d in list(main_results_path.rglob("*"))
            if d.is_dir()
        ]
        if not model_names:
            raise ValueError(
                "No embedding models found in the specified directory. "
                "You have selected the option to use already computed embeddings, "
                "but no embeddings were found. Please check the directory path."
                " If you want to compute new embeddings, please set the "
                "'already_computed' option to False in the config.yaml file."
            )
    else:
        model_names = embedding_model["selected_models"]


def model_specific_embedding_creation(audio_dir, dim_reduction_model, **kwargs):
    """
    Generate embeddings for each model in the list of model names.
    The embeddings are generated using the generate_embeddings function
    from the generate_embeddings module. The embeddings are saved
    in the directory specified by the audio_dir parameter. The
    function returns a dictionary containing the loader objects
    for each model, by which metadata and paths are stored.

    Parameters
    ----------
    audio_dir : string
        full path to audio files
    dim_reduction_model : string
        name of the dimensionality reduction model to be used
        for the embeddings. If "None" is selected, no
        dimensionality reduction is performed.

    Returns
    -------
    loader_dict : dict
        dictionary containing the loader objects for each model
    """
    loader_dict = {}
    for model_name in model_names:
        loader_dict[model_name] = get_embeddings(
            model_name=model_name,
            dim_reduction_model=dim_reduction_model,
            audio_dir=audio_dir,
            **kwargs,
        )
    return loader_dict


def model_specific_evaluation(
    loader_dict, audio_dir, evaluation_task, class_configs, distance_configs, **kwargs
):
    """
    Perform evaluation of the embeddings using the specified
    evaluation task. The evaluation task can be either
    classification, clustering, or pairwise distances.
    The evaluation is performed using the functions from
    the classification, clustering, and distance modules.
    The results of the evaluation are saved in the directory
    specified by the audio_dir parameter. The function
    returns a dictionary containing the paths for the
    results of the evaluation.

    Parameters
    ----------
    loader_dict : dict
        dictionary containing the loader objects for each model
    audio_dir : string
        full path to audio files
    evaluation_task : string
        name of the evaluation task to be performed.
    class_configs : dict
        dictionary containing the configuration for the
        classification tasks. The configurations are specified
        in the bacpipe/settings.yaml file.
    """
    for model_name in model_names:
        if not evaluation_task == "None":
            embeds = loader_dict[model_name].embedding_dict()
            paths = set_paths(audio_dir, model_name, **kwargs)
            ground_truth = ground_truth_by_model(paths, model_name)

        if "classification" in evaluation_task:
            print("\nTraining linear classifier to evaluate embeddings")

            assert len(embeds) > 1, (
                "Too few files to evaluate embeddings with linear classifier. "
                "Are you sure you have selected the right data?"
            )

            generate_annotations_for_classification_task(paths)

            class_embeds = embeds_array_without_noise(embeds, ground_truth)
            for class_config in class_configs.values():
                if class_config["bool"]:
                    classification_pipeline(paths, class_embeds, **class_config)

        if "clustering" in evaluation_task:
            clustering(paths, embeds, ground_truth, remove_noise=False, overwrite=True)

        if "pairwise_distances" in evaluation_task:
            for dist_config in distance_configs.values():
                if dist_config["bool"]:
                    calc_distances(paths, embeds, **dist_config)


def embeds_array_without_noise(embeds, ground_truth):
    return np.concatenate(list(embeds.values()))[ground_truth["labels"] > -1]


def cross_model_evaluation(audio_dir, dim_reduction_model, evaluation_task, **kwargs):
    if len(model_names) > 1:
        if not evaluation_task == "None":
            visualise_classification_results_across_models(evaluation_task, model_names)
        if not dim_reduction_model == "None":
            plot_comparison(
                audio_dir,
                model_names,
                dim_reduction_model,
            )


def visualize_using_dashboard():

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
    plot_overview(processed_embeds, name, no_noise=True)
    plot_clustering_by_metric_new(all_clusts_reordered, reduc_2d_embeds.keys())
    scatterplot_clust_vs_class()


def get_embeddings(
    model_name,
    audio_dir,
    dim_reduction_model="None",
    check_if_primary_combination_exists=True,
    check_if_secondary_combination_exists=True,
    overwrite=False,
    **kwargs,
):
    loader_embeddings = generate_embeddings(
        model_name=model_name,
        audio_dir=audio_dir,
        check_if_combination_exists=check_if_primary_combination_exists,
    )
    paths = set_paths(audio_dir, model_name, **kwargs)
    default_labels = create_default_labels(paths, model_name, audio_dir, **kwargs)

    ground_truth = ground_truth_by_model(
        paths, model_name, label_file="annotations.csv", **kwargs
    )

    if not dim_reduction_model == "None":

        assert len(loader_embeddings.files) > 1, (
            "Too few files to perform dimensionality reduction. "
            + "Are you sure you have selected the right data?"
        )
        loader_dim_reduced = generate_embeddings(
            model_name=model_name,
            dim_reduction_model=dim_reduction_model,
            audio_dir=audio_dir,
            check_if_combination_exists=check_if_secondary_combination_exists,
        )
        if not overwrite and (
            loader_dim_reduced.embed_dir.joinpath("embed.png").exists()
        ):
            logger.debug(
                f"Embedding visualization already exist in {loader_dim_reduced.embed_dir}"
                " Skipping visualization generation."
            )
        else:
            print(
                "### Generating visualizations of embeddings using "
                f"{dim_reduction_model}. Plots are saved in "
                f"{loader_dim_reduced.embed_dir} ###"
            )
            plot_embeddings(
                loader_dim_reduced.embed_dir,
                dim_reduction_model,
                default_labels,
                ground_truth=ground_truth,
                bool_plot_centroids=False,
                label_by="time_of_day",
            )
    return loader_embeddings


def generate_embeddings(save_files=True, **kwargs):
    if "dim_reduction_model" in kwargs:
        print(
            f"\n\n\n###### Generating embeddings using {kwargs['dim_reduction_model'].upper()} ######\n"
        )
    elif "model_name" in kwargs:
        print(
            f"\n\n\n###### Generating embeddings using {kwargs['model_name'].upper()} ######\n"
        )
    else:
        raise ValueError("model_name not provided in kwargs.")
    try:
        start = time.time()
        ld = ge.Loader(**kwargs)
        logger.debug(f"Loading the data took {time.time()-start:.2f}s.")
        if not ld.combination_already_exists:
            embed = ge.Embedder(**kwargs)
            for idx, file in enumerate(
                tqdm(ld.files, desc="processing files", position=1, leave=False)
            ):
                if not ld.dim_reduction_model:
                    sample = file
                    embeddings = embed.get_embeddings_from_model(sample)
                    ld.write_audio_file_to_metadata(idx, file, embed, embeddings)
                    embed.save_embeddings(idx, ld, file, embeddings)
                else:
                    if idx == 0:
                        embeddings = ld.embed_read(idx, file)
                    else:
                        embeddings = np.concatenate(
                            [embeddings, ld.embed_read(idx, file)]
                        )
            if ld.dim_reduction_model:
                dim_reduced_embeddings = embed.get_embeddings_from_model(embeddings)
                embed.save_embeddings(idx, ld, file, dim_reduced_embeddings)
            ld.write_metadata_file()
            ld.update_files()
        return ld
    except KeyboardInterrupt:
        if ld.embed_dir.exists() and ld.rm_embedding_on_keyboard_interrupt:
            print("KeyboardInterrupt: Exiting and deleting created embeddings.")
            import shutil

            shutil.rmtree(ld.embed_dir)
        import sys

        sys.exit()


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

        from exploration.explore_dashboard import plot_overview
        import exploration.explore_dashboard
        import importlib

        importlib.reload(exploration.explore_dashboard)
        from exploration.explore_dashboard import plot_overview

        fig = plot_overview(processed_embeds, name, no_noise=True)
        fig.savefig(plot_path.joinpath(f"{name}_overview_kmeans.png"), dpi=300)

        if remove_noise:
            name += "_no_noise"
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
