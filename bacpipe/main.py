import time
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

import bacpipe.generate_embeddings as ge

from bacpipe.embedding_evaluation.visualization.visualize import (
    plot_comparison,
    plot_embeddings,
    visualise_results_across_models,
    EmbedAndLabelLoader,
)
from bacpipe.embedding_evaluation.label_embeddings import (
    make_set_paths_func,
    generate_annotations_for_classification_task,
    ground_truth_by_model,
)

from bacpipe.embedding_evaluation.classification.classify import classification_pipeline

from bacpipe.embedding_evaluation.clustering.cluster import clustering

from bacpipe.embedding_evaluation.distance_evalutation.distances import (
    calc_distances,
)

from bacpipe.embedding_evaluation.visualization.dashboard import DashBoard

logger = logging.getLogger("bacpipe")


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
    loader_dict, evaluation_task, class_configs, distance_configs, **kwargs
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
    evaluation_task : string
        name of the evaluation task to be performed.
    class_configs : dict
        dictionary containing the configuration for the
        classification tasks. The configurations are specified
        in the bacpipe/settings.yaml file.
    distance_configs : dict
        dictionary to specify which distance calculations to perform
    """
    for model_name in model_names:
        if not evaluation_task in ["None", []]:
            embeds = loader_dict[model_name].embedding_dict()
            paths = get_paths(model_name)
            ground_truth = ground_truth_by_model(paths, model_name, **kwargs)

        if "classification" in evaluation_task:
            print(
                "\nTraining classifier to evaluate " f"{model_name.upper()} embeddings"
            )

            assert len(embeds) > 1, (
                "Too few files to evaluate embeddings with classifier. "
                "Are you sure you have selected the right data?"
            )

            generate_annotations_for_classification_task(paths)

            class_embeds = embeds_array_without_noise(embeds, ground_truth)
            for class_config in class_configs.values():
                if class_config["bool"]:
                    classification_pipeline(
                        paths, class_embeds, **class_config, **kwargs
                    )

        if "clustering" in evaluation_task:
            print(
                "\nGenerating clusterings to evaluate "
                f"{model_name.upper()} embeddings"
            )

            embeds_array = np.concatenate(list(embeds.values()))
            clustering(paths, embeds_array, ground_truth, **kwargs)

        if "pairwise_distances" in evaluation_task:
            for dist_config in distance_configs.values():
                if dist_config["bool"]:
                    calc_distances(paths, embeds, **dist_config)


def cross_model_evaluation(
    dim_reduction_model, evaluation_task, dashboard=False, **kwargs
):
    """
    Generate plots to compare models by the specified tasks.

    Parameters
    ----------
    dim_reduction_model : str
        name of dimensionality reduction model
    evaluation_task : list
        tasks to evaluate models by
    """
    if len(model_names) > 1:
        plot_path = get_paths(model_names[0]).plot_path.parent.parent.joinpath(
            "overview"
        )
        plot_path.mkdir(exist_ok=True, parents=True)
        if not len(evaluation_task) == 0:
            for task in evaluation_task:
                visualise_results_across_models(plot_path, task, model_names)
        if not dim_reduction_model == "None":
            plot_comparison(
                plot_path,
                model_names,
                dim_reduction_model,
                label_by="time_of_day",
                dashboard=False,
                **kwargs,
            )


def embeds_array_without_noise(embeds, ground_truth):
    return np.concatenate(list(embeds.values()))[ground_truth["labels"] > -1]


def visualize_using_dashboard(**kwargs):
    dashboard = DashBoard(model_names, **kwargs)
    dashboard.build_layout()

    dashboard.app.servable()


def get_embeddings(
    model_name,
    audio_dir,
    dim_reduction_model="None",
    check_if_primary_combination_exists=True,
    check_if_secondary_combination_exists=True,
    overwrite=False,
    testing=False,
    **kwargs,
):
    loader_embeddings = generate_embeddings(
        model_name=model_name,
        audio_dir=audio_dir,
        check_if_combination_exists=check_if_primary_combination_exists,
        testing=testing,
        **kwargs,
    )
    global get_paths
    get_paths = make_set_paths_func(audio_dir, testing=testing, **kwargs)
    paths = get_paths(model_name)

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
            testing=testing,
            **kwargs,
        )
        if (
            not overwrite
            and (paths.plot_path.joinpath("embeddings.png").exists())
            or testing
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
            vis_loader = EmbedAndLabelLoader(
                dim_reduction_model=dim_reduction_model, **kwargs
            )
            plot_embeddings(
                vis_loader,
                paths=paths,
                model_name=loader_dim_reduced.model_name,
                dim_reduction_model=dim_reduction_model,
                bool_plot_centroids=False,
                label_by="time_of_day",
                **kwargs,
            )
    return loader_embeddings


def generate_embeddings(**kwargs):
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
                    try:
                        embeddings = embed.get_embeddings_from_model(sample)
                    except Exception as e:
                        logger.warning(
                            f"Error generating embeddings, skipping file. \n"
                            f"Error: {e}"
                        )
                        continue
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
