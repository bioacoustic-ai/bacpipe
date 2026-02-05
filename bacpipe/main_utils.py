import os
import time
import logging
from pathlib import Path
import importlib.resources as pkg_resources
import bacpipe.imgs
import soundfile

from bacpipe.main import Embedder

import numpy as np
from tqdm import tqdm

import bacpipe.main as ge

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


from bacpipe.embedding_evaluation.visualization.dashboard import DashBoard

from bacpipe import TF_MODELS

logger = logging.getLogger("bacpipe")


def get_model_names(
    models,
    audio_dir,
    main_results_dir,
    embed_parent_dir,
    already_computed=False,
    **kwargs,
):
    """
    Get the names of the models used for embedding. This is either done
    by using already computed embeddings or by using the selected models
    from the config file. If already computed embeddings are used, the
    model names are extracted from the directory structure.

    Parameters
    ----------
    models : list
        list of embedding models
    audio_dir : string
        full path to audio files
    main_results_dir : string
        top level directory for the results of the embedding evaluation
    embed_parent_dir : string
        parent directory for the embeddings
    already_computed : bool, Default is False
        ignore model list and use only models whos embeddings already have
        been computed and are saved in the results dir

    Raises
    ------
    ValueError
        If already computed embeddings are used, but no embeddings
        are found in the specified directory.
    """
    if already_computed:

        dataset_name = Path(audio_dir).stem
        main_results_path = (
            Path(main_results_dir).joinpath(dataset_name).joinpath(embed_parent_dir)
        )
        model_names = [
            d.stem.split("___")[-1].split("-")[0]
            for d in list(main_results_path.glob("*"))
            if d.is_dir()
        ]
        if not model_names:
            error = (
                "\nNo embedding models found in the specified directory. "
                "You have selected the option to use already computed embeddings, "
                "but no embeddings were found. Please check the directory path."
                " If you want to compute new embeddings, please set the "
                "'already_computed' option to False in the config.yaml file."
            )
            logger.exception(error)
            raise ValueError(error)
        else:
            return np.unique(model_names).tolist()
    else:
        return models


def evaluation_with_settings_already_exists(
    audio_dir,
    dim_reduction_model,
    models,
    testing=False,
    **kwargs,
):
    """
    Check if the evaluation with the specified settings already exists.
    The function checks if the embeddings, dimensionality reduction,
    classification and clustering evaluation results
    already exist in the specified directory. If any of these
    results do not exist, the function returns False. Otherwise,
    it returns True.

    Parameters
    ----------
    audio_dir : string
        full path to audio files
    dim_reduction_model : string
        name of the dimensionality reduction model to be used
    models : list
        embedding models

    Returns
    -------
    bool
        True if the evaluation with the specified settings
    """
    if testing:
        return False
    for model_name in models:
        paths = make_set_paths_func(audio_dir, **kwargs)(model_name)
        bool_paths = (
            paths.main_embeds_path.exists()
            and paths.dim_reduc_parent_dir.exists()
            and paths.class_path.exists()
            and paths.clust_path.exists()
        )
        if not bool_paths:
            return False
        else:
            bool_dim_reducs = [
                True
                for d in paths.dim_reduc_parent_dir.rglob(
                    f"*{dim_reduction_model}*{model_name}*"
                )
            ]
            bool_dim_reducs = len(bool_dim_reducs) > 0 and all(bool_dim_reducs)
        if not bool_dim_reducs:
            return False
    return True


def model_specific_embedding_creation(audio_dir, dim_reduction_model, models, **kwargs):
    """
    Generate embeddings for each model in the list of model names.
    The embeddings are generated using the generate_embeddings function
    from the generate_embeddings module. The embeddings are saved
    in the directory specified by the audio_dir parameter. The
    function returns a dictionary containing the loader objects
    for each model, by which metadata and paths are stored.
    
        
    code example:
    ```
    loader = bacpipe.model_specific_embedding_creation(
    **vars(bacpipe.config), **vars(bacpipe.settings)
    )

    # this call will initiate the embedding generation process, it will check if embeddings
    # already exist for the combination of each model and the dataset and if so it will
    # be ready to load them. The loader keys will be the model name and the values will
    # be the loader objects for each model. Each object contains all the information
    # on the generated embeddings. To name access them:
    loader['birdnet'].embeddings() 
    # this will give you a dictionary with the keys corresponding to embedding files
    # and the values corresponding to the embeddings as numpy arrays

    loader['birdnet'].metadata_dict
    # This will give you a dictionary overview of:
    # - where the audio data came from,
    # - where the embeddings were saved
    # - all the audio files, 
    # - the embedding size of the model, 
    # - the audio file lengths,
    # - the number of embeddings for each audio files
    # - the sample rate
    # - the number of samples per window
    # - and the total length of the processed dataset in seconds
    # Thic dictionary is also saved as a yaml file in the directory of the embeddings
    ```

    Parameters
    ----------
    audio_dir : string
        full path to audio files
    dim_reduction_model : string
        name of the dimensionality reduction model to be used
        for the embeddings. If "None" is selected, no
        dimensionality reduction is performed.
    models : list
        embedding models

    Returns
    -------
    loader_dict : dict
        dictionary containing the loader objects for each model
    """
    loader_dict = {}
    for model_name in models:
        loader_dict[model_name] = run_pipeline_for_model(
            model_name=model_name,
            dim_reduction_model=dim_reduction_model,
            audio_dir=audio_dir,
            **kwargs,
        )
    return loader_dict

def model_specific_evaluation(
    loader_dict, evaluation_task, class_configs, models, **kwargs
):
    """
    Perform evaluation of the embeddings using the specified
    evaluation task. The evaluation task can be either
    classification or clustering.
    The evaluation is performed using the functions from
    the classification and clustering modules.
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
    models : list
        embedding models
    """
    for model_name in models:
        paths = get_paths(model_name)
        loader_dict[model_name].check_if_default_clfier_should_be_run(
            paths, **kwargs
            )
                
        if not evaluation_task in ["None", [], False]:
            embeds = loader_dict[model_name].get_embeddings()
            try:
                ground_truth = ground_truth_by_model(paths, model_name, **kwargs)
            except FileNotFoundError as e:
                ground_truth = None

        if "classification" in evaluation_task and not ground_truth is None:
            print(
                "\nTraining classifier to evaluate " f"{model_name.upper()} embeddings"
            )

            assert len(embeds) > 1, (
                "Too few files to evaluate embeddings with classifier. "
                "Are you sure you have selected the right data?"
            )

            generate_annotations_for_classification_task(paths, **kwargs)

            class_embeds = embeds_array_without_noise(embeds, ground_truth, **kwargs)
            for class_config in class_configs.values():
                if class_config["bool"]:
                    if not len(class_embeds) > 0:
                        error = (
                            "\nNo embeddings were found for classification task. "
                            "Are you sure there are annotations for the data and the annotations.csv file "
                            "has been correctly linked? If you didn't intent do do classification, "
                            "simply remove it from the evaluation tasks list in the config.yaml file."
                        )
                        logger.exception(error)
                        raise AssertionError(error)
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



def cross_model_evaluation(dim_reduction_model, evaluation_task, models, **kwargs):
    """
    Generate plots to compare models by the specified tasks.

    Parameters
    ----------
    dim_reduction_model : str
        name of dimensionality reduction model
    evaluation_task : list
        tasks to evaluate models by
    models : list
        embedding models
    """
    if len(models) > 1:
        plot_path = get_paths(models[0]).plot_path.parent.parent.joinpath("overview")
        plot_path.mkdir(exist_ok=True, parents=True)
        if not len(evaluation_task) == 0:
            for task in evaluation_task:
                visualise_results_across_models(plot_path, task, models)
        if not dim_reduction_model == "None":
            kwargs.pop("dashboard")
            plot_comparison(
                plot_path,
                models,
                dim_reduction_model,
                label_by="time_of_day",
                dashboard=False,
                **kwargs,
            )


def embeds_array_without_noise(embeds, ground_truth, label_column, **kwargs):
    return np.concatenate(list(embeds.values()))[
        ground_truth[f"label:{label_column}"] > -1
    ]


def visualize_using_dashboard(models, **kwargs):
    """
    Create and serve the dashboard for visualization.

    Parameters
    ----------
    models : list
        embedding models
    kwargs : dict
        Dictionary with parameters for dashboard creation
    """
    from bacpipe.embedding_evaluation.visualization.dashboard import DashBoard
    import panel as pn

    # Configure dashboard
    dashboard = DashBoard(models, **kwargs)

    # Build the dashboard layout
    try:
        dashboard.build_layout()
    except Exception as e:
        logger.exception(
            f"\nError building dashboard layout: {e}\n \n "
            "Are you sure all the evaluations have been performed? "
            "If not, rerun the pipeline with `overwrite=True`.\n \n "
        )
        raise e

    with pkg_resources.path(bacpipe.imgs, 'bacpipe_favicon_white.png') as p:
        favicon_path = str(p)

    template = pn.template.BootstrapTemplate(
        site="bacpipe dashboard",
        title="Explore embeddings of audio data",
        favicon=str(favicon_path),  # must be a path ending in .ico, .png, etc.
        main=[dashboard.app],
    )
    

    template.show(port=5006, address="localhost")


def run_pipeline_for_model(
    model_name,
    audio_dir,
    dim_reduction_model="None",
    check_if_primary_combination_exists=True,
    check_if_secondary_combination_exists=True,
    overwrite=False,
    testing=False,
    **kwargs,
):
    global get_paths
    get_paths = make_set_paths_func(audio_dir, testing=testing, **kwargs)
    paths = get_paths(model_name)

    loader_embeddings = generate_embeddings(
        model_name=model_name,
        audio_dir=audio_dir,
        check_if_combination_exists=check_if_primary_combination_exists,
        paths=paths,
        testing=testing,
        **kwargs,
    )

    if not dim_reduction_model in ["None", False]:

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


def generate_embeddings(avoid_pipelined_gpu_inference=False, **kwargs):
    if "dim_reduction_model" in kwargs:
        print(
            f"\n\n\n###### Generating embeddings using {kwargs['dim_reduction_model'].upper()} ######\n"
        )
    elif "model_name" in kwargs:
        print(
            f"\n\n\n###### Generating embeddings using {kwargs['model_name'].upper()} ######\n"
        )
    else:
        error = "\nmodel_name not provided in kwargs."
        logger.exception(error)
        raise ValueError(error)
    try:
        start = time.time()
        ld = ge.Loader(**kwargs)
        logger.debug(f"Loading the data took {time.time()-start:.2f}s.")
        if not ld.combination_already_exists:
            embed = ge.Embedder(loader=ld, **kwargs)

            if ld.dim_reduction_model:
                # (1) Dimensionality reduction stage
                embed.run_dimensionality_reduction_pipeline()

            elif embed.model.device == "cuda" and not avoid_pipelined_gpu_inference:
                # (2) GPU path with pipelined embedding generation
                embed.run_inference_pipeline_using_multithreading()

            else:
                # (3) CPU path with sequential embedding generation
                embed.run_inference_pipeline_sequentially()

            # Finalize
            if embed.model.bool_classifier and not embed.dim_reduction_model:
                embed.classifier.save_annotation_table(ld)
            ld.write_metadata_file()
            ld.update_files()
        
            # clear GPU
            del embed
            
            if kwargs['model_name'] in TF_MODELS:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            
        return ld
    except KeyboardInterrupt:
        if ld.embed_dir.exists() and ld.rm_embedding_on_keyboard_interrupt:
            print("KeyboardInterrupt: Exiting and deleting created embeddings.")
            import shutil

            shutil.rmtree(ld.embed_dir)
        import sys

        sys.exit()
