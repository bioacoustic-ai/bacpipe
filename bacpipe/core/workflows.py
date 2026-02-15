import time
import logging
from pathlib import Path
import importlib.resources as pkg_resources

from pathlib import Path
from huggingface_hub import hf_hub_download
import tarfile

import numpy as np

from bacpipe.core.experiment_manager import (
    Loader, save_logs
    )
from bacpipe.model_pipelines.runner import Embedder

from bacpipe.embedding_evaluation.visualization.dashboard import (
    visualize_using_dashboard
)

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
from bacpipe.embedding_evaluation.classification.classify import (
    classification_pipeline
    )
from bacpipe.embedding_evaluation.clustering.cluster import clustering

from bacpipe.core.constants import TF_MODELS, NEEDS_CHECKPOINT
from bacpipe import config, settings

logger = logging.getLogger("bacpipe")

def play(config=config, settings=settings, bool_save_logs=False):
    """
    Play the bacpipe! The pipeline will run using the models specified in
    bacpipe.config.models and generate results in the directory
    bacpipe.settings.results_dir. For more details see the ReadMe file on the
    repository page https://github.com/bioacoustic-ai/bacpipe.

    Parameters
    ----------
    config : dict, optional
        configurations for pipeline execution, by default config
    settings : dict, optional
        settings for pipeline execution, by default settings
    bool_save_logs : bool, optional
        Save logs, config and settings file. This is important if you get a bug,
        sharing this will be very helpful to find the source of
        the problem, by default False


    Raises
    ------
    FileNotFoundError
        If no audio files are found we can't compute any embeddings. So make
        sure the path is correct :)
    """
    settings.model_base_path = ensure_models_exist(Path(settings.model_base_path),
                                                   model_names=config.models)
    overwrite, dashboard = config.overwrite, config.dashboard

    if config.audio_dir == 'bacpipe/tests/test_data' or settings.testing:
        with pkg_resources.path(
            __package__.split('.')[0] + ".tests.test_data", ""
            ) as audio_dir:
            audio_dir = Path(audio_dir)

        if not audio_dir.exists():
            error = (
                f"\nAudio directory {config.audio_dir} does not exist. Please check the path. "
                "It should be in the format 'C:\\path\\to\\audio' on Windows or "
                "'/path/to/audio' on Linux/Mac. Use single quotes '!"
            )
            logger.exception(error)
            raise FileNotFoundError(error)
        else:
            config.audio_dir = audio_dir

        # ----------------------------------------------------------------
    # Setup logging to file if requested
    # ----------------------------------------------------------------
    if bool_save_logs:
        save_logs(config, settings)

    config.models = get_model_names(**vars(config), **vars(settings))

    if overwrite or not evaluation_with_settings_already_exists(
        **vars(config), **vars(settings)
    ):

        loader_dict = model_specific_embedding_creation(
            **vars(config), **vars(settings)
        )

        model_specific_evaluation(loader_dict, **vars(config), **vars(settings))

        cross_model_evaluation(**vars(config), **vars(settings))

    if dashboard:
        visualize_using_dashboard(**vars(config), **vars(settings))




def ensure_models_exist(model_base_path, model_names, repo_id="vskode/bacpipe_models"):
    """
    Ensure that the model checkpoints for the selected models are
    available locally. Downloads from Hugging Face Hub if missing.

    Parameters
    ----------
    model_base_path : Path
        Local base directory where the checkpoints should be stored.
    model_names : list
        list of models to run
    repo_id : str, optional
        Hugging Face Hub repo ID, by default "vinikay/bacpipe_models"
    """
    model_base_path = Path(model_base_path)
    model_base_path.parent.mkdir(exist_ok=True, parents=True)
    
    logger.info(
        "Checking if the selected models require a checkpoint, and if so, "
        "if the checkpoint already exists.\n"
    )
    remove_from_list = []
    if 'naturebeats' in model_names and not 'beats' in model_names:
        model_names.append('beats')
        remove_from_list = ['beats']
        
    for model_name in model_names:
        if model_name in NEEDS_CHECKPOINT:
            if ((model_base_path / model_name).exists()
                and len(list((model_base_path / model_name).iterdir())) > 0):
                logger.info(f"{model_name} checkpoint exists.\n")    
                continue
            else:   
                if model_name == 'birdnet':
                    import tensorflow as tf
                    if tf.__version__ == '2.15.1':
                        hf_url = f"{model_name}/{model_name}_tf215.tar.xz"
                    else:
                        hf_url = f"{model_name}/{model_name}.tar.xz"
                else:
                    hf_url = f"{model_name}/{model_name}.tar.xz"
                    
                logger.info(
                    f"{model_name} checkpoint does not exists. "
                    "Downloading the model from "
                    f"https://huggingface.co/datasets/{repo_id}/blob/main/{hf_url}\n"
                    )    
                hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_url,
                    local_dir=model_base_path,
                    repo_type="dataset",
                )
                tar = tarfile.open(model_base_path / hf_url)
                tar.extractall(path=model_base_path)
                tar.close()
                
    [model_names.remove(l) for l in remove_from_list]
    return model_base_path.parent / "model_checkpoints"



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
    remove_models_from_list = []
    for model_name in models:
        try:
            loader_dict[model_name] = run_pipeline_for_model(
                model_name=model_name,
                dim_reduction_model=dim_reduction_model,
                audio_dir=audio_dir,
                **kwargs,
            )
        except AssertionError as e:
            remove_models_from_list.append(model_name)
            if kwargs['already_computed']:
                logger.exception(
                    f"Bacpipe was not able to process {model_name} because {e}. "
                    f"Because `already_computed` is True, it looks like {model_name} "
                    "didn't fully finish on the last run. "
                    "Bacpipe will continue without this model so that the rest of "
                    "the processing can still be completed. "
                    "To ensure this model get's processed, set `already_computed` to False."
                    )
            else:
                logger.exception(
                    f"Bacpipe was not able to process {model_name} because {e}."
                )
    if len(remove_models_from_list) > 0:
        for model in remove_models_from_list:
            models.remove(model)
    return loader_dict

def model_specific_evaluation(
    loader_dict, evaluation_task, class_configs, 
    models, dim_reduction_model=False, **kwargs
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
        if loader_dict[model_name].classifier_should_be_run(paths, **kwargs):
            embed = Embedder(model_name, loader_dict[model_name], **kwargs)
            if hasattr(embed.model, 'classifier_predictions'):
                embed.classifier.run_default_classifier(
                    loader_dict[model_name]
                    )
                
        if not evaluation_task in ["None", [], False]:
            embeds = loader_dict[model_name].embeddings()
            try:
                ground_truth = ground_truth_by_model(paths, model_name, **kwargs)
            except FileNotFoundError as e:
                ground_truth = None

        if "classification" in evaluation_task and not ground_truth is None:
            logger.info(
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
            logger.info(
                "\nGenerating clusterings to evaluate "
                f"{model_name.upper()} embeddings"
            )

            embeds_array = np.concatenate(list(embeds.values()))
            clustering(paths, embeds_array, ground_truth, **kwargs)


def embeds_array_without_noise(embeds, ground_truth, label_column, **kwargs):
    return np.concatenate(list(embeds.values()))[
        ground_truth[f"label:{label_column}"] > -1
    ]


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

        assert len(loader_embeddings.files) > 1, logger.exception(
            "Too few files to perform dimensionality reduction. "
            "Are you sure you have selected the right data? "
            f"Will continue without {model_name}."
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
            logger.info(
                "### Generating visualizations of embeddings using "
                f"{dim_reduction_model}. Plots are saved in "
                f"{loader_dim_reduced.embed_dir} ###"
            )
            vis_loader = EmbedAndLabelLoader(
                dim_reduction_model=dim_reduction_model, **kwargs
            )
            try:
                plot_embeddings(
                    vis_loader,
                    paths=paths,
                    model_name=loader_dim_reduced.model_name,
                    dim_reduction_model=dim_reduction_model,
                    bool_plot_centroids=False,
                    label_by="time_of_day",
                    **kwargs,
                )
            except AssertionError as e:
                logger.exception(
                    "Plotting of embeddings has failed. Continuing with processing "
                    f"embeddings, but this will cause evaluation problems later on. {e}"
                )
                
    return loader_embeddings


def generate_embeddings(avoid_pipelined_gpu_inference=False, **kwargs):
    if "dim_reduction_model" in kwargs:
        logger.info(
            f"\n\n\n###### Generating embeddings using {kwargs['dim_reduction_model'].upper()} ######\n"
        )
    elif "model_name" in kwargs:
        logger.info(
            f"\n\n\n###### Generating embeddings using {kwargs['model_name'].upper()} ######\n"
        )
    else:
        error = "\nmodel_name not provided in kwargs."
        logger.exception(error)
        raise ValueError(error)
    try:
        start = time.time()
        ld = Loader(**kwargs)
        logger.debug(f"Loading the data took {time.time()-start:.2f}s.")
        if not ld.combination_already_exists:
            embed = Embedder(loader=ld, **kwargs)

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
            all_files = list(Path(ld.embed_dir).rglob('*'))
            if len(all_files) < 25:
                logger.info(f"KeyboardInterrupt: Exiting and deleting created {ld.embed_dir}.")
                import shutil

                shutil.rmtree(ld.embed_dir)
            else:
                logger.info(f"KeyboardInterrupt: Exiting and but not deleting {ld.embed_dir}.")
        import sys

        sys.exit()