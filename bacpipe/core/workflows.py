import time
import logging
from pathlib import Path
import importlib.resources as pkg_resources

from pathlib import Path
from huggingface_hub import hf_hub_download
import tarfile

import numpy as np

from bacpipe.core.experiment_manager import (
    Loader, save_logs, replace_default_kwargs_with_user_kwargs
    )
from bacpipe.model_pipelines.runner import Embedder

from bacpipe.embedding_evaluation.visualization.dashboard import (
    visualize_using_dashboard
)

from bacpipe.embedding_evaluation.visualization.visualize import (
    visualise_results_across_models,
)
from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    plot_comparison,
    plot_embeddings,
    EmbedAndLabelLoader,
)
from bacpipe.embedding_evaluation.label_embeddings import (
    make_set_paths_func,
    ground_truth_by_model,
)
from bacpipe.embedding_evaluation.probing.probe import (
    probing_pipeline
    )
from bacpipe.embedding_evaluation.clustering.cluster import clustering_pipeline

from bacpipe.core.constants import TF_MODELS, NEEDS_CHECKPOINT
from bacpipe import config, settings

logger = logging.getLogger("bacpipe")

def play(bool_save_logs=False, **kwargs):
    """
    Play the bacpipe! The pipeline will run using the models specified in
    bacpipe.config.models and generate results in the directory
    bacpipe.settings.results_dir. For more details see the ReadMe file on the
    repository page https://github.com/bioacoustic-ai/bacpipe or the documentation
    under https://bacpipe.readthedocs.io/en/latest/.

    Parameters
    ----------
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
    kwargs = replace_default_kwargs_with_user_kwargs(**kwargs)
    
    kwargs['model_base_path'] = ensure_models_exist(
        Path(kwargs.get('model_base_path')), 
        model_names=kwargs.get('models')
        )
    overwrite, dashboard = kwargs.get('overwrite'), kwargs.get('dashboard')

    if kwargs.get('audio_dir') == 'bacpipe/tests/test_data' or kwargs.get('testing'):
        with pkg_resources.path(
            __package__.split('.')[0] + ".tests.test_data", ""
            ) as audio_dir:
            audio_dir = Path(audio_dir)

        if not audio_dir.exists():
            error = (
                f"\nAudio directory {kwargs.get('audio_dir')} does not exist. Please check the path. "
                "It should be in the format 'C:\\path\\to\\audio' on Windows or "
                "'/path/to/audio' on Linux/Mac. Use single quotes '!"
            )
            logger.exception(error)
            raise FileNotFoundError(error)
        else:
            kwargs['audio_dir'] = audio_dir

        # ----------------------------------------------------------------
    # Setup logging to file if requested
    # ----------------------------------------------------------------
    if bool_save_logs:
        save_logs()

    kwargs['models'] = get_model_names(**kwargs)

    if overwrite or not evaluation_with_settings_already_exists(**kwargs):

        loader_dict = run_pipeline_for_models(**kwargs)

        model_specific_evaluation(loader_dict, **kwargs)

        cross_model_evaluation(**kwargs)

    if dashboard:
        visualize_using_dashboard(**kwargs)




def ensure_models_exist(model_base_path, model_names, repo_id="vskode/bacpipe_models"):
    """
    Ensure that the model checkpoints for the selected models are
    available locally. Downloads from Hugging Face Hub if missing.

    Parameters
    ----------
    model_base_path : Path
        Local base directory where the checkpoints should be stored.
    model_names : str or list
        Model name or list of model names to run
    repo_id : str, optional
        Hugging Face Hub repo ID, by default "vinikay/bacpipe_models"

    Returns
    -------
    str
        path to saved models
    """
    if isinstance(model_names, str):
        model_names = [model_names]

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
    Get the names of the models used for processing. This is either done
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
    probing and clustering evaluation results
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
            and paths.probe_path.exists()
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


def run_pipeline_for_models(
    models, 
    audio_dir, 
    dim_reduction_model, 
    **kwargs
    ):
    """
    Generate embeddings for each model in the list of model names.
    The embeddings are generated using the generate_embeddings function
    from the generate_embeddings module. The embeddings are saved
    in the directory specified by the audio_dir parameter. The
    function returns a dictionary containing the loader objects
    for each model, by which metadata and paths are stored.
    kwargs that are not specifically passed will be taken from 
    bacpipe.config and bacpipe.settings.
    
        
    code example:
    ```
    loader = bacpipe.run_pipeline_for_models(
        models=['birdnet', 'naturebeats'],
        audio_dir='bacpipe/tests/test_data',
        dim_reduction_model='umap'
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
    models : list
        embedding models
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
    remove_models_from_list = []
    if 'CustomModels' in kwargs:
        assert (len(kwargs['CustomModels']) == len(models)) , (
            "If you provide custom models, the array needs to be the "
            "same length as the model name array. That way the association "
            "is clear. \n For example: models = ['birdnet', 'perch_v2', 'my_model] "
            "and CustomModels=[None, None, MyModel]. That way for models 0 and 1 "
            "the integrated models are loaded and for my_model the model class "
            "MyModel is loaded."
            )
        CustomModels = kwargs.pop('CustomModels')
    else:
        CustomModels = [None] * len(models)
    for idx, model_name in enumerate(models):
        try:
            loader_dict[model_name] = run_pipeline_for_single_model(
                model_name=model_name,
                dim_reduction_model=dim_reduction_model,
                audio_dir=audio_dir,
                CustomModel=CustomModels[idx],
                **kwargs,
            )
        except AssertionError as e:
            remove_models_from_list.append(model_name)
            if not 'already_computed' in kwargs:
                from bacpipe import config
                kwargs['already_computed'] = config.already_computed
            kwargs
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
    loader_dict, evaluation_task, probe_configs, 
    models, dim_reduction_model=False, **kwargs
):
    """
    Perform evaluation of the embeddings using the specified
    evaluation task. The evaluation task can be either
    probing or clustering.
    The evaluation is performed using the functions from
    the probing and clustering modules.
    The results of the evaluation are saved in the directory
    specified by the audio_dir parameter. 

    Parameters
    ----------
    loader_dict : dict
        dictionary containing the loader objects for each model
    evaluation_task : string
        name of the evaluation task to be performed.
    probe_configs : dict
        dictionary containing the configuration for the
        probing tasks. The configurations are specified
        in the bacpipe/settings.yaml file.
    models : list
        embedding models
    """
    if 'CustomModels' in kwargs:
        assert (len(kwargs['CustomModels']) == len(models)) , (
            "If you provide custom models, the array needs to be the "
            "same length as the model name array. That way the association "
            "is clear. \n For example: models = ['birdnet', 'perch_v2', 'my_model] "
            "and CustomModels=[None, None, MyModel]. That way for models 0 and 1 "
            "the integrated models are loaded and for my_model the model class "
            "MyModel is loaded."
            )
        CustomModels = kwargs.pop('CustomModels')
    else:
        CustomModels = [None] * len(models)
    ensure_models_exist(settings.model_base_path, models)
    
    for idx, model_name in enumerate(models):
        paths = get_paths(model_name)
        if loader_dict[model_name].classifier_should_be_run(paths, **kwargs):
            embed = Embedder(
                model_name, 
                loader_dict[model_name], 
                CustomModel=CustomModels[idx],
                **kwargs
                )
            if hasattr(embed.model, 'classifier_predictions'):
                embed.classifier.run_default_classifier(
                    loader_dict[model_name]
                    )
                
        # if not evaluation_task in ["None", [], None, False]:
        embeds = loader_dict[model_name].embeddings()
        try:
            ground_truth = ground_truth_by_model(
                model_name, paths=paths, single_label=True, **kwargs
                )
        except FileNotFoundError as e:
            logger.exception(
                f"unable to process ground truth, {e}"
            )
            ground_truth = None
        except IndexError as e:
            logger.exception(
                f"unable to process ground truth, {e}"
            )
            ground_truth = None



        ####################################################################
        ############      PROBING OF EMBEDDINGS THROUGH       ##############
        ############      LINEAR AND KNN CLASSIFICATION       ##############
        ############            SEE SETTINGS.YAML             ##############
        ####################################################################
        
        if "probing" in evaluation_task and not ground_truth is None:
            logger.info(
                "\nTraining probe to evaluate " f"{model_name.upper()} embeddings"
            )
            
            assert len(embeds) > 1, (
                "Too few files to evaluate embeddings with probing. "
                "Are you sure you have selected the right data?"
            )

            for class_config in probe_configs.values():
                if class_config["bool"]:
                    probing_pipeline(
                        model_name, 
                        ground_truth, embeds, 
                        paths, **class_config, **kwargs
                    )

        ####################################################################
        ############      CLUSTERING OF EMBEDDINGS THROUGH    ##############
        ######      KMEANS (AND WHATEVER SPECIFIED IN SETTINGS.YAML)   #####
        ####################################################################
                    
        if "clustering" in evaluation_task:
            logger.info(
                "\nGenerating clusterings to evaluate "
                f"{model_name.upper()} embeddings"
            )

            embeds_array = np.concatenate(list(embeds.values()))
            clustering_pipeline(model_name, ground_truth, embeds_array, paths, **kwargs)

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
            if "audio_dir" in kwargs: kwargs.pop("audio_dir")
            plot_comparison(
                plot_path,
                models,
                dim_reduction_model,
                label_by="time_of_day",
                dashboard=False,
                **kwargs,
            )


def run_pipeline_for_single_model(
    model_name,
    audio_dir,
    dim_reduction_model="None",
    check_if_already_processed=True,
    check_if_already_dim_reduced=True,
    testing=False,
    **kwargs,
):
    """
    Run the bacpipe pipeline, including embedding generation, classification
    using the pretrained classifier (if included), dimensionality reduction (if passed),
    and plotting of visualization to files. 
    All of this will be done for one model. The predefined folder structure will be created
    so that subsequent processing runs will be very fast, as they then only load the data. 
    kwargs that are not specifically passed will be taken from 
    bacpipe.config and bacpipe.settings.

    Parameters
    ----------
    model_name : string
        model name
    audio_dir : str
        path to audio data
    dim_reduction_model : str, optional
        name of dimensionality reduction model, by default "None"
    check_if_already_processed : bool, optional
        set to False if you want to force recomputing 
        of embeddings, by default True
    check_if_already_dim_reduced : bool, optional
        set to False if you want to force recomputing of 
        dimensionality reduced embeddings, by default True
    overwrite : bool, optional
        set to True if you want default labels and 
        ground truth labels to be processed again, by default False
    testing : bool, optional
        set to True for testing, by default False

    Returns
    -------
    bacpipe.Loader
        object to processed embeddings and classifier predictions
    """
    kwargs = replace_default_kwargs_with_user_kwargs(
        remove_keys=['audio_dir', 'dim_reduction_model', 'testing'], 
        **kwargs
        )
    global get_paths
    get_paths = make_set_paths_func(audio_dir, testing=testing, **kwargs)
    paths = get_paths(model_name)

    loader_embeddings = generate_embeddings(
        model_name=model_name,
        audio_dir=audio_dir,
        check_if_combination_exists=check_if_already_processed,
        paths=paths,
        testing=testing,
        **kwargs,
    )

    if not dim_reduction_model in ["None", False]:

        loader_dim_reduced = generate_embeddings(
            model_name=model_name,
            dim_reduction_model=dim_reduction_model,
            audio_dir=audio_dir,
            check_if_combination_exists=check_if_already_dim_reduced,
            testing=testing,
            **kwargs,
        )
        if (
            not (paths.plot_path.joinpath("embeddings.png").exists())
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
    """
    Run the embedding generation pipeline including classification
    using the pretrained classifier (if included).
    All of this will be done for one model. The predefined folder structure will be created
    so that subsequent processing runs will be very fast, as they then only load the data. 
    kwargs that are not specifically passed will be taken from 
    bacpipe.config and bacpipe.settings.


    Parameters
    ----------
    avoid_pipelined_gpu_inference : bool, optional
        set to True to avoid multiprocessing, by default False

    Returns
    -------
    bacpipe.Loader
        loader object to access embeddings and classifier predictions

    Raises
    ------
    ValueError
        if not model name is provided
    """
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
        ld = Loader(use_folder_structure=True, **kwargs)
        logger.debug(f"Loading the data took {time.time()-start:.2f}s.")
        if not ld.combination_already_exists:
            embed = Embedder(loader=ld, **kwargs)

            if ld.dim_reduction_model:
                # (1) Dimensionality reduction stage
                embed.run_dimensionality_reduction_pipeline()

            elif not avoid_pipelined_gpu_inference:
                # (2) pipelined embedding generation
                embed.run_inference_pipeline_using_multithreading()

            else:
                # (3) sequential embedding generation
                embed.run_inference_pipeline_sequentially()

            # Finalize
            if embed.model.bool_classifier and not embed.dim_reduction_model:
                try:
                    embed.classifier.save_annotation_table(ld, **kwargs)
                except Exception as e:
                    logger.warning(
                        "Error when trying to save classifier predictions. "
                        f"Continuing but only embeddings will be saved. {e}"
                    )
            ld.write_metadata_file()
            ld.update_files()
        
            # clear GPU
            del embed
            
            if kwargs['model_name'] in TF_MODELS:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                
        elif hasattr(kwargs, 'paths') and ld.classifier_should_be_run(**kwargs):
            embed = Embedder(loader=ld, **kwargs)
            if hasattr(embed.model, 'classifier_predictions'):
                embed.classifier.run_default_classifier(ld)
        return ld
    except KeyboardInterrupt:
        try:
            if ld.embed_dir.exists() and ld.rm_embedding_on_keyboard_interrupt:
                all_files = list(Path(ld.embed_dir).rglob('*'))
                if len(all_files) < 15:
                    logger.info(f"KeyboardInterrupt: Exiting and deleting created {ld.embed_dir}.")
                    import shutil

                    shutil.rmtree(ld.embed_dir)
                else:
                    logger.info(f"KeyboardInterrupt: Exiting but not deleting {ld.embed_dir}.")
        except NameError:
            logger.info("Bacpipe exiting.")
        import sys
        sys.exit()
