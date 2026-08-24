import logging
import json
import numpy as np
import torch

import bacpipe

logger = logging.getLogger(__name__)

from .train_probe import train_probe
from .evaluate_probe import eval_probe
from .dataset_probe import generate_annotations_for_probing_task
from bacpipe.embedding_evaluation.visualization.visualize_embeddings import (
    get_boolean_array_for_annotated_embeddings
    )


def embeds_array_where_single_label(embeds, ground_truth, bool_noise, df, **kwargs):
    """
    Filter the embeddings to only include the segments that have exactly
    one label.

    Parameters
    ----------
    embeds : np.array
        embeddings
    ground_truth : pandas.DataFrame
        ground truth dataframe with the simultaneous labels
    bool_noise : np.array
        boolean array marking the segments that are noise
    df : pandas.DataFrame
        classification dataframe with a predefined_set column

    Returns
    -------
    pandas.DataFrame
        classification dataframe filtered to the probing sets
    np.array
        embeddings filtered to the probing sets
    """
    # first extract the segments that have annotations
    ground_truth = ground_truth[ground_truth.simultaneous_labels > 0]
    
    # now get the segments that have exactly one label
    bool_single_label = (ground_truth.simultaneous_labels == 1).values
    
    bool_array_probing = df.predefined_set.isin(
        ["train", "val", "test"]
        ).values

    df = df[bool_array_probing]
    df.index = range(len(df))

    if isinstance(embeds, np.ndarray):
        embeds = embeds[~bool_noise]
        embeds = embeds[bool_single_label]
        return df, embeds[bool_array_probing]


def probing_pipeline(
    model_name,
    ground_truth,
    embeds,
    paths=None,
    name="linear",
    overwrite=True,
    label_column=bacpipe.settings.label_column,
    dataset_csv_path="probing_dataframe.csv",
    **kwargs,
):
    """
    Probing pipeline consisting of building the classifier,
    evaluating it and saving metrics and plots of performance.

    Examples::
    
        # Run (or load, if ``overwrite=False``) the linear probing evaluation
        # for the already computed ``birdnet`` embeddings:

        loader = bacpipe.Loader(
            'bacpipe/tests/test_data',
            model_name='birdnet',
            use_folder_structure=True,
        )
        
        embeds = loader.embeddings(return_type='array')
        
        gt = bacpipe.ground_truth_by_model(
            model='birdnet',
            audio_dir='bacpipe/tests/test_data',
            main_results_dir='bacpipe_results',
            overwrite=False,
        )
        
        probe, label2index, metrics = bacpipe.probing_pipeline(
            model_name='birdnet',
            ground_truth=gt,
            embeds=embeds,
            name='linear',
            overwrite=False,
            audio_dir='bacpipe/tests/test_data',
            main_results_dir='bacpipe_results',
        )

    Parameters
    ----------
    model_name : str
        name of the model
    ground_truth : pandas.DataFrame
        ground truth dataframe
    paths : SimpleNamespace object
        dict with attributes corresponding to paths for loading and saving
    embeds : np.array
        embeddings
    name : string
        Type of Probing
    label_column : str
        name of the label column
    dataset_csv_path : string
        name of Probing dataframe as specified in settings.yaml
    overwrite : bool
        overwrite existing Probing?, defaults to False
    """
    kwargs = {**vars(bacpipe.settings), **kwargs}
    kwargs.pop("label_column", None)
    if not paths:
        get_paths_func = bacpipe.make_set_paths_func(
            kwargs.get("audio_dir", bacpipe.config.audio_dir),
            kwargs.get("main_results_dir", bacpipe.settings.main_results_dir),
        )
        paths = get_paths_func(model_name)

    df = generate_annotations_for_probing_task(
        ground_truth,
        paths,
        label_column=label_column,
        dataset_csv_path=dataset_csv_path,
        **kwargs,
    )

    if (
        overwrite
        or name == "knn"
        or not paths.probe_path.joinpath(f"probe_results_{name}.json").exists()
    ):
        if len(df) == 0:
            logger.exception(
                "\nNot enough data in annotations to perform probing task\n"
            )
            return None

        bool_noise = get_boolean_array_for_annotated_embeddings(
            ground_truth, model_name, overwrite=overwrite, **kwargs
            )
        df, embeds = embeds_array_where_single_label(
            embeds, ground_truth, bool_noise, df, **kwargs
        )
        if not len(df) == embeds.shape[0]:
            error = (
                "\nYour embeddings and ground truth dataframe ('probing_dataframe.csv') "
                "have different lengths and are therefore incompatible. This could be the "
                "case for multiple reasons, the most likely one is that the file was created "
                "when `only_embed_annotations` was `True` and now it's false, or vice versa. "
                "This error can be fixed by setting `overwrite` to `True` and deleting the "
                "existing 'probing_dataframe.csv'. \n"
            )
            logger.exception(error)
            raise AttributeError(error)

        if not len(embeds) > 0:
            error = (
                "\nNo embeddings were found for classification task. "
                "Are you sure there are annotations for the data and the annotations.csv file "
                "has been correctly linked? If you didn't intent do do classification, "
                "simply remove it from the evaluation tasks list in the config.yaml file.\n"
            )
            logger.exception(error)
            raise AssertionError(error)

        label2index = {label: i for i, label in enumerate(df.label.unique())}

        probe = train_probe(embeds, df, label2index, config=name, **kwargs)

        metrics = eval_probe(
            probe, embeds, df, label2index, config=name, paths=paths, **kwargs
        )

    else:
        logger.info(
            "\nClassification file probe_results_{}.json already exists and"
            " so is not computed. If you want to overwrite existing results, "
            "set overwrite to True in config.yaml.\n".format(name)
        )
        from bacpipe.embedding_evaluation.probing.train_probe import (
            LinearProbe,
        )

        state_dict = torch.load(
            paths.probe_path / f"{name}_probe.pt",
            map_location=torch.device(
                kwargs.get("device", bacpipe.settings.device)
            ),
        )
        probe = LinearProbe(
            in_dim=embeds.shape[-1],
            out_dim=list(state_dict.values())[-1].shape[0],
            **kwargs,
        )
        probe.load_state_dict(state_dict=state_dict)
        with open(paths.probe_path / "label2index.json", "r") as f:
            label2index = json.load(f)

        load_path = paths.probe_path.joinpath(f"probe_results_{name}.json")
        with open(load_path, "r") as f:
            metrics = json.load(f)

    return probe, label2index, metrics
