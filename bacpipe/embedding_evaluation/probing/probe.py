import logging
import json
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

from .train_probe import (
    train_classifier,
    LinearClassifier
)
from .evaluate_probe import eval_probe, save_probe_results

from bacpipe.embedding_evaluation.visualization.visualize_predictions import (
    plot_classification_results,
)

from bacpipe.embedding_evaluation.label_embeddings import (
    generate_annotations_for_probing_task
)




def embeds_array_without_noise(embeds, ground_truth, label_column, **kwargs):
    if len(ground_truth[f"label:{label_column}"].shape) > 1:
        bool_array = np.any(ground_truth[f"label:{label_column}"] > -1, axis=1)
    else:
        bool_array = ground_truth[f"label:{label_column}"] > -1
        
    return np.concatenate(list(embeds.values()))[
        bool_array
    ]

def probing_pipeline(
    ground_truth, embeds, 
    paths=None, name='linear', 
    overwrite=True, save_probe=False, **kwargs
):
    """
    Classification pipeline consisting of building the classifier,
    evaluating it and saving metrics and plots of performance.

    Parameters
    ----------
    paths : SimpleNamespace object
        dict with attributes corresponding to paths for loading and saving
    embeds : np.array
        embeddings
    name : string
        Type of classification
    dataset_csv_path : string
        name of classification dataframe as specified in settings.yaml
    overwrite : bool
        overwrite existing classification?, defaults to False
    """
    if (
        overwrite
        or paths is None
        or not paths.probe_path.joinpath(f"probe_results_{name}.json").exists()
    ):
        df = generate_annotations_for_probing_task(ground_truth, paths, **kwargs)

        embeds = embeds_array_without_noise(embeds, ground_truth, **kwargs)
        
        if not len(embeds) > 0:
            error = (
                "\nNo embeddings were found for classification task. "
                "Are you sure there are annotations for the data and the annotations.csv file "
                "has been correctly linked? If you didn't intent do do classification, "
                "simply remove it from the evaluation tasks list in the config.yaml file."
            )
            logger.exception(error)
            raise AssertionError(error)        

        label2index = {label: i for i, label in enumerate(df.label.unique())}

        probe = train_classifier(embeds, df, label2index, config=name, **kwargs)
        
        metrics = eval_probe(
            probe, embeds, df, label2index, config=name, **kwargs
            )

        if save_probe and not paths is None:
            state_dict = probe.state_dict()
            torch.save(state_dict, paths.probe_path / f"{name}_probe.pt")
            with open(paths.probe_path / "label2index.json", "w") as f:
                json.dump(label2index, f, indent=1)
            save_probe_results(paths, name, metrics, **kwargs)
            plot_classification_results(paths=paths, task_name=name, metrics=metrics)
        
        return probe, metrics
    else:
        logger.info(
            f"Classification file probe_results_{name}.json already exists and"
            " so is not computed. If you want to overwrite existing results, "
            "set overwrite to True in config.yaml."
        )
