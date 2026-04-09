import logging
import json
import numpy as np
import torch

from pathlib import Path

import bacpipe
logger = logging.getLogger(__name__)

from .train_probe import train_probe, LinearProbe
from .evaluate_probe import eval_probe
from .dataset_probe import generate_annotations_for_probing_task


def embeds_array_without_noise(embeds, ground_truth, label_column, **kwargs):
    if len(ground_truth[f"label:{label_column}"].shape) > 1:
        bool_array = np.any(ground_truth[f"label:{label_column}"] > -1, axis=1)
    else:
        bool_array = ground_truth[f"label:{label_column}"] > -1
        
    if isinstance(embeds, np.ndarray):
        return embeds[bool_array]
    elif isinstance(embeds, dict):
        return np.concatenate(list(embeds.values()))[
        bool_array
    ]

def probing_pipeline(
    ground_truth, embeds, 
    paths=None, name='linear', 
    overwrite=True, 
    label_column=bacpipe.settings.label_column, 
    **kwargs
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
        df = generate_annotations_for_probing_task(
            ground_truth, paths, label_column=label_column, **kwargs
            )

        embeds = embeds_array_without_noise(
            embeds, ground_truth, label_column=label_column, **kwargs
            )
        
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

        probe = train_probe(embeds, df, label2index, config=name, **kwargs)
        
        metrics = eval_probe(
            probe, embeds, df, label2index, config=name, paths=paths, **kwargs
            )

        return probe, metrics
    else:
        logger.info(
            f"Classification file probe_results_{name}.json already exists and"
            " so is not computed. If you want to overwrite existing results, "
            "set overwrite to True in config.yaml."
        )

    
def prepare_probe_inference(model, probe_path=''):
    from bacpipe import config, settings
    if probe_path == '':
        import bacpipe.embedding_evaluation.label_embeddings as le
        path_func = le.make_set_paths_func(
            config.audio_dir, 
            settings.main_results_dir, 
            settings.dim_reduc_parent_dir
        )
        probe_path = (
            path_func(model).probe_path / 'linear_probe.pt'
            ).as_posix()
    
    with open(Path(probe_path).parent / 'label2index.json', 'r') as f:
        label2index = json.load(f)
        
    probe_weights = torch.load(probe_path, map_location=settings.device)
    probe = LinearProbe(
        probe_weights['probe.weight'].shape[-1], 
        len(label2index)
        )
    probe.load_state_dict(probe_weights)
    probe.to(settings.device)
    
    return probe, label2index


def run_probe_inference(
    model, linear_probe, threshold, 
    embeds=None, return_binary_presence=True, callbacks=None
    ):
    if embeds is None:
        from bacpipe.core.experiment_manager import Loader
        from bacpipe import config, settings
        
        ld = Loader(
            audio_dir=config.audio_dir, 
            model_name=model,
            **vars(settings)
            )
        embeds = torch.Tensor(ld.embeddings(return_type='array')).to(settings.device)
    
    import torch.nn.functional as F
    return_values = []
    for idx, batch in enumerate(embeds):
        logits = linear_probe(batch)
        probabilities = F.softmax(logits, dim=0).detach().cpu().numpy()
        if return_binary_presence:
            binary_presence = np.zeros(probabilities.shape, dtype=np.int8)
            binary_presence[probabilities > threshold] = 1
            return_values.append(binary_presence.tolist())
            return_dtype = np.int8
        else:
            return_values.append(probabilities.tolist())
            return_dtype = np.float32
        
        if isinstance(callbacks, dict) and hasattr(callbacks, 'progress_bar'):
            callbacks.progress_bar.value = int((idx+1)/len(embeds)*100)
    
    return np.array(return_values, dtype=return_dtype)