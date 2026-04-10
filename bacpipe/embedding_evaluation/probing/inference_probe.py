import logging
import json
import numpy as np
import torch

from pathlib import Path

logger = logging.getLogger(__name__)

from .train_probe import LinearProbe



def prepare_probe_inference(model, probe_path=''):
    """
    Load a linear probe that was previously trained and saved.
    The probe is loaded and the state_dict of the model is loaded
    so that the probe is ready and in the exact same state as after
    training. 

    Parameters
    ----------
    model : str
        model name of backbone
    probe_path : str, optional
        path to probe, will default to the standard bacpipe path, by default ''

    Returns
    -------
    torch model object
        linear probe model
    dict
        dictionary to associate the columns of the generated predictions array
        with the corresponding class label
    """
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
    embeds=None, return_binary_presence=True, 
    callbacks=None, device='cpu'
    ):
    """
    Apply a previously trained linear probe to data. 
    This requires either that the embeddings were already created
    using the backbone and saved using the bacpipe folder structure,
    or that the embeddings are directly passed to this function. 
    See the examples notebooks for an example use case. 
    This function then loads the embeddings and applies the
    linear probe to classify the data. 

    Parameters
    ----------
    model : str
        model name
    linear_probe : torch model
        linear probe torch model object
    threshold : float
        float value to process the predictions
    embeds : torch.Tensor, optional
        embeddings array, by default None
    return_binary_presence : bool, optional
        if true a binary presence array is returned, by default True
    callbacks : function, optional
        use to have custom progress bars increment, by default None
    device : str, optional
        select device to process the probe, by default 'cpu'

    Returns
    -------
    np.ndarray
        generated probe predictions
    """
    if embeds is None:
        from bacpipe.core.experiment_manager import Loader
        from bacpipe import config, settings
        
        ld = Loader(
            audio_dir=config.audio_dir, 
            model_name=model,
            **vars(settings)
            )
        embeds = torch.Tensor(ld.embeddings(return_type='array')).to(settings.device)
    elif isinstance(embeds, np.ndarray):
        embeds = torch.Tensor(embeds)
    
    embeds = embeds.to(device)
    linear_probe = linear_probe.to(device)
    
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