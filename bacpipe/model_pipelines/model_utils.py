import os
import logging
from pathlib import Path
import importlib.resources as pkg_resources

import bacpipe
import torch

logger = logging.getLogger("bacpipe")

class ModelBaseClass:    
    def __init__(self, sr, segment_length, device, 
                 model_base_path, global_batch_size, 
                 model_name, dim_reduction_model=False,
                 **kwargs):
        """
        This base class defines key methods and attributes for all feature
        extractors to ensure that we can use the same processing pipeline
        to generate embeddings. The idea is to 
        
        
        1. initialize the model with prepare_inference, thereby loading 
        the model and loading it onto the selected device.
        
        2. load and resample audio to the sample rate required by the model
        
        3. window the audio into segments corresponding to the required
        input segment length.
        
        4. Calculating spectrograms (if the model architecture is accessible)
        to batch preprocess the audio and potentially be able to in 
        retrospect build the spectrograms to investigate
        
        5. Initialize a torch dataloader object based on the model specific
        audio loading characteristics to speed up the inference process
        and looping through the segments
        
        6. Perform batch inference
        
        
        If 'cuda' has been selected as device, a threading approach is used
        to load data in parallel while performing inference. The return value 
        are the embeddings.

        Parameters
        ----------
        sr : int
            sample rate
        segment_length : int
            segment length in samples
        device : str
            'cpu' or 'cuda'
        model_base_path : pathlib.Path
            path to moin model checkpoint dir
        global_batch_size : int
            global batch size that is then used in comjunction with the 
            segment length to calculate a model-specific batch size that
            results in approximately equal batches for different models
        """
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (self.run_pretrained_classifier 
            and hasattr(self, 'classifier_predictions')):
            self.bool_classifier = True
        else:
            self.bool_classifier = False
            
        self.device = device
        if (
            self.device == 'cuda' 
            and not dim_reduction_model 
            and model_name in bacpipe.TF_MODELS
            ):
            if not check_if_cudnn_tensorflow_compatible():
                # Force TensorFlow to ignore all GPUs
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                
                self.device = 'cpu'
        elif self.device == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ.pop("CUDA_VISIBLE_DEVICES")
                
        logger.info(f"Using {device=}")
                
        self.model_base_path = Path(model_base_path)
        with pkg_resources.path(bacpipe, "model_specific_utils") as utils_path:
            self.model_utils_base_path = Path(utils_path)
        
        self.global_batch_size = global_batch_size
        
        self.sr = sr
        self.segment_length = segment_length
        if segment_length:
            self.batch_size = int(100_000 * self.global_batch_size / segment_length)

    def prepare_inference(self):
        if 'umap' in self.__module__:
            return
        try:
            self.model.eval()
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                logger.error(e)
                pass
        except AttributeError:
            logger.error("Skipping model.eval() because model is from tensorflow.")
            pass


def check_if_cudnn_tensorflow_compatible():
    import torch
    version = (torch.backends.cudnn.version() % 1000) // 100
    if version < 3:
        logger.info(
            "cuDNN version does not match the required 9.3 for tensorflow. "
            "Device is therefore set to cpu for the tensorflow models."
            )
        return False
    else:
        return True
    