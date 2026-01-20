import os
import logging
from pathlib import Path
import importlib.resources as pkg_resources
from tqdm import tqdm

import bacpipe
import librosa as lb
import numpy as np
import torchaudio as ta
import torch
import tensorflow

logger = logging.getLogger("bacpipe")

class ModelBaseClass:    
    def __init__(self, sr, segment_length, device, 
                 model_base_path, global_batch_size, 
                 model_name, padding, 
                 dim_reduction_model=False,
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
        padding : str
            how to pad audio segments that are shorter than the segment length
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
        self.padding = padding
        
        self.classifier_outputs = torch.tensor([])
        
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

    def load_and_resample(self, path):
        try:
            audio, sr = ta.load(path, normalize=True)
        except Exception as e:
            logger.exception(
                f"\nError loading audio with torchaudio. "
                f"Skipping {path}."
                f"Error: {e}"
            )
            raise e
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0).unsqueeze(0)
        if len(audio[0]) == 0:
            error = f"Audio file {path} is empty. " f"Skipping {path}."
            logger.exception(error)
            raise ValueError(error)
        re_audio = ta.functional.resample(audio, sr, self.sr)
        return re_audio

    def only_load_annotated_segments(self, file_path, audio):
        import pandas as pd
        annots = pd.read_csv(Path(self.audio_dir) / 'annotations.csv')
        # filter current file
        file_annots = annots[annots.audiofilename==file_path.relative_to(self.audio_dir)]
        if len(file_annots) == 0:
            file_annots = annots[annots.audiofilename==file_path.stem+file_path.suffix]
        
        starts = np.array(file_annots.start, dtype=np.float32)*self.sr
        ends = np.array(file_annots.end, dtype=np.float32)*self.sr

        audio = audio.cpu().squeeze()
        for idx, (s, e) in enumerate(zip(starts, ends)):
            s, e = int(s), int(e)
            if e > len(audio):
                logger.warning(
                    f"Annotation with start {s} and end {e} is outside of "
                    f"range of {file_path}. Skipping annotation."
                )
                continue
            segments = lb.util.fix_length(
                audio[s:e+1],
                size=self.segment_length,
                mode=self.padding
                )
            if idx == 0:
                cumulative_segments = segments
            else:
                cumulative_segments = np.vstack([cumulative_segments, segments])
        cumulative_segments = torch.Tensor(cumulative_segments)
        cumulative_segments = cumulative_segments.to(self.device)
        return cumulative_segments

    def window_audio(self, audio):
        num_frames = int(np.ceil(len(audio[0]) / self.segment_length))
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu()
        padded_audio = lb.util.fix_length(
            audio,
            size=int(num_frames * self.segment_length),
            mode=self.padding,
        )
        logger.debug(f"{self.padding} was used on an audio segment.")
        frames = padded_audio.reshape([num_frames, self.segment_length])
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames)
        frames = frames.to(self.device)
        return frames

    def init_dataloader(self, audio):
        if "tensorflow" in str(type(audio)):
            import tensorflow as tf

            return tf.data.Dataset.from_tensor_slices(audio).batch(self.batch_size)
        elif "torch" in str(type(audio)):

            return torch.utils.data.DataLoader(
                audio, batch_size=self.batch_size, shuffle=False
            )

    def batch_inference(self, batched_samples, callback=None):
        embeds = []
        total_batches = len(batched_samples)

        for idx, batch in enumerate(
            tqdm(batched_samples, desc=" processing batches", position=0, leave=False)
        ):
            with torch.no_grad():
                if self.bool_classifier:
                    if self.device == "cuda" and not isinstance(batch, tensorflow.Tensor):
                        batch = batch.cuda()
                        self.classifier_outputs = self.classifier_outputs.cuda()
                        self.classifier = self.classifier.cuda()

                    embedding, cls_vals = self.__call__(batch, return_class_results=True)
                    if not isinstance(batch, tensorflow.Tensor):
                        self.classifier_outputs = torch.cat(
                            [self.classifier_outputs, cls_vals.clone().detach()]
                        )
                    else:
                        self.classifier_outputs = torch.cat(
                            [self.classifier_outputs, torch.Tensor(cls_vals)]
                        )
                else:
                    if self.device == "cuda" and not isinstance(batch, tensorflow.Tensor):
                        batch = batch.cuda()
                    embedding = self.__call__(batch)

            if isinstance(embedding, torch.Tensor) and embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            embeds.append(embedding)

            # callback with progress if progressbar should be updated
            if callback and total_batches > 0:
                fraction = (idx + 1) / total_batches
                callback(fraction)

        if self.bool_classifier:
            self.classifier_outputs = self.classifier_outputs.cpu()

        if isinstance(embeds[0], torch.Tensor):
            return torch.cat(embeds, axis=0)
        else:
            import tensorflow as tf
            return_embeds = tf.concat(embeds, axis=0).numpy().squeeze()
            return return_embeds


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
    