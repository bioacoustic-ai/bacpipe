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
                 padding, **kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.run_pretrained_classifier:
            self.bool_classifier = True
        else:
            self.bool_classifier = False
            
        self.device = device
        self.model_base_path = Path(model_base_path)
        with pkg_resources.path(bacpipe, "model_specific_utils") as utils_path:
            self.model_utils_base_path = Path(utils_path)
        
        self.global_batch_size = global_batch_size
        self.padding = padding
        
        self.classifier_outputs = torch.tensor([])
        self.model_base_path = self.model_base_path
        self.classification_threshold = 0.1
        self.sr = sr
        self.segment_length = segment_length
        if segment_length:
            self.batch_size = int(100_000 * self.global_batch_size / segment_length)

    def prepare_inference(self):
        try:
            self.model.eval()
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                print(e)
                pass
        except AttributeError:
            logger.debug("Skipping model.eval() because model is from tensorflow.")
            pass

    def load_and_resample(self, path):
        try:
            audio, sr = ta.load(path, normalize=True)
        except Exception as e:
            logger.debug(
                f"Error loading audio with torchaudio. "
                f"Skipping {path}."
                f"Error: {e}"
            )
            raise e
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0).unsqueeze(0)
        if len(audio[0]) == 0:
            logger.debug(f"Audio file {path} is empty. " f"Skipping {path}.")
            raise ValueError(f"Audio file {path} is empty.")
        re_audio = ta.functional.resample(audio, sr, self.sr)
        return re_audio.to(self.device)

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

    def filter_classifier_predictions(self, cls_vals):
        if not isinstance(self.classes, np.ndarray):
            self.classes = np.array(self.classes)

        bool_exceeding_threshold = np.where(cls_vals > self.classification_threshold)[1]
        classes_exceeding_thresh = np.unique(self.classes[bool_exceeding_threshold])

        cls_results = {
            k: cls_vals[:, idx]
            for idx, k in enumerate(self.classes)
            if k in classes_exceeding_thresh
        }

        return cls_results
