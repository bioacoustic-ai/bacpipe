import yaml
import librosa as lb
import numpy as np
import torchaudio as ta
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger("bacpipe")

MODEL_BASE_PATH = "bacpipe/model_checkpoints"
GLOBAL_BATCH_SIZE = 16


class ModelBaseClass:
    def __init__(self, sr, segment_length, **kwargs):
        with open("bacpipe/config.yaml", "rb") as f:
            self.config = yaml.safe_load(f)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model_base_path = MODEL_BASE_PATH
        self.sr = sr
        self.segment_length = segment_length
        if segment_length:
            self.batch_size = int(100_000 * GLOBAL_BATCH_SIZE / segment_length)

    def prepare_inference(self):
        try:
            self.model.eval()
            try:
                self.model = self.model.to(self.config["device"])
            except AttributeError as e:
                print(e)
                pass
        except AttributeError:
            logger.debug("Skipping model.eval() because model is from tensorflow.")
            pass

    def load_and_resample(self, path):
        audio, sr = ta.load(path, normalize=True)
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0).unsqueeze(0)
        re_audio = ta.functional.resample(audio, sr, self.sr)
        return re_audio

    def window_audio(self, audio):
        num_frames = int(np.ceil(len(audio[0]) / self.segment_length))
        padded_audio = lb.util.fix_length(
            audio, size=int(num_frames * self.segment_length), mode="reflect"
        )
        frames = padded_audio.reshape([num_frames, self.segment_length])
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames)
        frames = frames.to(self.config["device"])
        return frames

    def init_dataloader(self, audio):
        if "tensorflow" in str(type(audio)):
            import tensorflow as tf

            return tf.data.Dataset.from_tensor_slices(audio).batch(self.batch_size)
        elif "torch" in str(type(audio)):

            return torch.utils.data.DataLoader(
                audio, batch_size=self.batch_size, shuffle=False
            )

    def batch_inference(self, batched_samples):
        embeds = []
        for batch in tqdm(
            batched_samples, desc=" processing batches", position=0, leave=False
        ):
            embedding = self.__call__(batch)
            if isinstance(embedding, torch.Tensor) and embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            embeds.append(embedding)
        if isinstance(embeds[0], torch.Tensor):
            return torch.cat(embeds, axis=0)
        else:
            import tensorflow as tf

            return_embeds = tf.concat(embeds, axis=0).numpy().squeeze()
            return return_embeds
