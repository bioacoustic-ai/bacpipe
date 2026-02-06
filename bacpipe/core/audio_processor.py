
import torch
import logging
import numpy as np
import librosa as lb
import torchaudio as ta
from pathlib import Path

logger = logging.getLogger("bacpipe")


class AudioHandler:
    def __init__(self, model, padding, audio_dir, **kwargs):
        """
        Helper class for all methods related to loading and padding audio. 

        Parameters
        ----------
        model : Model object
            has attributes for all the model characteristics like 
            sample rate, segment length etc. as well as the methods
            to run the model
        padding : str
            padding function to use for where padding is necessary
        audio_dir : pathlib.Path object
            path to audio dir
        """
        self.model = model
        self.padding = padding
        self.audio_dir = audio_dir
    
    def prepare_audio(self, sample):
        """
        Use bacpipe pipeline to load audio file, window it according to 
        model specific window length and preprocess the data, ready for 
        batch inference computation. Also log file length and shape for
        metadata files.

        Parameters
        ----------
        sample : pathlib.Path or str
            path to audio file

        Returns
        -------
        torch.Tensor
            audio frames preprocessed with model specific preprocessing
        """
        audio = self._load_and_resample(sample)
        audio = audio.to(self.model.device)
        if self.model.only_embed_annotations:
            frames = self._only_load_annotated_segments(sample, audio)
        else:
            frames = self._window_audio(audio)
        preprocessed_frames = self.model.preprocess(frames)
        self.file_length[sample.stem] = len(audio[0]) / self.model.sr
        self.preprocessed_shape = tuple(preprocessed_frames.shape)
        if self.model.device == 'cuda':
            del audio, frames
            torch.cuda.empty_cache()
        return preprocessed_frames
    
    def _load_and_resample(self, path):
        try:
            audio, sr = ta.load(str(path), normalize=True)
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
        re_audio = ta.functional.resample(audio, sr, self.model.sr)
        return re_audio

    def _only_load_annotated_segments(self, file_path, audio):
        import pandas as pd
        annots = pd.read_csv(Path(self.audio_dir) / 'annotations.csv')
        # filter current file
        file_annots = annots[annots.audiofilename==file_path.relative_to(self.audio_dir)]
        if len(file_annots) == 0:
            file_annots = annots[annots.audiofilename==file_path.stem+file_path.suffix]
        if len(file_annots) == 0:
            file_annots = annots[annots.audiofilename==str(file_path.relative_to(self.audio_dir))]
        
        starts = np.array(file_annots.start, dtype=np.float32)*self.model.sr
        ends = np.array(file_annots.end, dtype=np.float32)*self.model.sr

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
                size=self.model.segment_length,
                mode=self.padding
                )
            if idx == 0:
                cumulative_segments = segments
            else:
                cumulative_segments = np.vstack([cumulative_segments, segments])
        cumulative_segments = torch.Tensor(cumulative_segments)
        cumulative_segments = cumulative_segments.to(self.device)
        return cumulative_segments

    def _window_audio(self, audio):
        num_frames = int(np.ceil(len(audio[0]) / self.model.segment_length))
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu()
        padded_audio = lb.util.fix_length(
            audio,
            size=int(num_frames * self.model.segment_length),
            mode=self.padding,
        )
        logger.debug(f"{self.padding} was used on an audio segment.")
        frames = padded_audio.reshape([num_frames, self.model.segment_length])
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames)
        frames = frames.to(self.model.device)
        return frames
