import torch
import logging
import numpy as np
import librosa as lb
from pathlib import Path
import audioread

logger = logging.getLogger("bacpipe")


class AudioHandler:
    """
    Helper class for all methods related to loading and padding audio.
    This class takes care of loading the audio files as a whole
    or just the annotated segments, resampling, windowing, resampling, etc.
    
    The class is built around extracting audio relative to what different
    deep learning models require, therefore it requires a Embedder.model object
    which has information like the sampling rate, the segment length included.
    These attributes can be changed by changing Embedder.model.sr before
    passing the object to AudioHandler.
    
    Examples::
    
        # Load the ``birdnet`` model and use it to window the test audio files
        # into frames that match the model input length:

        from bacpipe import Embedder, get_audio_files, AudioHandler
        import numpy as np

        embed = Embedder('birdnet')

        aud = AudioHandler(
            model=embed.model,
            audio_dir='bacpipe/tests/test_data'
        )
        files = get_audio_files('bacpipe/tests/test_data')

        all_frames = []
        for audio_file in files:
            audio, sr = aud.load_and_resample(audio_file)
            frames = aud.window_audio(audio)
            all_frames.extend(frames)
        all_frames = np.stack(all_frames)

    """

    def __init__(
        self,
        model,
        audio_dir,
        padding='constant',
        bool_change_speed=False,
        new_speed=None,
        **kwargs,
    ):
        """
        Helper class for all methods related to loading and padding audio.

        Parameters
        ----------
        model : Model object
            has attributes for all the model characteristics like
            sample rate, segment length etc. as well as the methods
            to run the model
        audio_dir : pathlib.Path object
            path to audio dir
        padding : str, optional
            padding function to use for where padding is necessary.
            Detaults to constant.
        bool_change_speed : bool, optional
            whether to change the speed of the audio before processing,
            by default False
        new_speed : float, optional
            new speed to use when changing the playback speed of the
            audio, by default None
        """
        self.model = model
        self.padding = padding
        self.audio_dir = audio_dir
        self.bool_change_speed = bool_change_speed
        self.new_speed = new_speed
        self.kwargs = kwargs
        
    def prepare_audio(self, sample):
        """
        Use bacpipe pipeline to load audio file, window it according to
        model specific window length.
        The audio then gets preprocessed based on the model-specific
        preprocessing, i.e. transforming it into spectrograms.
        Following that, the data is ready for batch inference computation. 
        Also log file length and shape for metadata files.

        Parameters
        ----------
        sample : pathlib.Path or str
            path to audio file

        Returns
        -------
        torch.Tensor
            audio frames preprocessed with model specific preprocessing
        """
        
        if self.model.only_embed_annotations:
            frames = self.only_load_annotated_segments(
                sample, **self.kwargs
            )
            sr = None
        else:
            audio, sr = self.load_and_resample(sample)
            frames = self.window_audio(audio)
        preprocessed_frames = self.model.preprocess(frames)
        self.preprocessed_shape = tuple(preprocessed_frames.shape)
        if self.model.device == "cuda":
            if self.model.only_embed_annotations:
                del frames
            else:
                del audio, frames
            torch.cuda.empty_cache()
        return preprocessed_frames

    def get_file_length(self, path):
        """
        Determine the length of the audio file at ``path`` and store it
        in ``self.file_length`` under the file stem. When
        ``bool_change_speed`` is set, the stored length is divided by the
        new speed.

        Parameters
        ----------
        path : pathlib.Path or str
            path to the audio file
        """
        with audioread.audio_open(str(path)) as f:
            length = f.duration
        if not hasattr(self, 'file_length'):
            self.file_length = dict()
        if not self.bool_change_speed:
            self.file_length[path.stem] = length
        else:
            self.file_length[path.stem] = length / self.new_speed

    def load_and_resample(self, path):
        """
        Load an audio file and resample it to the model sample rate.

        Parameters
        ----------
        path : pathlib.Path or str
            path to the audio file

        Returns
        -------
        torch.Tensor
            mono audio waveform
        int
            sample rate of the loaded audio
        """
        try:
            self.get_file_length(path)
            if not self.bool_change_speed:
                audio, sr = lb.load(str(path), sr=self.model.sr, mono=True)
            else:
                audio, sr = lb.load(str(path), sr=None, mono=True)
                if "batdetect2" in self.model_name:
                    fake_original_sr = self.model.sr
                else:
                    fake_original_sr = int(sr * self.new_speed)
                audio = lb.resample(
                    audio, orig_sr=fake_original_sr, target_sr=self.model.sr
                )
            audio = audio.reshape(1, -1)
        except Exception as e:
            logger.exception(
                f"\nError loading audio. Skipping {str(path)}." f"Error: {str(e)}"
            )
            raise e
        if len(audio) == 0:
            error = f"Audio file {path} is empty. " f"Skipping {path}."
            logger.exception(error)
            raise ValueError(error)
        return torch.tensor(audio), sr

    def only_load_annotated_segments(
        self, file_path, annotations_filename="annotations.csv", **_
    ):
        """
        Load only the segments of an audio file that are covered by
        annotations in the annotations CSV file.
        
        Several species can share the same time window, so the raw
        annotations can contain multiple rows with the same (start, end)
        pair. Deduplicate the *pairs* as a unit. 
        ``filter_df_by_file`` already sorted the annotations by start and
        ``drop_duplicates`` keeps that order, so the segments are loaded in
        the same order in which the classifier predictions are collected.
        
        Example::
        
            from bacpipe import Embedder, get_audio_files, AudioHandler
            import numpy as np
            embed = Embedder('birdnet')

            aud = AudioHandler(
                model=embed.model,
                audio_dir='bacpipe/tests/test_data',
                only_embed_annotations=True
            )
            files = get_audio_files('bacpipe/tests/test_data')

            all_frames = []
            for audio_file in files:
                frames = aud.only_load_annotated_segments(audio_file)
                all_frames.extend(frames)
            all_frames = np.stack(all_frames)

        Parameters
        ----------
        file_path : pathlib.Path or str
            path to the audio file
        annotations_filename : str, optional
            name of the annotations CSV file located in the audio
            directory, by default "annotations.csv"

        Returns
        -------
        torch.Tensor
            tensor containing the annotated audio segments padded to the
            model segment length
        """
        import pandas as pd
        from bacpipe import Loader


        annots = pd.read_csv(Path(self.audio_dir) / annotations_filename)
        # filter current file
        file_annots = Loader.filter_df_by_file(
            self.audio_dir, annots, file_path
        )
        if len(file_annots) == 0:
            raise AssertionError(
                f"No annotations found for audio file {file_path.relative_to(self.audio_dir)}. "
                "Continuing with next file."
            )

        file_annots = file_annots.drop_duplicates(subset=["start", "end"])

        self.get_file_length(file_path)
        file_duration = self.file_length[file_path.stem]

        segments = []
        for s, e in zip(file_annots["start"], file_annots["end"]):
            s, e = float(s), float(e)
            if e <= s:
                logger.warning(
                    f"Annotation with start {s} and end {e} has duration "
                    f"zero or negative, which doesn't make any sense. "
                    f"Skipping annotation for {file_path}."
                )
                continue
            if s >= file_duration:
                logger.warning(
                    f"Annotation with start {s} and end {e} is outside of "
                    f"range of {file_path}. Skipping annotation."
                )
                continue
            duration = min(e - s, file_duration - s)
            audio, _ = lb.load(
                str(file_path),
                sr=self.model.sr,
                mono=True,
                offset=s,
                duration=duration,
            )
            segments.append(
                lb.util.fix_length(
                    audio,
                    size=self.model.segment_length,
                    mode=self.padding,
                )
            )

        if len(segments) == 0:
            raise AssertionError(
                f"No valid annotations found for audio file "
                f"{file_path.relative_to(self.audio_dir)}. "
                "Continuing with next file."
            )

        cumulative_segments = torch.Tensor(np.vstack(segments))
        return cumulative_segments

    def _load_audio_based_on_fixed_segment_length(
        self, audio, segment_length, **_
    ):
        """
        Compute the start and end indices used to split an audio signal
        into non-overlapping fixed-length segments.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor
            audio signal
        segment_length : float
            length of each segment in seconds

        Returns
        -------
        np.ndarray
            array of start indices in samples
        np.ndarray
            array of end indices in samples
        """
        nr_segments = len(audio) // segment_length + 1
        starts = np.arange(nr_segments) * segment_length * self.model.sr
        ends = np.arange(1, nr_segments + 1) * segment_length * self.model.sr
        return starts, ends

    def _load_and_pad_audio_based_on_grid(
        self, audio, starts, ends, file_path
    ):
        """
        Extract the audio segments defined by ``starts`` and ``ends`` from
        an audio signal and pad them to the model segment length.

        Parameters
        ----------
        audio : torch.Tensor
            audio signal
        starts : np.ndarray
            array of segment start indices in samples
        ends : np.ndarray
            array of segment end indices in samples
        file_path : pathlib.Path
            path to the audio file, used for logging warnings

        Returns
        -------
        torch.Tensor
            tensor containing the padded audio segments
        """
        audio = audio.cpu().squeeze()
        for idx, (s, e) in enumerate(zip(starts, ends)):
            s, e = int(s), int(e)
            if s > len(audio):
                logger.warning(
                    f"Annotation with start {s} and end {str(e)} is outside of "
                    f"range of {file_path}. Skipping annotation."
                )
                continue
            segments = lb.util.fix_length(
                audio[s : e + 1],
                size=self.model.segment_length,
                mode=self.padding,
            )
            if idx == 0:
                cumulative_segments = segments
            else:
                cumulative_segments = np.vstack(
                    [cumulative_segments, segments]
                )
        cumulative_segments = torch.Tensor(cumulative_segments)
        cumulative_segments = cumulative_segments.to(self.device)
        return cumulative_segments

    def window_audio(self, audio):
        """
        Split an audio signal into windows of the model segment length and
        pad the final window if necessary.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor
            audio signal

        Returns
        -------
        torch.Tensor
            audio frames of shape (num_frames, segment_length)
        """
        num_frames = int(np.ceil(len(audio[0]) / self.model.segment_length))
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu()
        padded_audio = lb.util.fix_length(
            audio,
            size=int(num_frames * self.model.segment_length),
            mode=self.padding,
        )
        logger.debug(f"{self.padding} was used on an audio segment.")
        if len(padded_audio.shape) > 1 and padded_audio.shape[0] > 1:
            frames = padded_audio
        else:
            frames = padded_audio.reshape([num_frames, self.model.segment_length])
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames)
        return frames
