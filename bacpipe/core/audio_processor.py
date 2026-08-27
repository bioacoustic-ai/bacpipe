import torch
import logging
import numpy as np
import librosa as lb
from pathlib import Path
import audioread

logger = logging.getLogger("bacpipe")


class _ModelStub:
    """
    Lightweight stand-in for a feature extractor model.

    ``AudioHandler`` only needs the sample rate and the segment length of a
    model to load, resample and window audio. When a model is passed by
    name, loading the real model (which requires the model checkpoint) is
    therefore postponed until the model specific preprocessing is needed,
    i.e. until ``AudioHandler.prepare_audio`` is called.
    """

    def __init__(
        self, name, sr, segment_length, only_embed_annotations=False
    ):
        """
        Parameters
        ----------
        name : str
            name of the model
        sr : int
            sample rate the model expects
        segment_length : int
            segment length in samples the model expects
        only_embed_annotations : bool, optional
            whether only the annotated segments are loaded,
            by default False
        """
        self.name = name
        self.model_name = name
        self.sr = sr
        self.segment_length = segment_length
        self.only_embed_annotations = only_embed_annotations
        self.device = "cpu"


def _get_model_constants(model_name):
    """
    Look up the sample rate and the segment length of a supported model
    without instantiating (and thereby downloading) the model itself.

    Parameters
    ----------
    model_name : str
        name of a model supported by bacpipe

    Returns
    -------
    str
        the (lowercased) model name
    int
        sample rate the model expects
    int
        segment length in samples the model expects

    Raises
    ------
    NameError
        If the model name is not supported by bacpipe.
    ImportError
        If the module of the model exists but cannot be imported.
    AttributeError
        If the sample rate and segment length cannot be found in the
        module of the model.
    """
    import sys
    from importlib import import_module
    import bacpipe

    # this raises a NameError listing all supported models, so a typo is
    # not reported as a missing module
    model_name = bacpipe.confirm_model_name(model_name)
    try:
        model_module = import_module(
            f"bacpipe.model_pipelines.feature_extractors.{model_name}"
        )
    except Exception as e:
        raise ImportError(
            f"\nThe module of the model {model_name} could not be imported "
            f"({type(e).__name__}: {e}). Please check that the requirements "
            f"of {model_name} are installed.\n"
        ) from e

    # some model modules only subclass the model of another module (e.g.
    # birdaves_especies subclasses aves_especies) and use the sample rate
    # and segment length defined there, so the parent modules are checked
    # as well
    modules = [model_module] + [
        sys.modules.get(base.__module__)
        for base in getattr(model_module, "Model", object).__mro__[1:]
    ]
    sr, segment_length = None, None
    for module in modules:
        if module is None:
            continue
        if sr is None:
            sr = getattr(module, "SAMPLE_RATE", None)
        if segment_length is None:
            segment_length = getattr(module, "LENGTH_IN_SAMPLES", None)
        if not sr is None and not segment_length is None:
            break

    if sr is None or segment_length is None:
        raise AttributeError(
            f"\nThe sample rate and the segment length of the model "
            f"{model_name} could not be determined from its module. Please "
            "pass the model object itself, for example "
            f"bacpipe.Embedder('{model_name}').model, instead of its name.\n"
        )
    return model_name, sr, segment_length


class AudioHandler:
    """
    Helper class for all methods related to loading and padding audio.
    This class takes care of loading the audio files as a whole
    or just the annotated segments, resampling, windowing, resampling, etc.

    The class is built around extracting audio relative to what different
    deep learning models require, therefore it needs to know the sample rate
    and the segment length of a model. Either pass the name of a supported
    model (e.g. ``'birdnet'``), in which case both values are read from the
    model module and the model itself is only loaded once the model specific
    preprocessing is needed, or pass a model object (``Embedder.model``),
    which additionally provides the preprocessing and the model itself.
    Both values can be changed at any time by setting ``aud.model.sr`` and
    ``aud.model.segment_length``.

    Examples::

        # Window the test audio files into frames that match the input
        # length of ``birdnet``. Passing the model name is enough, no model
        # checkpoint is loaded for this:

        from bacpipe import get_audio_files, AudioHandler
        import numpy as np

        aud = AudioHandler(
            model='birdnet',
            audio_dir='bacpipe/tests/test_data'
        )
        files = get_audio_files('bacpipe/tests/test_data')

        all_frames = []
        for audio_file in files:
            audio, sr = aud.load_and_resample(audio_file)
            frames = aud.window_audio(audio)
            all_frames.extend(frames)
        all_frames = np.stack(all_frames)

        # The sample rate and the segment length of the model can be
        # changed, for example to window the audio into 10 second frames
        # at a sample rate of 32 kHz:

        aud.model.sr = 32_000
        aud.model.segment_length = 10 * aud.model.sr

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
        model : Model object or str
            either a model object, which has attributes for all the model
            characteristics like sample rate, segment length etc. as well
            as the methods to run the model, or the name of a model
            supported by bacpipe. In the latter case the sample rate and
            the segment length are read from the model module and the
            model itself is only loaded if it is needed (see
            ``prepare_audio``).
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
        **kwargs
            additional keyword arguments, e.g.
            ``only_embed_annotations`` or the ``annotations_df`` and
            ``annotations_filename`` used by
            ``only_load_annotated_segments``

        Raises
        ------
        NameError
            If a model name is passed that is not supported by bacpipe.
        """

        if isinstance(model, str):
            from bacpipe import settings as bacpipe_settings

            model_name, sr, segment_length = _get_model_constants(model)
            self.model = _ModelStub(
                name=model_name,
                sr=sr,
                segment_length=segment_length,
                # a model object receives this setting on instantiation, so
                # the stub is given the passed value or the default, too
                only_embed_annotations=kwargs.get(
                    "only_embed_annotations",
                    bacpipe_settings.only_embed_annotations,
                ),
            )
        else:
            self.model = model
        if not hasattr(self, "model_name"):
            # ``Embedder`` sets the name (which can also be the name of a
            # dimensionality reduction model) before calling this
            # constructor, so it is never overwritten here
            self.model_name = getattr(
                self.model, "model_name", getattr(self.model, "name", "")
            )
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
        if isinstance(self.model, _ModelStub):
            # the model was passed by name, so only its sample rate and
            # segment length were known so far. The model specific
            # preprocessing requires the model itself, which is loaded now.
            self._replace_model_stub_with_model()
        preprocessed_frames = self.model.preprocess(frames)
        self.preprocessed_shape = tuple(preprocessed_frames.shape)
        if self.model.device == "cuda":
            if self.model.only_embed_annotations:
                del frames
            else:
                del audio, frames
            torch.cuda.empty_cache()
        return preprocessed_frames

    def _replace_model_stub_with_model(self):
        """
        Load the model that was passed by name and use it from now on.

        The sample rate, the segment length and ``only_embed_annotations``
        are carried over from the stub, so values the user changed (e.g.
        ``aud.model.sr = 32_000``) are preserved and the audio is
        preprocessed the way it was loaded.
        """
        from bacpipe import Embedder

        stub = self.model
        self.model = Embedder(stub.name, audio_dir=self.audio_dir).model
        self.model.sr = stub.sr
        self.model.segment_length = stub.segment_length
        self.model.only_embed_annotations = stub.only_embed_annotations

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
        self,
        file_path,
        annotations_filename="annotations.csv",
        annotations_df=None,
        **_,
    ):
        """
        Load only the segments of an audio file that are covered by
        annotations, either from the annotations CSV file in the audio
        directory or from an annotations dataframe.
        
        Several species can share the same time window, so the raw
        annotations can contain multiple rows with the same (start, end)
        pair. Deduplicate the *pairs* as a unit. 
        ``filter_df_by_file`` already sorted the annotations by start and
        ``drop_duplicates`` keeps that order, so the segments are loaded in
        the same order in which the classifier predictions are collected.
        
        Example::

            from bacpipe import get_audio_files, AudioHandler
            import numpy as np

            # passing the model name is enough here, the model itself is
            # only needed for the model specific preprocessing
            aud = AudioHandler(
                model='birdnet',
                audio_dir='bacpipe/tests/test_data',
                only_embed_annotations=True
            )
            files = get_audio_files('bacpipe/tests/test_data')

            all_frames = []
            for audio_file in files:
                frames = aud.only_load_annotated_segments(audio_file)
                all_frames.extend(frames)
            all_frames = np.stack(all_frames)
            
            # or if you want to modify the model's original sample rate
            aud.model.sr = 32_000
            aud.model.segment_length = 3 * aud.model.sr
            all_frames = []
            for audio_file in files:
                frames = aud.only_load_annotated_segments(audio_file)
                all_frames.extend(frames)
            all_frames = np.stack(all_frames)

            # instead of the annotations csv your own annotations can be
            # passed as a dataframe. Only the segments it annotates are
            # returned, the rest of the audio is ignored.
            import pandas as pd

            annots = pd.read_csv('bacpipe/tests/test_data/annotations.csv')
            annots = annots[annots.start < 10]

            all_frames = []
            for audio_file in files:
                frames = aud.only_load_annotated_segments(
                    audio_file, annotations_df=annots
                )
                all_frames.extend(frames)
            all_frames = np.stack(all_frames)

        Parameters
        ----------
        file_path : pathlib.Path or str
            path to the audio file
        annotations_filename : str, optional
            name of the annotations CSV file located in the audio
            directory, by default "annotations.csv"
        annotations_df : pd.DataFrame, optional
            annotations to use instead of the annotations CSV file. Needs
            the columns ``start`` and ``end`` (in seconds). If it also has
            an ``audiofilename`` column, it is filtered down to the rows
            belonging to ``file_path``, otherwise it is expected to only
            contain the annotations of that file, by default None

        Returns
        -------
        torch.Tensor
            tensor containing the annotated audio segments padded to the
            model segment length

        Raises
        ------
        AssertionError
            If the annotations contain no (valid) annotation for
            ``file_path`` or if a dataframe without ``start`` and ``end``
            columns is passed.
        """
        import pandas as pd
        from bacpipe import Loader

        file_path = Path(file_path)
        if isinstance(annotations_df, pd.DataFrame):
            file_annots = annotations_df
            missing_columns = {"start", "end"} - set(file_annots.columns)
            if missing_columns:
                raise AssertionError(
                    f"\nThe annotations_df is missing the column(s) "
                    f"{sorted(missing_columns)}. Annotated segments are "
                    "loaded based on the columns 'start' and 'end', which "
                    "have to contain the times in seconds.\n"
                )
            if "audiofilename" in file_annots.columns:
                # a user dataframe usually covers all files of the dataset,
                # so it is filtered down to the current file, exactly like
                # the annotations csv
                file_annots = Loader.filter_df_by_file(
                    self.audio_dir, file_annots, file_path
                )
        else:
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

        The following input shapes are supported:

        - ``(1, num_samples)``: one (long) recording, which is split into
          ``ceil(num_samples / segment_length)`` windows
        - ``(num_segments, segment_length)``: an already stacked array of
          segments, which is returned unchanged
        - ``(num_segments, num_samples)``: a stack of segments that are
          shorter or longer than the model segment length. Shorter segments
          are padded, longer ones are split into several windows.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor
            audio signal of shape (num_rows, num_samples)

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
        # every row is now a multiple of the segment length, so all rows can
        # be split into windows of exactly segment_length samples
        frames = padded_audio.reshape([-1, self.model.segment_length])
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames)
        return frames
