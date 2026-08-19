import pandas as pd
import numpy as np
import plotly.express as px
import librosa as lb
from scipy.signal.windows import tukey
from pathlib import Path
from bacpipe import settings
import logging

logger = logging.getLogger("bacpipe")


def timestamps_match(start_click, start_metadata, tolerance=1e-2):
    """
    Check whether two start timestamps (in seconds) refer to the same audio
    segment.

    A small tolerance is used instead of ``int()`` truncation so that
    sub-second shifts between the plotted and the stored start time are
    detected, while tiny float rounding differences (the embedding plot rounds
    start times to 4 decimals) do not cause false alarms.

    Parameters
    ----------
    start_click : float
        start time stored in the click/embedding data
    start_metadata : float
        start time stored in the metadata labels
    tolerance : float, optional
        maximum allowed difference in seconds, by default 1e-2

    Returns
    -------
    bool
        True if the timestamps refer to the same segment
    """
    try:
        return abs(float(start_click) - float(start_metadata)) <= tolerance
    except (TypeError, ValueError):
        return False

class SpectrogramPlot:
    """
    Utility for creating and updating the dashboard spectrogram figure.
    """

    def __init__(
        self, audio_dir, loader, model_name, panel_static_text, paths, **kwargs
    ):
        """
        Initialize the spectrogram plotting utility.

        Parameters
        ----------
        audio_dir : str or pathlib.Path
            directory containing the audio files
        loader : object
            embeddings loader used to fetch model metadata
        model_name : object
            dropdown/selector widget holding the model options
        panel_static_text : object
            panel text widget updated with the playback metadata
        **kwargs
            additional plotting options (e.g., spec colorscale, height,
            padding, bool_change_speed, new_speed)
        """
        self.audio_dir = audio_dir
        self.panel_static_text = panel_static_text
        self.all_sample_rates = {}
        self.all_segment_lengths = {}
        self.paths = paths
        # Widget (or bool) that is True whenever noise-filtered embeddings are
        # displayed. In that mode the clicked point indices no longer map onto
        # the metadata labels, so the timestamp verification is skipped.
        self.remove_noise_widget = kwargs.pop("remove_noise", None)
        for model in model_name.options:
            loader.get_data(model, "time_of_day")
            metadata = loader.embeds[model]["metadata"]
            self.all_sample_rates[model] = metadata["sample_rate (Hz)"]
            self.all_segment_lengths[model] = metadata[
                "segment_length (samples)"
            ]
        self.kwargs = kwargs

    def _update_spec_obj(self, model, bool_autoplay_audio):
        """
        Cache the active model and its audio playback settings.

        Parameters
        ----------
        model : str
            name of the active model
        bool_autoplay_audio : bool
            whether audio should autoplay when a point is selected
        """
        self.model_name = model
        self.sample_rate = self.all_sample_rates[model]
        self.segment_length = self.all_segment_lengths[model]
        self.bool_autoplay_audio = bool_autoplay_audio

    def _cache_selected_points(self, selected_points):
        """
        Cache the currently selected plot points.

        Parameters
        ----------
        selected_points : list
            currently selected embedding points
        """
        self.selected_points = selected_points

    @staticmethod
    def dummy_image(title):
        """
        Create a placeholder figure to display before a point is selected.

        Parameters
        ----------
        title : str
            title of the placeholder image

        Returns
        -------
        plotly.graph_objects.Figure
            dummy image figure
        """
        # initial dummy figure, as a placeholder
        fig = px.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        fig.update_layout(
            title=title,
            margin=dict(l=20, r=20, t=40, b=20),
            height=settings.spectrogram_plot_height,
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    def update_spectrogram(
        self, clickData=None, play_btn=None, autoplay_radio=None
    ):
        """
        Update the spectrogram figure based on the clicked embedding point.

        Parameters
        ----------
        clickData : dict or None
            plotly click event data
        play_btn : object or None
            play button widget (unused)
        autoplay_radio : object or None
            autoplay radio widget (unused)

        Returns
        -------
        plotly.graph_objects.Figure
            updated spectrogram figure
        """
        # Sohw black image initially
        if not clickData:
            return SpectrogramPlot.dummy_image(
                "Click an embedding to see the corresponding spectrogram"
            )

        # Extract data from click
        point_data = clickData.get("customdata", [None] * 8)
        (
            audiofilename,
            start_s,
            end_s,
            idx,
            label,
            variable_labels_json,
            label_id,
            model_name
        ) = point_data
        self.check_timestamp_of_click_data_against_metadata(model_name, idx, start_s)

        # Load Audio
        audio, file_stem = self.load_audio(start_s, end_s, audiofilename)
        spec_fig = self.create_specs(audio)

        self.update_text(
            start_s, end_s, audiofilename, label, variable_labels_json
        )

        return spec_fig

    def check_timestamp_of_click_data_against_metadata(self, model_name, idx, start_s):
        """
        Verify that the start timestamp of the clicked embedding point matches
        the start timestamp stored in the metadata labels of the corresponding
        model.

        This is a safety check to avoid displaying a spectrogram that does not
        correspond to the audio that was actually used to generate the clicked
        embedding (e.g. after switching ``only_embed_annotations`` or changing
        other settings that affect how the timestamps are generated). The check
        is advisory only: on any failure a warning is logged and the spectrogram
        display continues normally.

        To keep the delay added by this check as small as possible, the
        ``start`` column of the metadata labels is read from disk only once per
        model and cached for all subsequent clicks.

        Parameters
        ----------
        model_name : str or None
            name of the model the clicked point belongs to
        idx : int
            row index of the clicked point in the metadata labels
        start_s : float
            start timestamp (seconds) of the clicked point
        """
        if model_name is None or self.paths is None:
            return

        # When noise-filtered embeddings are shown the clicked point indices
        # no longer map onto the metadata labels, so the check is meaningless.
        remove_noise = getattr(
            self.remove_noise_widget, "value", self.remove_noise_widget
        )
        if remove_noise:
            return

        if not hasattr(self, "_metadata_starts_cache"):
            self._metadata_starts_cache = {}

        if model_name not in self._metadata_starts_cache:
            self._metadata_starts_cache[model_name] = self._load_metadata_starts(
                model_name
            )

        start_from_metadata = self._metadata_starts_cache[model_name]
        if start_from_metadata is None:
            return

        try:
            start_from_metadata = float(start_from_metadata.iloc[idx])
        except (IndexError, TypeError, KeyError):
            logger.warning(
                f"Could not find a metadata label at index {idx} for model "
                f"{model_name}. Skipping the timestamp verification."
            )
            return

        if not timestamps_match(start_s, start_from_metadata):
            logger.warning(
                "\nThe timestamps of the clicked point do not match with the timestamps "
                "of the metadata file. This means the displayed spectrogram might not "
                "correspond to the audio that was actually used to generate this embedding. "
                "Please double check that your crucial settings have not changed since "
                "you generated embeddings. This means that you have not switched the value "
                "of only_embed_annotations or other parameters that would affect how "
                "the timestamps are generated. If this error persists, remove the "
                "dim_reduced_embeddings, as these contain the timestamps that are "
                "used for the visualization of embedding points.\n"
            )

    def _load_metadata_starts(self, model_name):
        """
        Load the ``start`` column of the metadata labels for a model once.

        Reads ``metadata_labels.csv`` when available and falls back to
        ``metadata_labels.parquet``. Returns ``None`` (without raising) when
        no metadata labels file can be found so the dashboard keeps working.

        Parameters
        ----------
        model_name : str
            name of the model

        Returns
        -------
        pandas.Series or None
            the ``start`` column of the metadata labels, or None if unavailable
        """
        try:
            labels_path = self.paths(model_name).labels_path
        except (AttributeError, TypeError):
            return None

        csv_path = labels_path / "metadata_labels.csv"
        parquet_path = labels_path / "metadata_labels.parquet"

        try:
            if csv_path.exists():
                return pd.read_csv(
                    csv_path, usecols=["start"], index_col=False
                )["start"]
            if parquet_path.exists():
                return pd.read_parquet(parquet_path, columns=["start"])["start"]
        except Exception as e:
            logger.warning(
                "Could not read the metadata labels to verify the clicked "
                f"timestamp: {e}"
            )
            return None

        logger.warning(
            f"No metadata_labels file found for model {model_name}. "
            "Skipping the timestamp verification of the clicked points."
        )
        return None

    def update_text(
        self, start_s, end_s, audiofilename, label, variable_labels_json=None
    ):
        """
        Update the static text panel with the selected audio metadata.

        Parameters
        ----------
        start_s : float
            start offset of the audio segment in seconds
        end_s : float
            end offset of the audio segment in seconds
        audiofilename : str
            name of the audio file
        label : str
            label of the selected point
        variable_labels_json : str or None
            JSON string with additional variable labels
        """
        # Parse variable labels from JSON
        variable_labels_html = ""
        if variable_labels_json:
            try:
                import json

                var_labels_dict = json.loads(variable_labels_json)
                for key, value in var_labels_dict.items():
                    if '_species' in key:
                        clean_species = [v for v in value if not v == '']
                        variable_labels_html += f"<br><b>{key}</b> = {', '.join(clean_species)}; "
                    elif '_confidence' in key:
                        clean_conf = [f'{v:.3f}' for v in value if not v == 0]
                        variable_labels_html += f"<br><b>{key}</b> = {', '.join(clean_conf)}; "
                    else:
                        variable_labels_html += f"<b>{key}</b> = {value}; "
            except:
                pass

        self.panel_static_text.visible = True
        self.panel_static_text.value = f"""
            <b>model sample rate</b> = {self.sample_rate} Hz; 
            <b>model segment_length</b> = {self.segment_length} samples; <br>
            <b>filename</b> = {audiofilename}; <br>
            <b>offset</b> = {start_s} s;        
            <b>duration</b> = {end_s - start_s} s;        
            <b>label</b> = {label}; <br>
            {variable_labels_html}
        """

    def create_specs(self, audio):
        """
        Create the spectrogram figure for an audio segment.

        Parameters
        ----------
        audio : np.ndarray
            audio samples to visualize

        Returns
        -------
        plotly.graph_objects.Figure
            spectrogram figure
        """
        S = np.abs(lb.stft(audio, win_length=1024))
        S_dB = lb.amplitude_to_db(S, ref=np.max)
        f_max, S_dB = self.set_axis_lims_dep_sr(S_dB)
        fig = px.imshow(
            S_dB,
            origin="lower",
            aspect="auto",
            y=np.linspace(0, f_max, S_dB.shape[0]),
            x=np.linspace(
                0, self.segment_length / self.sample_rate, S_dB.shape[1]
            ),
            labels={"x": "time (s)", "y": "freq (Hz)"},
            color_continuous_scale=self.kwargs.get("spec_colorscale"),
        )
        fig.update_layout(
            height=self.kwargs.get("spectrogram_plot_height"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    def play_audio(self, event):
        """
        Play the currently cached audio segment.

        Parameters
        ----------
        event : object
            widget event triggering playback (unused)
        """
        import sounddevice as sd

        if not hasattr(self, "audio"):
            return
        audio = tukey(len(self.audio), alpha=0.01) * self.audio

        sd.play(
            audio, self.sample_rate
        )

    def load_audio(self, start, end, filename):
        """
        Load an audio segment from file and return it at the model sample rate.

        Parameters
        ----------
        start : float
            start offset in seconds
        end : float
            end offset in seconds
        filename : str
            name of the audio file

        Returns
        -------
        tuple of (np.ndarray, str)
            loaded audio samples and the file stem
        """
        path = Path(self.audio_dir) / filename
        if not self.kwargs.get("bool_change_speed"):
            import h5py
            idx = int(start / (self.segment_length / self.sample_rate))
            with h5py.File(path, 'r') as data:
                audio = data['audio'][idx]
            # audio, self.orig_sr = lb.load(
            #     path,
            #     sr=self.sample_rate,
            #     offset=float(start),
            #     duration=float(end) - float(start),
            # )
        else:
            audio, self.orig_sr = lb.load(
                path,
                sr=None,
                offset=float(start),
                duration=float(end) - float(start),
            )
            audio = lb.resample(
                audio,
                orig_sr=int(self.orig_sr * self.kwargs.get("new_speed")),
                target_sr=self.sample_rate,
            )

        if (
            float(end) - float(start)
        ) * self.sample_rate < self.segment_length:
            audio = tukey(len(audio), alpha=0.01) * audio
            return_audio = lb.util.fix_length(
                audio, size=self.segment_length, mode=self.kwargs["padding"]
            )
        else:
            return_audio = audio
        self.audio = return_audio
        return return_audio, path.stem

    def set_axis_lims_dep_sr(self, S_dB):
        """
        Trim the spectrogram to the Nyquist frequency band of the sample rate.

        Parameters
        ----------
        S_dB : np.ndarray
            spectrogram magnitudes in dB

        Returns
        -------
        tuple of (float, np.ndarray)
            maximum frequency shown and the trimmed spectrogram
        """
        f_max = self.sample_rate / 2
        reduce = self.sample_rate / (f_max * 2)
        S_dB = S_dB[: int(S_dB.shape[0] / reduce), :]
        return f_max, S_dB
