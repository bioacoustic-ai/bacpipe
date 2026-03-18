import pandas as pd
import numpy as np
import plotly.express as px
import librosa as lb
from scipy.signal.windows import tukey
from pathlib import Path

SPECTROGRAM_PLOT_HEIGHT = 300
SPEC_COLORSCALE = 'Viridis'

class SpectrogramPlot:
    def __init__(self, audio_dir, loader, model_name, panel_static_text, **kwargs):
        self.audio_dir = audio_dir
        self.panel_static_text = panel_static_text
        self.all_sample_rates = {}
        self.all_segment_lengths = {}
        for model in model_name.options:
            loader.get_data(model, 'time_of_day')
            metadata = loader.embeds[model]['metadata']
            self.all_sample_rates[model] = metadata['sample_rate (Hz)']
            self.all_segment_lengths[model] = metadata['segment_length (samples)']
        self.kwargs = kwargs
        
    def _update_spec_obj(self, model, bool_autoplay_audio):
        self.model_name = model
        self.sample_rate = self.all_sample_rates[model]
        self.segment_length = self.all_segment_lengths[model]
        self.bool_autoplay_audio = bool_autoplay_audio
        
    def _cache_selected_points(self, selected_points):
        self.selected_points = selected_points

    @staticmethod
    def dummy_image(title):
        # initial dummy figure, as a placeholder
        fig = px.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        fig.update_layout(
            title=title, 
            margin=dict(l=20, r=20, t=40, b=20),
            height=SPECTROGRAM_PLOT_HEIGHT,
            xaxis={'visible': False}, 
            yaxis={'visible': False}
        )
        return fig
    
    
    def update_spectrogram(
        self, clickData=None, play_btn=None, autoplay_radio=None
        ):
        # Sohw black image initially
        if not clickData:
            return SpectrogramPlot.dummy_image(
                "Click an embedding to see the corresponding spectrogram"
            )
        
        # Extract data from click
        point_data = clickData.get('customdata', [None]*6)
        audiofilename, start_s, end_s, idx, label, label_id = point_data
        
        # Load Audio
        audio, file_stem = self.load_audio(start_s, end_s, audiofilename)
        spec_fig = self.create_specs(audio)
        
        self.update_text(start_s, end_s, audiofilename, label)
            
        return spec_fig


    def update_text(self, start_s, end_s, audiofilename, label):
        self.panel_static_text.visible=True
        self.panel_static_text.value = f"""
            <b>model sample rate</b> = {self.sample_rate} Hz; 
            <b>model segment_length</b> = {self.segment_length} samples; <br>
            <b>filename</b> = {audiofilename}; <br>
            <b>offset</b> = {start_s} s;        
            <b>duration</b> = {end_s - start_s} s;        
            <b>label</b> = {label}
        """
    
    def create_specs(self, audio):
        S = np.abs(lb.stft(audio, win_length=1024))
        S_dB = lb.amplitude_to_db(S, ref=np.max)
        f_max, S_dB = self.set_axis_lims_dep_sr(S_dB)
        fig = px.imshow(
            S_dB, origin='lower', aspect='auto',
            y=np.linspace(
                0, 
                f_max, 
                S_dB.shape[0]
                ),
            x=np.linspace(
                0, 
                self.segment_length / self.sample_rate, 
                S_dB.shape[1]
                ),
            labels={'x': 'time (s)', 'y': 'freq (Hz)'}, 
            color_continuous_scale=SPEC_COLORSCALE,
        )
        fig.update_layout(
            height=SPECTROGRAM_PLOT_HEIGHT, 
            margin=dict(l=20, r=20, t=20, b=20)
            )
        return fig
        
    def play_audio(self, event):
        import sounddevice as sd
        if not hasattr(self, 'audio'):
            return
        if True:#self.use_tukey_filter:
            audio = tukey(len(self.audio), alpha=0.01) * self.audio
        else:
            audio = self.audio
        sd.play(audio, self.sample_rate)
        
    def load_audio(self, start, end, filename):
        path = Path(self.audio_dir) / filename
        audio, sr = lb.load(
            path, 
            sr=self.sample_rate, 
            offset=float(start), 
            duration=float(end)-float(start)
            )
        if (
            (float(end)-float(start))
            / self.sample_rate < self.segment_length
            ):
            if True:#self.use_tukey_filter:
                audio = tukey(len(audio), alpha=0.01) * audio
            return_audio = lb.util.fix_length(
                audio, 
                size=int((end - start) * sr), 
                mode=self.kwargs['padding']
                )
        else:
            return_audio = audio
        self.audio = return_audio
        return return_audio, path.stem
    
    def set_axis_lims_dep_sr(self, S_dB):
        f_max = self.sample_rate / 2
        reduce = self.sample_rate / (f_max * 2)
        S_dB = S_dB[:int(S_dB.shape[0] / reduce), :]
        return f_max, S_dB

