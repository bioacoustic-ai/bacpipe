import dash
import yaml
import numpy as np
import pandas as pd
import librosa as lb
import re
import datetime as dt
import sounddevice as sd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

DT_FORMAT = ''
AMP = 5
LOAD_PATH = None
PREPROC = {}
PREPROC['downsample'] = True
PREPROC['downsample_sr'] = 2000
PREPROC['model_sr'] = 6000
SPEC_WIN_LEN = 1024
SPEC_HEIGHT = 800

SAMPLE_RATE = 48_000
EXAMPLE_WINDOW_SECONDS = 3.

SRCS = {
    'anura': '/mnt/swap/Work/Data/Amphibians/AnuranSet/AnuranSet',
    'wabad': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/WABAD',
    'arctic': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/ArcticBirdSounds/DataS1/audio_annots'
    }

    

def dummy_image():
    return np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                ], dtype=np.uint8)
    
def build_dash_layout(fig):
    dash.html.Col
    return dash.html.Div(
        [
            dash.html.Div([
                dash.dcc.Graph(
                    id="bar_chart",
                    figure=fig
                            )
            ], style={'width': '75%', 'display': 'inline-block',
                      'vertical-align': 'top'}),
            
            dash.html.Div([
                    dash.html.H2(children='Spectrogram'),
                    dash.html.Div(id='graph_heading', children='file: ...'),
                	dash.html.Button(id="play_audio_btn", children="Play Sound", 
                                  n_clicks = 0),
                    dash.dcc.RadioItems(['Autoplay on', 'Autoplay off'], 
                                    'Autoplay off', id='radio_autoplay'),
                    dash.dcc.Graph(id="table_container", 
                                   figure = px.imshow(dummy_image(), 
                                                      height = 500)
                                   ),
            ], style={'width': '25%', 'display': 'inline-block',
                      'vertical-align': 'bottom'})
        ]
    )    



def interactive_plot(embeddings, time_in_file, files_array, fig, title='Testing'):

    data = dict()
    
    data.update({'x': embeddings['x'],
                 'y': embeddings['y'],
                 'time_in_file' : time_in_file,
                 'filename' : files_array})

    app = dash.Dash(__name__, external_stylesheets=['./styles.css'])
    app.layout = build_dash_layout(data, title, fig)

    @app.callback(
        Output("table_container", "figure"),
        Output("graph_heading", "children"),
        Input("bar_chart", "clickData"),
        Input("play_audio_btn", "n_clicks"),
        Input("radio_autoplay", "value"))
    
    def fig_click(clickData, play_btn, autoplay_radio):
        if not clickData:
            return (px.imshow(dummy_image(), height = 400),
                    "file: ...")
        
        else:
            point_data = clickData['points'][0]['customdata']
            dataset = point_data[0]
            filename = point_data[1]
            time_in_file = point_data[2]
            cluster = point_data[3]
            
            audio, sr, file_stem = load_audio(time_in_file, filename, dataset)
            spec = create_specs(audio)
            if autoplay_radio == "Autoplay on":
                play_audio(audio, sr)
            
        if "play_audio_btn" == dash.ctx.triggered_id:
            play_audio(audio, sr)
            
        title = dash.html.P([f"file: {file_stem.split('.Table')[0]}",
                             dash.html.Br(),
                            f"time in file: {time_in_file}",
                             dash.html.Br(),
                            f"location: {file_stem.split('_')[-1]}"])
            
        return spec, title
    
    app.run(debug = False)
    
def smoothing_func(num_samps, func='sin'):
    return getattr(np, func)(np.linspace(0, np.pi/2, num_samps))

def fade_audio(audio):
    perc_aud = np.linspace(0, len(audio), 100).astype(int)
    return [*[0]*perc_aud[3], 
            *audio[perc_aud[3]:perc_aud[7]]*AMP
                *smoothing_func(perc_aud[7]-perc_aud[3]),
            *audio[perc_aud[7]:perc_aud[70]]*AMP, 
            *audio[perc_aud[70]:perc_aud[93]]*AMP
                *smoothing_func(perc_aud[93]-perc_aud[70], func='cos'),
            *[0]*(perc_aud[-1]-perc_aud[93])]
    
def play_audio(audio, sr):
    sd.play(fade_audio(audio), sr)
    
    
def load_audio(t_s, filename, dataset):
    path = list(Path(SRCS[dataset]).rglob(filename))[0]
    
    audio, sr = lb.load(path, 
                        offset=float(t_s), 
                        sr=SAMPLE_RATE, 
                        duration = EXAMPLE_WINDOW_SECONDS)
    return audio, sr, path.stem
 
def set_axis_lims_dep_sr(S_dB):
    if PREPROC['downsample']:
        f_max = PREPROC['downsample_sr'] / 2
        reduce = PREPROC['model_sr'] / (f_max * 2)
        S_dB = S_dB[:int(S_dB.shape[0] / reduce), :]
    else:
        f_max = PREPROC['model_sr'] / 2 
    return f_max, S_dB

def create_specs(audio):
    S = np.abs(lb.stft(audio, win_length = SPEC_WIN_LEN))
    S_dB = lb.amplitude_to_db(S, ref=np.max)
    f_max, S_dB = set_axis_lims_dep_sr(S_dB)
    
    if PREPROC['downsample']:
        f_max = PREPROC['downsample_sr']
    else:
        f_max = SAMPLE_RATE/2

    fig = px.imshow(S_dB, origin='lower', 
                    aspect = 'auto',
                    y = np.linspace(SAMPLE_RATE/40, 
                                    f_max, 
                                    S_dB.shape[0]),
                    x = np.linspace(0, 
                                    EXAMPLE_WINDOW_SECONDS, 
                                    S_dB.shape[1]),
                    labels = {'x' : 'time in s', 
                            'y' : 'frequency in Hz'},
                    height = SPEC_HEIGHT)
    return fig