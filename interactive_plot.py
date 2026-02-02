import dash
from dash import dcc, html
import plotly.express as px
import numpy as np

# --- Existing Constants ---
# (Kept from your snippet)
SAMPLE_RATE = 48_000
EXAMPLE_WINDOW_SECONDS = 3.
SPEC_WIN_LEN = 1024
PAD_FUNC = 'minimum'

import librosa as lb
import sounddevice as sd
from pathlib import Path

DT_FORMAT = ''
AMP = 5
LOAD_PATH = None
PREPROC = {}
PREPROC['downsample'] = True
PREPROC['downsample_sr'] = 2000
PREPROC['model_sr'] = 6000
SPEC_HEIGHT = 800


SRCS = {
    'anura': '/mnt/swap/Work/Data/Amphibians/AnuranSet/AnuranSet',
    'wabad': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/WABAD',
    'arctic': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/ArcticBirdSounds/DataS1/audio_annots'
    }

    

def dummy_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)

def build_dash_layout(fig, title):
    # Main wrapper
    return html.Div([
        html.H1(title, style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Controls Row (Placed at the top)
        html.Div([
            html.Div([
                html.B("File: "), html.Span(id='graph_heading', children='...'),
                html.Br(),
                html.Button(id="play_audio_btn", children="Play Sound", style={'marginRight': '10px'}),
                dcc.RadioItems(
                    ['Autoplay on', 'Autoplay off'], 
                    'Autoplay off', 
                    id='radio_autoplay',
                    inline=True
                ),
            ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'})
        ], style={'width': '95%', 'margin': '0 auto 20px auto'}),

        # Graphs Row
        html.Div([
            # --- Left Figure (Passed Figure) ---
            html.Div([
                dcc.Graph(
                    id="bar_chart",
                    figure=fig,
                    style={'height': '700px'} # Fixed height to prevent collapse
                )
            ], style={'flex': '1.3', 'marginRight': '10px', 'border': '1px solid #ddd'}),

            # --- Right Figure (Spectrogram) ---
            html.Div([
                dcc.Graph(
                    id="table_container", 
                    figure=px.imshow(dummy_image()),
                    style={'height': '700px'} 
                ),
            ], style={'flex': '1', 'border': '1px solid #ddd'}) # Spectrogram gets slightly less width to help with squareness
            
        ], style={
            'width': '95%', 
            'margin': '0 auto', 
            'display': 'flex', 
            'flexDirection': 'row',
            'alignItems': 'flex-start'
        })
    ])

def create_specs(audio):
    # Standard spectrogram logic
    import librosa as lb
    S = np.abs(lb.stft(audio, win_length=SPEC_WIN_LEN))
    S_dB = lb.amplitude_to_db(S, ref=np.max)
    
    f_max, S_dB = set_axis_lims_dep_sr(S_dB)
    # We remove the hardcoded height=SPEC_HEIGHT here so the 
    # CSS 'aspect-ratio' of the container can take control.
    fig = px.imshow(
        S_dB, 
        origin='lower', 
        aspect='auto',
        y=np.linspace(0, f_max, S_dB.shape[0]),
        x=np.linspace(0, EXAMPLE_WINDOW_SECONDS, S_dB.shape[1]),
        labels={'x': 'time (s)', 'y': 'freq (Hz)'},
        color_continuous_scale='Viridis'
    )
    
    # Remove margins to make it fill the 4:3 box cleanly
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig

# (Remaining helper functions load_audio, play_audio, etc. stay as they were)


def interactive_plot(fig, title='Testing'):
    # Note: Added 'title' to matches your call inside build_dash_layout
    app = dash.Dash(__name__) 
    
    # Restructured to pass the figure and title
    app.layout = build_dash_layout(fig, title)

    @app.callback(
        dash.Output("table_container", "figure"),
        dash.Output("graph_heading", "children"),
        dash.Input("bar_chart", "clickData"),
        dash.Input("play_audio_btn", "n_clicks"),
        dash.Input("radio_autoplay", "value"))
    def fig_click(clickData, play_btn, autoplay_radio):
        # Default state
        if not clickData:
            return px.imshow(dummy_image()), "file: ..."
        
        # Extract data from click
        point_data = clickData['points'][0].get('customdata', [None]*6)
        dataset, filename, start_s, cluster, idx, end_s = point_data
        
        audio, sr, file_stem = load_audio(start_s, end_s, filename, dataset)
        # import h5py
        # file_name = 'unknown_sounds_2_within_file_1_diff_file_3s.h5'
        # data_file = h5py.File(file_name)
        # audio = data_file['audio'][int(idx)]
        spec_fig = create_specs(audio)
        
        # Update title info
        heading = html.P([
            f"file: {file_stem}", html.Br(),
            f"time in file: {start_s}"
        ])

        # Handle Audio logic
        if autoplay_radio == "Autoplay on" or dash.ctx.triggered_id == "play_audio_btn":
            play_audio(audio, sr)
            
        return spec_fig, heading

    app.run(debug=False)


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
    
    
def load_audio(start, end, filename, dataset):
    path = list(Path(SRCS[dataset]).rglob(filename))[0]
    
    audio, sr = lb.load(path, 
                        offset=float(start), 
                        # sr=SAMPLE_RATE, 
                        # duration = EXAMPLE_WINDOW_SECONDS)
                        duration = float(end)-float(start))
    
    audio_padded = lb.util.fix_length(
                audio,
                size=int(EXAMPLE_WINDOW_SECONDS * sr),
                mode=PAD_FUNC,
            )
    return audio_padded, sr, path.stem
 
def set_axis_lims_dep_sr(S_dB):
    if PREPROC['downsample']:
        f_max = PREPROC['downsample_sr'] / 2
        reduce = PREPROC['model_sr'] / (f_max * 2)
        S_dB = S_dB[:int(S_dB.shape[0] / reduce), :]
    else:
        f_max = PREPROC['model_sr'] / 2 
    return f_max, S_dB