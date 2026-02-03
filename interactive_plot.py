import dash
from dash import dcc, html, ctx
import plotly.express as px
import numpy as np
import librosa as lb
import sounddevice as sd
from pathlib import Path
from scipy.signal.windows import tukey
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt

# --- Helper Functions ---

def dummy_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)

def build_dash_layout(title, 
                      clust_opts, species_opts, eval_opts, 
                      def_clust, def_species, def_eval):
    # Main wrapper
    return html.Div([
        html.H1(title, style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # --- Dropdown Controls Row ---
        html.Div([
            html.Div([
                html.Label("Clustering:"),
                dcc.Dropdown(
                    id='dropdown_clust', 
                    options=clust_opts, 
                    value=def_clust, 
                    clearable=False
                )
            ], style={'width': '30%'}),
            
            html.Div([
                html.Label("Species:"),
                dcc.Dropdown(
                    id='dropdown_species', 
                    options=species_opts, 
                    value=def_species,
                    clearable=False
                )
            ], style={'width': '30%'}),
            
            html.Div([
                html.Label("Eval Task:"),
                dcc.Dropdown(
                    id='dropdown_eval', 
                    options=eval_opts, 
                    value=def_eval,
                    clearable=False
                )
            ], style={'width': '30%'}),
            
        ], style={'width': '95%', 'margin': '0 auto 15px auto', 'display': 'flex', 'justifyContent': 'space-between'}),

        # --- Audio Controls Row ---
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

        # --- Graphs Row ---
        html.Div([
            # Left Figure (Main Scatter Plot)
            # Note: We do NOT pass 'figure=' here. The callback will populate it immediately on load.
            html.Div([
                dcc.Graph(
                    id="bar_chart",
                    style={'height': '700px'} 
                )
            ], style={'flex': '1.3', 'marginRight': '10px', 'border': '1px solid #ddd'}),

            # Right Figure (Spectrogram)
            html.Div([
                dcc.Graph(
                    id="table_container", 
                    figure=px.imshow(dummy_image()),
                    style={'height': '700px'} 
                ),
            ], style={'flex': '1', 'border': '1px solid #ddd'}) 
            
        ], style={'width': '95%', 'margin': '0 auto', 'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start'})
    ])


class InteractivePlot():
        
    def __init__(
        self,
        data_file,
        clust_results,
        label_dict_bool,
        clust_dict,
        umaps,
        src_path = '/media/siriussound/Extreme SSD/Recordings',
        sample_rate = 48_000,
        example_window_seconds = 3.,
        spec_win_len = 1024,
        pad_func = 'constant',
        ):
        
        self.sample_rate = sample_rate
        self.example_window_seconds = example_window_seconds
        self.spec_win_len = spec_win_len
        self.pad_func = pad_func
        
        self.labels = data_file['labels'][:].astype(str)
        self.datasets = data_file['datasets'][:].astype(str)
        self.filenames = data_file['filenames'][:].astype(str)
        self.starts = data_file['starts'][:].astype(str)
        self.ends = data_file['ends'][:].astype(str)
        self.clust_results = clust_results
        self.label_dict_bool = label_dict_bool
        self.clust_dict = clust_dict
        self.umaps = umaps

        self.dt_format = ''
        self.amp = 5
        self.load_path = None
        self.preproc = {}
        self.preproc['downsample'] = True
        self.preproc['downsample_sr'] = 2000
        self.preproc['model_sr'] = 6000
        self.spec_height = 800

        if pad_func == 'constant':
            self.use_tukey_filter = True
        else:
            self.use_tukey_filter = False

        self.srcs = {
            'anura': f'{src_path}/terrestrial/Amphibians/AnuranSet/AnuranSet',
            'wabad': f'{src_path}/terrestrial/Birds/WABAD',
            'arctic': f'{src_path}/terrestrial/Birds/ArcticBirdSounds/DataS1/audio_annots'
            }

    # --- MAIN INTERACTIVE METHOD ---
    def interactive_plot(self, 
                         model_name, # Pass the model name here
                         clust_opts, species_opts, eval_opts, # LISTS of options
                         def_clust, def_species, def_eval,    # SINGLE default values
                         title='Testing', port=8050):

            app = dash.Dash(__name__) 
            
            # Initialize Layout with Dropdowns
            app.layout = build_dash_layout(
                title, 
                clust_opts, species_opts, eval_opts,
                def_clust, def_species, def_eval
            )

            # --- CALLBACK 1: Update Scatter Plot when Dropdowns Change ---
            @app.callback(
                dash.Output("bar_chart", "figure"),
                dash.Input("dropdown_clust", "value"),
                dash.Input("dropdown_species", "value"),
                dash.Input("dropdown_eval", "value")
            )
            def update_scatter_plot(clust_val, species_val, eval_val):
                # This function is triggered on load (using defaults) and whenever a user changes a dropdown
                print(f"Updating plot for: {clust_val}, {species_val}, {eval_val}")
                fig = self.plotly_mutual_information(model_name, clust_val, species_val, eval_val)
                return fig

            # --- CALLBACK 2: Update Spectrogram when Scatter Plot is Clicked ---
            @app.callback(
                dash.Output("table_container", "figure"),
                dash.Output("graph_heading", "children"),
                dash.Input("bar_chart", "clickData"),
                dash.Input("play_audio_btn", "n_clicks"),
                dash.Input("radio_autoplay", "value")
            )
            def update_spectrogram(clickData, play_btn, autoplay_radio):
                # Default state
                if not clickData:
                    return px.imshow(dummy_image()), "Click a point to see spectrogram"
                
                # Extract data from click
                point_data = clickData['points'][0].get('customdata', [None]*6)
                dataset, filename, start_s, cluster, idx, end_s = point_data
                
                # Load Audio
                audio, sr, file_stem = self.load_audio(start_s, end_s, filename, dataset)
                spec_fig = self.create_specs(audio)
                
                # Update title info
                heading = html.P([
                    f"file: {file_stem}", html.Br(),
                    f"time in file: {start_s}"
                ])

                # Handle Audio Playback
                # dash.ctx.triggered_id tells us exactly which input fired the callback
                trigger_id = dash.ctx.triggered_id
                
                should_play = (autoplay_radio == "Autoplay on" and trigger_id == "bar_chart") or \
                              (trigger_id == "play_audio_btn")

                if should_play:
                    self.play_audio(audio, sr)
                    
                return spec_fig, heading

            # Run the app
            print(f"Starting Dash on port {port}...")
            app.run(debug=False, port=port)


    # --- HELPERS ---
    def create_specs(self, audio):
        S = np.abs(lb.stft(audio, win_length=self.spec_win_len))
        S_dB = lb.amplitude_to_db(S, ref=np.max)
        f_max, S_dB = self.set_axis_lims_dep_sr(S_dB)
        fig = px.imshow(
            S_dB, origin='lower', aspect='auto',
            y=np.linspace(0, f_max, S_dB.shape[0]),
            x=np.linspace(0, self.example_window_seconds, S_dB.shape[1]),
            labels={'x': 'time (s)', 'y': 'freq (Hz)'}, color_continuous_scale='Viridis'
        )
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        return fig

    def smoothing_func(self, num_samps, func='sin'):
        return getattr(np, func)(np.linspace(0, np.pi/2, num_samps))

    def fade_audio(self, audio):
        # Your custom fade logic
        perc_aud = np.linspace(0, len(audio), 100).astype(int)
        return [*[0]*perc_aud[3], 
                *audio[perc_aud[3]:perc_aud[7]]*self.amp
                    *self.smoothing_func(perc_aud[7]-perc_aud[3]),
                *audio[perc_aud[7]:perc_aud[70]]*self.amp, 
                *audio[perc_aud[70]:perc_aud[93]]*self.amp
                    *self.smoothing_func(perc_aud[93]-perc_aud[70], func='cos'),
                *[0]*(perc_aud[-1]-perc_aud[93])]
        
    def play_audio(self, audio, sr):
        sd.play(self.fade_audio(audio), sr)
        
    def load_audio(self, start, end, filename, dataset):
        path = list(Path(self.srcs[dataset]).rglob(filename))[0]
        audio, sr = lb.load(path, offset=float(start), duration=float(end)-float(start))
        if self.use_tukey_filter:
            audio = tukey(len(audio), alpha=0.01) * audio
        audio_padded = lb.util.fix_length(audio, size=int(self.example_window_seconds * sr), mode=self.pad_func)
        return audio_padded, sr, path.stem
    
    def set_axis_lims_dep_sr(self, S_dB):
        if self.preproc['downsample']:
            f_max = self.preproc['downsample_sr'] / 2
            reduce = self.preproc['model_sr'] / (f_max * 2)
            S_dB = S_dB[:int(S_dB.shape[0] / reduce), :]
        else:
            f_max = self.preproc['model_sr'] / 2 
        return f_max, S_dB
    
    def plotly_mutual_information(self, model, clust_name, species, eval_name):
        current_labels = [l if l == species else 'noise' for l in self.labels[self.label_dict_bool[eval_name][species]]]
        label_colors = {i: rgb2hex(c) for i, c in enumerate(plt.colormaps['tab20'].colors)}

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(model, 
                            f'{clust_name} {eval_name} {species} score={self.clust_results[model][clust_name][eval_name][species]:.4f}'),
            horizontal_spacing=0.1,
            vertical_spacing=0.12
        )
        
        hover = (
            '<b>{}</b>'
            '<br>cluster: %{{customdata[3]}}'
            '<br>dataset: %{{customdata[0]}}'
            '<br>filename: %{{customdata[1]}}'
            '<br>start: %{{customdata[2]}}'
            '<br>end: %{{customdata[5]}}'
            '<br>idx: %{{customdata[4]}}'
            '<extra></extra>'
        )

        for l_idx, label in enumerate(np.unique(current_labels)):
            b_array = self.label_dict_bool[eval_name][species]
            mask_mask = np.array(current_labels) == label
            
            # Bottom Plot (Sub-clusters)
            for clust_label in np.unique(self.clust_dict[model][clust_name][b_array][mask_mask]):
                clust_mask = self.clust_dict[model][clust_name][b_array][mask_mask] == clust_label
                fig.add_trace(go.Scatter(
                    x=self.umaps[model][b_array, 0][mask_mask][clust_mask],
                    y=self.umaps[model][b_array, 1][mask_mask][clust_mask],
                    mode='markers',
                    name=label,
                    marker=dict(size=8, color=label_colors[len(np.unique(current_labels))+clust_label]),
                    customdata=np.column_stack((
                        self.datasets[b_array][mask_mask][clust_mask], 
                        self.filenames[b_array][mask_mask][clust_mask], 
                        self.starts[b_array][mask_mask][clust_mask],
                        self.clust_dict[model][clust_name][b_array][mask_mask][clust_mask],
                        np.arange(len(self.starts))[b_array][mask_mask][clust_mask],
                        self.ends[b_array][mask_mask][clust_mask]
                    )),
                    hovertemplate=hover.format(label),
                    opacity=0.5,
                    showlegend=False
                ), row=2, col=1)
            
            # Top Plot (Main Clusters)
            fig.add_trace(go.Scatter(
                x=self.umaps[model][b_array, 0][mask_mask],
                y=self.umaps[model][b_array, 1][mask_mask],
                mode='markers',
                name=label,
                marker=dict(size=8, color=label_colors[l_idx]),
                customdata=np.column_stack((
                    self.datasets[b_array][mask_mask], 
                    self.filenames[b_array][mask_mask], 
                    self.starts[b_array][mask_mask],
                    [False] * len(self.starts[b_array][mask_mask]),
                    np.arange(len(self.starts))[b_array][mask_mask],
                    self.ends[b_array][mask_mask]
                )),
                hovertemplate=hover.format(label),
                opacity=0.5,
                showlegend=True
            ), row=1, col=1)
                
        fig.update_layout(height=1200, hovermode='closest')
        return fig