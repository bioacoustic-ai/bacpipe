import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bacpipe.main import model_specific_embedding_creation
from bacpipe import config, settings
import h5py

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from matplotlib.colors import rgb2hex

from sklearn.cluster import KMeans, SpectralClustering
from hdbscan import HDBSCAN


from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import homogeneity_score as HS

from tqdm import tqdm

config.overwrite = False
config.audio_dir = 'unknown_sounds_2_within_file_1_diff_file_3s'
config.already_computed = True
# config.models = ['birdnet']
config.dim_reduction_model='None'

loader_dict = model_specific_embedding_creation(
        # check_if_primary_combination_exists=False,
        # check_if_secondary_combination_exists=False,
        **vars(config), **vars(settings)
)

config.dim_reduction_model='umap'
# birdnet_embeddings = list(loader_dict['birdnet'].embedding_dict().values())[0]
umap_loader_dict = model_specific_embedding_creation(
        return_umap=True,
        # check_if_primary_combination_exists=False,
        # check_if_secondary_combination_exists=False,
        **vars(config), **vars(settings)
)
# birdnet_umaps_dict = list(umap_loader_dict['birdnet'].embedding_dict().values())[0].item()
# birdnet_umaps = [np.array([x, y]) for x, y in zip(birdnet_umaps_dict['x'], birdnet_umaps_dict['y'])]
# birdnet_umaps = np.array(birdnet_umaps)
embeddings = {}
umaps = {}
for model in loader_dict.keys():
        embeddings[model] = list(loader_dict[model].embedding_dict().values())[0]
        umap = list(umap_loader_dict[model].embedding_dict().values())[0].item()
        umaps[model] = np.array([np.array([x, y]) for x, y in zip(umap['x'], umap['y'])])

file_name = 'unknown_sounds_2_within_file_1_diff_file_3s.h5'
data_file = h5py.File(file_name)

labels = data_file['labels'][:].astype(str)
dataset = data_file['datasets'][:].astype(str)
filename = data_file['filenames'][:].astype(str)
start = data_file['starts'][:].astype(str)

if False:
        label_dict_bool = {}
        for eval_name in ['species_vs_species', 'species_vs_infile_noise', 'species_vs_other_noise', 'species_vs_all']:
                label_dict_bool[eval_name] = {}
                for species in np.unique([l for l in labels if not l in ['within_file', 'diff_file']]):
                        if eval_name == 'species_vs_species':
                                labels_bool = [
                                        True 
                                        if not l in ['within_file', 'diff_file'] 
                                        else False 
                                        for l in labels
                                        ]
                        elif eval_name == 'species_vs_infile_noise':
                                labels_bool = [
                                        True 
                                        if l == species
                                        else False 
                                        for l in labels
                                        ]
                                this_species_filenames = np.unique([
                                        f 
                                        for idx, f 
                                        in enumerate(filename) 
                                        if labels[idx] == species
                                        ])
                                within_files = [
                                        True 
                                        if fn in this_species_filenames 
                                        and labels[idx] == 'within_file' 
                                        else False 
                                        for idx, fn in enumerate(filename)
                                        ]
                                labels_bool = [a or b for a,b in zip (labels_bool, within_files)]
                        elif eval_name == 'species_vs_other_noise':
                                labels_bool = [
                                        True 
                                        if l in [species, 'diff_file']
                                        else False 
                                        for l in labels
                                        ]
                        elif eval_name == 'species_vs_all':
                                labels_bool = [True] * len(labels)
                        label_dict_bool[eval_name][species] = labels_bool

        clust_dict = {}

        embeds = {}
        for model_name, model_dict in loader_dict.items():
                embeds[model_name] = np.vstack(list(model_dict.embedding_dict().values()))

        ## compute clusterings

        # kmeans

        kmeans = KMeans(n_clusters=18) # because 15 species + noise for the within and diff file ...?
        hdb = HDBSCAN(min_cluster_size=15, min_samples=None)
        spec = SpectralClustering(n_clusters=16)

        for model_name in tqdm(loader_dict):
                clust_dict[model_name] = {}
                clust_dict[model_name]['kmeans'] = kmeans.fit_predict(embeds[model_name])
                clust_dict[model_name]['hdb'] = hdb.fit_predict(embeds[model_name])
                # clust_dict[model_name]['spec'] = spec.fit_predict(embeds[model_name])



        clust_results = {}

        for model_name, clusterings in clust_dict.items():
                clust_results[model_name] = {}
                for clust_name, clust in clust_dict[model_name].items():
                        clust_results[model_name][clust_name] = {}
                        for eval_name, eval_by_species in label_dict_bool.items():
                                clust_results[model_name][clust_name][eval_name] = {}
                                for species, b_array in eval_by_species.items():
                                        ground_truth = [1 if l == species else 0 for l in labels[b_array]]
                                        clustering = clust[b_array]
                                        clust_results[model_name][clust_name][eval_name].update({
                                                        species: HS(clustering, ground_truth)
                                        })
                                clust_results[model_name][clust_name][eval_name].update({
                                        'avg': np.mean(list(clust_results[model_name][clust_name][eval_name].values()))
                                })        
        with open('label_dict_bool.npy', 'wb') as f:
                np.save(f, label_dict_bool, allow_pickle=True)
        with open('clust_results.npy', 'wb') as f:
                np.save(f, clust_results, allow_pickle=True)
        with open('clust_dict.npy', 'wb') as f:
                np.save(f, clust_dict, allow_pickle=True)
else:
        clust_results = np.load('clust_results.npy', allow_pickle=True).item()
        clust_dict = np.load('clust_dict.npy', allow_pickle=True).item()
        label_dict_bool = np.load('label_dict_bool.npy', allow_pickle=True).item()
        
                
print(clust_results)

model = 'naturebeats'
clust_name = 'hdb'
species = 'Dendropsophus cruzi'
eval_name = 'species_vs_infile_noise'
# Convert string labels to numeric codes for coloring


def plotly_mutual_information(model, clust_name, species, eval_name):
        current_labels = [l if l == species else 'noise' for l in labels[label_dict_bool[eval_name][species]]]

        label_colors = {i: rgb2hex(c) for i, c in enumerate(plt.colormaps['tab20'].colors)}

        fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(model, f'{clust_name=} {eval_name=} {species=} score={clust_results[model][clust_name][eval_name][species]}.4f'),
        horizontal_spacing=0.1,
        vertical_spacing=0.12
        )
        hover = (
                '<b>{}</b>'
                '<br>cluster: %{{customdata[3]}}'
                '<br>dataset: %{{customdata[0]}}'
                '<br>filename: %{{customdata[1]}}'
                '<br>start: %{{customdata[2]}}'
                '<extra></extra>'
                )

        for l_idx, label in enumerate(np.unique(current_labels)):
                b_array = label_dict_bool[eval_name][species]
                mask_mask = np.array(current_labels) == label
                
                for clust_label in np.unique(clust_dict[model][clust_name][b_array][mask_mask]):
                        clust_mask = clust_dict[model][clust_name][b_array][mask_mask] == clust_label
                        fig.add_trace(go.Scatter(
                                x=umaps[model][b_array, 0][mask_mask][clust_mask],
                                y=umaps[model][b_array, 1][mask_mask][clust_mask],
                                mode='markers',
                                name=label,
                                marker=dict(size=8, color=label_colors[len(np.unique(current_labels))+clust_label]),
                                customdata=np.column_stack((
                                        dataset[b_array][mask_mask][clust_mask], 
                                        filename[b_array][mask_mask][clust_mask], 
                                        start[b_array][mask_mask][clust_mask],
                                        clust_dict[model][clust_name][b_array][mask_mask][clust_mask],
                                        )),
                                hovertemplate=hover.format(label),
                                opacity=0.5,
                                legendgroup=label,  # Group for shared legend
                                showlegend=False
                        ), row=1, col=2)
                        
                fig.add_trace(go.Scatter(
                        x=umaps[model][b_array, 0][mask_mask],
                        y=umaps[model][b_array, 1][mask_mask],
                        mode='markers',
                        name=label,
                        marker=dict(size=8, color=label_colors[l_idx]),
                        customdata=np.column_stack((
                                dataset[b_array][mask_mask], 
                                filename[b_array][mask_mask], 
                                start[b_array][mask_mask],
                                [False] * len(start[b_array][mask_mask])
                                )),
                        hovertemplate=hover.format(label),
                        opacity=0.5,
                        legendgroup=label,  # Group for shared legend
                        showlegend=True
                ), row=1, col=1)
                

        fig.update_layout(height=1200, width=1600, hovermode='closest')
        fig.write_html('test.html')
        fig.show()

plotly_mutual_information(model, clust_name, species, eval_name)
        
def plotly_compare_models():
        unique_labels = np.unique(labels)
        label_to_num = {label: i for i, label in enumerate(unique_labels)}

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        label_colors = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

        fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('BirdNET UMAP', 'Perch Bird UMAP', 'Perch V2 UMAP', 'NatureBeats UMAP'),
        horizontal_spacing=0.1,
        vertical_spacing=0.12
        )

        for label in unique_labels:
                mask = labels == label
                def trace(model, row, col, showlegend=False):
                        # Add to first subplot
                        fig.add_trace(go.Scatter(
                                x=umaps[model][mask, 0],
                                y=umaps[model][mask, 1],
                                mode='markers',
                                name=label,
                                marker=dict(size=8, color=label_colors[label]),
                                customdata=np.column_stack((dataset[mask], filename[mask], start[mask])),
                                hovertemplate=f'<b>{label}</b><br>dataset: %{{customdata[0]}}<br>filename: %{{customdata[1]}}<br>start: %{{customdata[2]}}<extra></extra>',
                                legendgroup=label,  # Group for shared legend
                                showlegend=showlegend
                        ), row=row, col=col)
                        
                trace('birdnet', 1, 1, showlegend=True)
                trace('perch_bird', 1, 2)
                trace('perch_v2', 2, 1)
                trace('naturebeats', 2, 2)

        fig.update_layout(height=1200, width=1600, hovermode='closest')
        fig.write_html('test.html')
        fig.show()