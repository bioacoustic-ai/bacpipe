import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bacpipe.main import model_specific_embedding_creation
from bacpipe import config, settings
import h5py

config.overwrite = False

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

# plt.figure()
# plt.scatter(birdnet_umaps[:, 0], birdnet_umaps[:, 1], label=labels.tolist())
# plt.legend()
# plt.savefig('test.png')



# plt.figure()
# # Convert string labels to numeric codes for coloring
# unique_labels = np.unique(labels)
# label_to_num = {label: i for i, label in enumerate(unique_labels)}
# numeric_labels = np.array([label_to_num[label] for label in labels])

# scatter = plt.scatter(birdnet_umaps[:, 0], birdnet_umaps[:, 1], c=numeric_labels, cmap='tab20')
# cbar = plt.colorbar(scatter, label='Species')
# # Map colorbar ticks back to species names
# cbar.set_ticks(range(len(unique_labels)))
# cbar.set_ticklabels(unique_labels)
# plt.savefig('test.png')





clust_dict = {}

for model_name, model_dict in loader_dict.items():
        # clustering = np.load(model_dict.paths.clust_path / 'clust_labels.npy', allow_pickle=True).item()
        ground_truth = np.load(model_dict.paths.labels_path / 'ground_truth.npy', allow_pickle=True).item()
        clust_dict.update({
                # model_name: {**clustering, **ground_truth}
                model_name: {**ground_truth}
        })

embeds = {}
for model_name, model_dict in loader_dict.items():
        embeds[model_name] = np.vstack(list(model_dict.embedding_dict().values()))

## compute clusterings

# kmeans
from sklearn.cluster import KMeans, SpectralClustering
from hdbscan import HDBSCAN

kmeans = KMeans(n_clusters=16) # because 15 species + noise for the within and diff file ...?
hdb = HDBSCAN(min_cluster_size=5, min_samples=8)
spec = SpectralClustering(n_clusters=16)

for model_name in loader_dict:
        clust_dict[model_name]['kmeans'] = kmeans.fit_predict(embeds[model_name])
        clust_dict[model_name]['hdb'] = hdb.fit_predict(embeds[model_name])
        # clust_dict[model_name]['spec'] = spec.fit_predict(embeds[model_name])




from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI

clust_results = {}

for model_name, clusterings in clust_dict.items():
        clust_results[model_name] = {}
        for name, clust in clust_dict[model_name].items():
                clust_results[model_name].update({
                                name: AMI(clust, clusterings['label:species'])
                })
        # for clust_name in ['kmeans', 'hdbscan']:
        #         clust_results[model_name].update({
        #                         clust_name: AMI(clusterings[clust_name], 
        #                                         clusterings['label:species'])
        #         })
        # for clust_name in ['kmeans_no_noise', 'hdbscan_no_noise']:
        #         clust_results[model_name].update({
        #                         clust_name: AMI(clusterings[clust_name], 
        #                                         clusterings['label:species'][clusterings['label:species']>=0])
        #         })
                
print(clust_results)


from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Convert string labels to numeric codes for coloring
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

# Add traces for each unique label to both subplots
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
    
#     # Add to second subplot (example: plot same data or different data)
#     fig.add_trace(go.Scatter(
#         x=umaps[][mask, 0],
#         y=umaps[][mask, 1],
#         mode='markers',
#         name=label,
#         marker=dict(size=8),
#         hovertemplate=f'<b>{label}</b><br>x: %{{x}}<br>y: %{{y}}<extra></extra>',
#         legendgroup=label,  # Same group as first plot
#         showlegend=False  # Only show legend once
#     ), row=1, col=2)

# fig.update_xaxes(title_text='UMAP 1', row=1, col=1)
# fig.update_xaxes(title_text='UMAP 1', row=1, col=2)
# fig.update_yaxes(title_text='UMAP 2', row=1, col=1)
# fig.update_yaxes(title_text='UMAP 2', row=1, col=2)

fig.update_layout(height=1200, width=1600, hovermode='closest')
fig.write_html('test.html')
fig.show()