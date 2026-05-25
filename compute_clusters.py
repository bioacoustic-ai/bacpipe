import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from bacpipe.main import model_specific_embedding_creation
# from bacpipe import config, settings
import h5py
import json

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from matplotlib.colors import rgb2hex

from sklearn.cluster import KMeans, SpectralClustering
from hdbscan import HDBSCAN


from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import homogeneity_score as HS

from tqdm import tqdm

from bacpipe.embedding_evaluation.label_embeddings import DefaultLabels as Labels

from bacpipe import Embedder, Loader

from create_dataset import read_dataset

def get_embeddings(path, file_name):
        audio_dir = Path(path)
        df, audio = read_dataset(audio_dir / file_name)
        embeds, umaps = {}, {}

        for model_name in ['birdnet', 'audioprotopnet']:
                loader = Loader(audio_dir, model_name, use_folder_structure=True, audio_suffixes=['.h5'])
                try:
                        embeds[model_name] = loader.embeddings(return_type='array')
                except:
                        embed_obj = Embedder(model_name, loader=loader, device='cuda', padding='wrap', global_batch_size=12)
                        embeds[model_name] = embed_obj.embeddings_using_multithreading(audio)
                        embeds[model_name] = np.vstack(embeds[model_name])
                        length = (
                                embeds[model_name].shape[0] 
                                * embed_obj.model.segment_length 
                                / embed_obj.model.sr
                        )
                        file_length = {Path(file_name).stem: length}
                        loader._write_audio_file_to_metadata(
                                audio_dir / file_name, 
                                embed_obj.model, 
                                embeds[model_name], 
                                file_length
                        )
                        loader.save_embedding_file(
                                audio_dir / file_name, 
                                embeds[model_name]
                                )
                
                        loader.write_metadata_file()
                
                loader_dr = Loader(audio_dir, model_name, use_folder_structure=True, dim_reduction_model='umap', audio_suffixes=['.h5'])
                
                try:
                        file = list(loader_dr.embed_dir.rglob('*json'))[0]
                        with open(file, 'r') as f:
                                umaps[model_name] = json.load(f)
                except:
                        umap_embed_obj = Embedder(model_name, loader=loader_dr, device='cuda', padding='wrap', dim_reduction_model='umap')
                        umaps[model_name] = umap_embed_obj.get_reduced_dimensionality_embeddings(np.vstack(embeds[model_name]))
                        loader_dr.metadata_dict["files"]["embedding_files"].append(
                                str(loader_dr.files[0].relative_to(loader_dr.metadata_dict['embed_dir']))
                                )
                        loader_dr.metadata_dict["files"]["embedding_dimensions"].append(embeds[model_name].shape)
                        loader_dr.save_embedding_file(
                                audio_dir / file_name, 
                                umaps[model_name]
                                )
                        loader_dr.write_metadata_file()
        return embeds, umaps



path = Path('data/data_h5_files/3_ratio-n2t_1')
# file_name = 'unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_4.h5'
file_name = 'unknown_sounds_len_3_sr_32000_repetitions_3_ratio-n2t_1.h5'

embeds, umaps = get_embeddings(path, file_name)


df, audio = read_dataset(path / file_name)



main_results_path = Path('data') / 'clusterings' / path.stem
main_results_path.mkdir(exist_ok=True, parents=True)

if True:
        label_dict_bool = {}
        for eval_name in ['species_vs_species', 'species_vs_infile_noise', 'species_vs_other_noise', 'species_vs_all']:
                label_dict_bool[eval_name] = {}
                for species in df.species.unique():
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
                                        in enumerate(filenames) 
                                        if labels[idx] == species
                                        ])
                                within_files = [
                                        True 
                                        if fn in this_species_filenames 
                                        and labels[idx] == 'within_file' 
                                        else False 
                                        for idx, fn in enumerate(filenames)
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
                                                        species: AMI(clustering, ground_truth)
                                        })
                                clust_results[model_name][clust_name][eval_name].update({
                                        'avg': np.mean(list(clust_results[model_name][clust_name][eval_name].values()))
                                })        
        with open(main_results_path / 'label_dict_bool.npy', 'wb') as f:
                np.save(f, label_dict_bool, allow_pickle=True)
        with open(main_results_path / 'clust_results.npy', 'wb') as f:
                np.save(f, clust_results, allow_pickle=True)
        with open(main_results_path / 'clust_dict.npy', 'wb') as f:
                np.save(f, clust_dict, allow_pickle=True)
else:
        clust_results = np.load(main_results_path / 'clust_results.npy', allow_pickle=True).item()
        clust_dict = np.load(main_results_path / 'clust_dict.npy', allow_pickle=True).item()
        label_dict_bool = np.load(main_results_path / 'label_dict_bool.npy', allow_pickle=True).item()
        
                
print(clust_results)

model = 'birdnet'
clust_name = 'hdb'
species = 'Black-bellied Plover'
eval_name = 'species_vs_infile_noise'
# Convert string labels to numeric codes for coloring
print(clust_results[model][clust_name][eval_name][species])
print(clust_results[model]['kmeans'][eval_name][species])

clust_name_list = list(clust_results[model].keys())
eval_name_list = list(clust_results[model]['kmeans'].keys())
species_list = list(clust_results[model][clust_name][eval_name].keys())
species_list.remove('avg')

dropdown_vars = {
        'def_clust': clust_name,
        'def_eval': eval_name,
        'def_species': species,
        'clust_opts': clust_name_list,
        'species_opts': species_list,
        'eval_opts': eval_name_list,
}

from interactive_plot import InteractivePlot
file_dts = [Labels.get_dt_filename(f) for f in filenames]

padding_func = main_results_path.stem.split('_')[-1]
plot_obj = InteractivePlot(
        data_file,
        clust_results,
        label_dict_bool,
        clust_dict,
        umaps,
        src_path = '/media/siriussound/Extreme SSD/Recordings',
        sample_rate = 48_000,
        example_window_seconds = 3.,
        pad_func = padding_func
        )

# fig = plotly_mutual_information(model, clust_name, species, eval_name)
plot_obj.interactive_plot('birdnet', title=f'cluster_evaluation {padding_func}', port=8052, **dropdown_vars)
        
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
                                customdata=np.column_stack((datasets[mask], filenames[mask], starts[mask])),
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
        
        