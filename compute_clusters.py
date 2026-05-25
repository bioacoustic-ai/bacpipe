import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from bacpipe.main import model_specific_embedding_creation
# from bacpipe import config, settings
import h5py
import json

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
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
    embeds, umaps = {}, {}
    audio_dir = Path(path)
    df, audio = read_dataset(audio_dir / file_name)

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



path = Path('data/data_h5_files/1_ratio-n2t_2')
# file_name = 'unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_4.h5'
file_name = 'unknown_sounds_len_3_sr_32000_repetitions_1_ratio-n2t_2.h5'

embeds, umaps = get_embeddings(path, file_name)


df, audio = read_dataset(path / file_name)



main_results_path = Path('data') / 'clusterings' / path.stem
main_results_path.mkdir(exist_ok=True, parents=True)


if False:
    cluster_booleans = df.copy()
    clust_df = df.copy()

    ## compute clusterings

    # kmeans

    kmeans = KMeans(n_clusters=4) # because 15 species + noise for the within and diff file ...?
    hdb = HDBSCAN(min_cluster_size=10, min_samples=None)
    spec = SpectralClustering(n_clusters=16)

    clust_results = {}
    for model_name, embed_array in tqdm(embeds.items()):
        for snr in df.snr.unique():
            if snr < 0:
                continue
            
                
            indices = df[df.snr.isin([snr, -1])].index.values

            for clust_name in ['kmeans', 'hdb']:
            
                clust_df[f"{model_name}_{snr}_{clust_name}"] = np.nan
                clust_df.loc[indices, f"{model_name}_{snr}_{clust_name}"] = vars().get(clust_name).fit_predict(embed_array[indices])
        # clust_dict[model_name]['spec'] = spec.fit_predict(embeds[model_name])




        clust_results[model_name] = {}
        
        for snr in df.snr.unique():
            if snr < 0:
                continue
            clust_results[model_name][snr] = {}
            
            for clust_name in ['kmeans', 'hdb']:
                clust_results[model_name][snr][clust_name] = {}
                
                evaluation_cases = ['species_vs_species', 'species_vs_infile_noise', 'species_vs_other_noise', 'species_vs_all']
            
                for eval_name in evaluation_cases:
                    clust_results[model_name][snr][clust_name][eval_name] = {}
                    
            
                    for noise_env in df.noise_env.unique():
                        clust_results[model_name][snr][clust_name][eval_name][noise_env] = {}
                        
                        for species in df.species.unique():
                            if species == '':
                                continue
                        
                        
                            if eval_name == 'species_vs_all':
                                df_tmp = df[df.noise_env==noise_env][df.snr.isin([snr, -1])]
                            elif eval_name in 'species_vs_species':
                                df_tmp = df[df.noise_env==noise_env][df.snr==snr]
                            elif eval_name == 'species_vs_infile_noise':
                                df_tmp = df[df.noise_env==noise_env][df.snr.isin([snr, -1])][df.species.isin([species, ''])]
                            elif eval_name == 'species_vs_other_noise':
                                df_tmp = df[df.noise_env!=noise_env][df.snr.isin([snr, -1])][df.species!=species]
                            
                            cluster_booleans.loc[df_tmp.index, f"{snr}_{eval_name}_{species}_{noise_env}"] = True    
                            
                            ground_truth = [1 if l == species else 0 for l in df_tmp.species]
                            
                            clusters = clust_df[f"{model_name}_{snr}_{clust_name}"][df_tmp.index]
                            
                            clust_results[model_name][snr][clust_name][eval_name][noise_env].update({
                                            species: HS(clusters, ground_truth)
                            })
                            clust_results[model_name][snr][clust_name][eval_name][noise_env].update({
                                'avg': np.mean(list(
                                    clust_results[model_name][snr][clust_name][eval_name][noise_env].values()
                                    ))
                            })        
                            
    cluster_booleans.to_csv(main_results_path / 'cluster_booleans.csv')
    clust_df.to_csv(main_results_path / 'clusters.csv')
    
    with open(main_results_path / 'clust_results.json', 'w') as f:
        json.dump(clust_results, f)
else:
    cluster_booleans = pd.read_csv(main_results_path / 'cluster_booleans.csv', index_col=False)
    clust_df = pd.read_csv(main_results_path / 'clusters.csv', index_col=False)
    
    with open(main_results_path / 'clust_results.json', 'r') as f:
        clust_results = json.load(f)
        
                
print(clust_results)

model = 'audioprotopnet'
clust_name = 'hdb'
# species = 'Black-bellied Plover'
eval_name = 'species_vs_infile_noise'
## plot species by snr 
for model in ['birdnet', 'audioprotopnet']:
    plot_data= {}
    for species in df.species.unique():
        if species == '':
            continue
        plot_data[species] = {}
        for noise_env in df.noise_env.unique():
            plot_data[species][noise_env] = [clust_results[model][str(snr)][clust_name][eval_name][noise_env][species] for snr in df.snr.unique() if snr >= 0]


    fig = plt.figure(figsize=[10, 8])
    ax = fig.subplots(2, 1)
    idx = 0
    for species in df.species.unique():
        if species == '':
            continue
        for noise_env in df.noise_env.unique():
            ax[idx].plot([s for s in df.snr.unique() if s >= 0], plot_data[species][noise_env], label=noise_env)
        ax[idx].set_title(species)
        ax[idx].set_xticks([s for s in df.snr.unique() if s >= 0], [str(s) for s in df.snr.unique() if s >= 0])
        idx += 1
    ax[0].set_ylabel('Homogeneity Score')
    ax[-1].set_ylabel('Homogeneity Score')
    ax[-1].legend()
    ax[-1].set_xlabel('SNR')
    fig.suptitle(f'{clust_name} {eval_name} {model}')
    fig.savefig(main_results_path / f'{clust_name}_{eval_name}_{model}.png')

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
        
        