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

from bacpipe import Embedder, Loader, get_audio_files, visualize_using_dashboard, settings, config, ground_truth_by_model

from create_dataset import read_dataset

def get_embeddings(path):
    embeds_snr, umaps = {}, {}
    audio_dir = Path(path)

    for model_name in ['birdnet', 'audioprotopnet', 'perch_v2']:#
        embeds_snr[model_name] = {}
        for snr_dir in audio_dir.iterdir():
            snr_string = snr_dir.stem if snr_dir.is_dir() else False
            if not snr_string:
                continue

            loader = Loader(snr_dir, model_name, use_folder_structure=True, audio_suffixes=['.h5'], main_results_dir=f'bacpipe_results/{audio_dir.stem}')
            try:
                if loader.continue_incomplete_run:
                    raise FileNotFoundError('Not all files have been processed yet.')
                embeds_snr[model_name][snr_string] = loader.embeddings()
            except:
                embed_obj = Embedder(model_name, loader=loader, device='cuda', padding='wrap', global_batch_size=24)
                for file in loader.files:
                    df, audio = read_dataset(file)
                    embeds_snr[model_name][snr_string] = embed_obj.embeddings_using_multithreading(audio)
                    embeds_snr[model_name][snr_string] = np.vstack(embeds_snr[model_name][snr_string])
                    length = (
                        embeds_snr[model_name][snr_string].shape[0] 
                        * embed_obj.model.segment_length 
                        / embed_obj.model.sr
                    )
                    file_length = {file.stem: length}
                    loader._write_audio_file_to_metadata(
                        file, 
                        embed_obj.model, 
                        embeds_snr[model_name][snr_string], 
                        file_length
                    )
                    loader.save_embedding_file(
                        file, 
                        embeds_snr[model_name][snr_string]
                    )

                loader.write_metadata_file()
            if isinstance(embeds_snr[model_name][snr_string], dict):
                embeds_snr[model_name][snr_string] = np.vstack(list(embeds_snr[model_name][snr_string].values()))
        

            loader_dr = Loader(snr_dir, model_name, use_folder_structure=True, dim_reduction_model='umap', audio_suffixes=['.h5'], main_results_dir=f'bacpipe_results/{audio_dir.stem}')

            umaps[model_name] = {}
            try:
                files = list(loader_dr.embed_dir.rglob('*json'))
                if len(files) == 0:
                    raise FileNotFoundError("umaps haven't been computed yet")
                for file in files:
                    with open(file, 'r') as f:
                        umaps[model_name][file.stem] = json.load(f)
            except:
                umap_embed_obj = Embedder(model_name, loader=loader_dr, device='cuda', padding='wrap', dim_reduction_model='umap')
                umap_embed_obj.run_dimensionality_reduction_pipeline()
                loader_dr.write_metadata_file()
                
                
                
                
        
        # reorganize embeds dict
        # embeds_snr[model_name] = {}
        # audio_files = get_audio_files(path, audio_suffixes=['.h5'])
        # noise_keys = [k for k in embeds[model_name] if not 'snr' in k]
        # noise_keys.sort()
        # snr_keys = [l for l in embeds[model_name] if 'snr' in l]
        # snr_audio_keys = [l for l in audio_files if 'snr' in l.stem]
        # for snr_file in snr_keys:
        #     snr_string = [s.stem for s in snr_audio_keys if s.stem in snr_file][0]
                
                # umaps[model_name][snr_string] = umap_embed_obj.get_reduced_dimensionality_embeddings(embeds_snr[model_name][snr_string])
                
                # loader_dr.metadata_dict["files"]["embedding_files"].append(
                #     [l.stem+l.suffix for l in loader_dr.files]
                # )
                
                # # loader_dr.metadata_dict["files"]["embedding_dimensions"].append(embeds_array.shape)
                
                # # if idx == 0:
                # #     for k in loader.metadata_dict['files'].keys():
                # #         loader_dr.metadata_dict["files"][k][idx:idx] = (
                # #             loader.metadata_dict['files'][k][:len(noise_keys)]
                # #         )
                        
                # #     for k in loader.metadata_dict['files'].keys():
                # #         loader_dr.metadata_dict["files"][k] = (
                # #             loader_dr.metadata_dict["files"][k][idx:len(noise_keys)+1+idx]
                # #         )
                # # else:
                # #     for k in loader.metadata_dict['files'].keys():
                # #         loader_dr.metadata_dict["files"][k][-1] = loader.metadata_dict['files'][k][len(noise_keys)+1:][0]
                    
                # loader_dr.metadata_dict['nr_embeds_total'] = sum(loader_dr.metadata_dict["files"]['nr_embeds_per_file'])
                    
                    
                # loader_dr._save_embeddings_dict_with_timestamps(
                #     loader_dr.embed_dir / f'{snr}.json', 
                #     umaps[model_name][snr],
                #     loader.metadata_dict['segment_length (samples)'] / loader.metadata_dict['sample_rate (Hz)']
                # )
                




                # # loader_dr.files = snr_keys
                # keep_idxs = [idx for idx, f in enumerate(loader.metadata_dict['files']['audio_files']) if 'snr' in f]
                # for k in loader.metadata_dict['files']:
                #     loader_dr.metadata_dict['files'][k] = [loader.metadata_dict['files'][k][idx] for idx in keep_idxs]
                # for idx, (snr, embeds_array) in tqdm(enumerate(embeds_snr[model_name].items())):
                
                #     umaps[model_name][snr] = umap_embed_obj.get_reduced_dimensionality_embeddings(embeds_array)
                    
                #     loader_dr.metadata_dict["files"]["embedding_files"].append(
                #         loader_dr.files[idx]
                #     )
                    
                #     loader_dr.metadata_dict["files"]["embedding_dimensions"].append(embeds_array.shape)
                    
                #     if idx == 0:
                #         for k in loader.metadata_dict['files'].keys():
                #             loader_dr.metadata_dict["files"][k][idx:idx] = (
                #                 loader.metadata_dict['files'][k][:len(noise_keys)]
                #             )
                            
                #         for k in loader.metadata_dict['files'].keys():
                #             loader_dr.metadata_dict["files"][k] = (
                #                 loader_dr.metadata_dict["files"][k][idx:len(noise_keys)+1+idx]
                #             )
                #     else:
                #         for k in loader.metadata_dict['files'].keys():
                #             loader_dr.metadata_dict["files"][k][-1] = loader.metadata_dict['files'][k][len(noise_keys)+1:][0]
                        
                #     loader_dr.metadata_dict['nr_embeds_total'] = sum(loader_dr.metadata_dict["files"]['nr_embeds_per_file'])
                        
                        
                #     loader_dr._save_embeddings_dict_with_timestamps(
                #         loader_dr.embed_dir / f'{snr}.json', 
                #         umaps[model_name][snr],
                #         loader.metadata_dict['segment_length (samples)'] / loader.metadata_dict['sample_rate (Hz)']
                #     )
                    
                    
                # loader_dr.metadata_dict['files']['nr_embeds_per_file'] = [loader_dr.metadata_dict['nr_embeds_total']] * len(embeds_snr[model_name])
                # loader_dr.write_metadata_file()
    return embeds_snr, umaps

def collect_noise_dfs(path):
    h5_files = get_audio_files(path, audio_suffixes=['.h5'])
    h5_noise_files = [k for k in h5_files if not 'snr' in k.stem]
    h5_noise_files.sort()
    noise_df = pd.DataFrame()
    
    for noise_file in h5_noise_files:
        df_noise_tmp, audio = read_dataset(noise_file, return_audio=False)
        noise_df = pd.concat([noise_df, df_noise_tmp])
    return noise_df

# main_path = Path('/mnt/swap/Work/Data/identifying_unknown_sounds_data')
# main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
path = main_path / Path('data_h5_files/6_ratio-n2t_10')
# file_name = 'unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_4.h5'
file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem}.h5'

embeds, umaps = get_embeddings(path)


df = pd.DataFrame()
for model in embeds:

    for idx, file in enumerate(embeds[model]):
        noise_df = collect_noise_dfs(path / file)
        noise_df['model'] = model
        snr_file = list(Path(path / file).glob('*snr*.h5'))[0]
        df_tmp, audio = read_dataset(snr_file, return_audio=False)    
        df_tmp['model'] = model
        df = pd.concat([df, df_tmp])
    
    df = pd.concat([df, noise_df])
    
        
df.index = range(len(df))


model = 'perch_v2'
snr_str = 'snr=12_perch_v2'
df_vis = pd.DataFrame()
for h5_file in umaps[model][snr_str]['metadata']['audio_files']:
    df_tmp, audio = read_dataset(path / snr_str.split('_')[0] / h5_file, return_audio=False)   
    df_tmp['audiofilename'] = h5_file
    input_length = (
        umaps[model][snr_str]['metadata']['segment_length (samples)']
        / umaps[model][snr_str]['metadata']['sample_rate (Hz)']
    )
    df_tmp['start'] = np.arange(len(df_tmp)) * input_length
    df_tmp['end'] = df_tmp['start'] + input_length
    df_vis = pd.concat([df_vis, df_tmp])
    
cols2rename = [c for c in df_vis.columns if not c in ['audiofilename', 'start' ,'end', 'noise_start', 'noise_end', 'species_end', 'noise_filename']]
df_vis = df_vis.rename(columns={k: f'label:{k}' for k in cols2rename})

snr_str = 'snr=12'
ground_truth_by_model(
    model=model,
    main_results_dir = Path(settings.main_results_dir) / path.stem,
    audio_dir=path / snr_str,
    label_df=df_vis,
    overwrite=False
    # label_idx_dict=None,
    # label_column='label:species',
)

# with h5py.File('/media/siriussound/Extreme SSD/identifying_unknown_sounds/data_h5_files/6_ratio-n2t_10/snr=0/unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_10_AnuranSet_INCT41.h5', 'r') as data:
#     a = data['audio'][:]
    

remaining_settings = {**vars(settings)}

vis_settings = {
    'models':[model],#list(embeds.keys()), 
    'audio_dir':path / snr_str, 
    'audio_suffixes' : ['.h5'],
    'main_results_dir':Path(settings.main_results_dir) / path.stem, 
    'default_label_keys':settings.default_label_keys,#{}, 
    'evaluation_task':config.evaluation_task, 
    'dim_reduction_model':config.dim_reduction_model, 
    'dim_reduc_parent_dir':settings.dim_reduc_parent_dir,
    'only_embed_annotations':True,
    'annotations_df' : df_vis,
}


for k in vis_settings.keys():
    if k in remaining_settings:
        remaining_settings.pop(k)
        
        
visualize_using_dashboard(
    **vis_settings,
    **remaining_settings
    )



main_results_path = main_path / Path('data') / 'clusterings' / path.stem
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
    for model_name, embed_dict in tqdm(embeds.items(), 'iterating through models', total=len(embeds), position=0):
        if not (main_results_path / f'{model_name}_df_with_clusters.csv').exists():
            for snr_file, embed_array in tqdm(embed_dict.items(), "iterating through snr's", total=len(embed_dict), position=1):
                print('\n ')
                
                snr_val = float(snr_file.split('snr=')[-1])
                indices = df[df.snr.isin([snr_val, -1]).values & (df.model==model_name).values].index.values

                for clust_name in ['kmeans', 'hdb']:
                
                    clust_df[f"{model_name}_{snr_val}_{clust_name}"] = np.nan
                    clust_df.loc[indices, f"{model_name}_{snr_val}_{clust_name}"] = vars().get(clust_name).fit_predict(embed_array)
            clust_df.to_csv(main_results_path / f'{model_name}_df_with_clusters.csv', index=False)
        else:
            clust_df = pd.read_csv(main_results_path / f'{model_name}_df_with_clusters.csv', index_col=False)
        # clust_dict[model_name]['spec'] = spec.fit_predict(embeds[model_name])

        clust_results[model_name] = {}

        # 1. Use a dictionary to collect boolean data instead of a fragmented DataFrame
        boolean_dict = {}

        for snr in tqdm(df.snr.unique(), 'evaluating clusters by snr', total=len(df.snr.unique())):
            if snr < 0:
                continue
            else:
                snr = str(snr)
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
                            
                            # Pre-calculate base masks to keep code readable
                            m_model = df.model == model_name
                            m_env = df.noise_env == noise_env
                            m_snr_match = df.snr == float(snr)
                            m_snr_all = df.snr.isin([float(snr), -1])
                        
                            # 2. Fix the Chained Indexing Warning by combining masks with &
                            if eval_name == 'species_vs_all':
                                df_tmp = df.loc[m_env & m_snr_all & m_model]
                                
                            elif eval_name == 'species_vs_species':
                                df_tmp = df.loc[m_env & m_snr_match & m_model]
                                
                            elif eval_name == 'species_vs_infile_noise':
                                m_spec = df.species.isin([species, ''])
                                df_tmp = df.loc[m_env & m_snr_all & m_spec & m_model]
                                
                            elif eval_name == 'species_vs_other_noise':
                                m_not_env = df.noise_env != noise_env
                                m_not_spec = df.species != species
                                df_tmp = df.loc[m_not_env & m_snr_all & m_not_spec & m_model]
                            
                            # 3. Fix Performance Warning: Save to dict instead of .loc
                            col_name = f"{snr}_{eval_name}_{species}_{noise_env}"
                            # Create a series initialized with False, set true values where index matches
                            col_series = pd.Series(False, index=df.index)
                            col_series.loc[df_tmp.index] = True
                            boolean_dict[col_name] = col_series
                            
                            # Ground truth and evaluation processing
                            ground_truth = [1 if l == species else 0 for l in df_tmp.species]
                            clusters = clust_df[f"{model_name}_{snr}_{clust_name}"][df_tmp.index]
                            
                            clust_results[model_name][snr][clust_name][eval_name][noise_env].update({
                                species: HS(clusters, ground_truth)
                            })
                            
                            # Calculate average safely
                            vals = list(clust_results[model_name][snr][clust_name][eval_name][noise_env].values())
                            clust_results[model_name][snr][clust_name][eval_name][noise_env]['avg'] = np.mean(vals)

        # 4. Once the loops are finished, build/merge the DataFrames instantly
        new_booleans = pd.DataFrame(boolean_dict)
        # If cluster_booleans already exists, combine them; otherwise, assign it directly
        cluster_booleans = pd.concat([cluster_booleans, new_booleans], axis=1)

    clust_df.to_csv(main_results_path / 'clusters.csv', index=False)
    cluster_booleans.to_csv(main_results_path / 'cluster_booleans.csv', index=False)
    
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
snr = '9.0'
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
    ax = fig.subplots(4, 1)
    idx = 0
    for species in df.species.unique():
        if species == '':
            continue
        for noise_env in df.noise_env.unique():
            ax[idx].plot([s for s in df.snr.unique() if s >= 0], plot_data[species][noise_env], label=noise_env)
        ax[idx].set_title(species)
        
        ax[idx].set_ylim([0, 1])
        
        ax[idx].set_xticks([s for s in df.snr.unique() if s >= 0], [str(s) for s in df.snr.unique() if s >= 0])
        idx += 1
    ax[0].set_ylabel('Homogeneity Score')
    ax[-1].set_ylabel('Homogeneity Score')
    ax[-1].legend()
    ax[-1].set_xlabel('SNR')
    fig.suptitle(f'{clust_name} {eval_name} {model}')
    fig.tight_layout()
    fig.savefig(main_results_path / f'{clust_name}_{eval_name}_{model}.png')

# Convert string labels to numeric codes for coloring
print(clust_results[model][snr][clust_name][eval_name]['BIRB_NES'][species])
print(clust_results[model][snr]['kmeans'][eval_name]['BIRB_NES'][species])

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
        
        