# clustering_utils.py

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


import pickle


from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import homogeneity_score as HS

from tqdm import tqdm

from bacpipe.embedding_evaluation.label_embeddings import DefaultLabels as Labels

from bacpipe import Embedder, Loader, get_audio_files, visualize_using_dashboard, settings, config, ground_truth_by_model

from cluster_algos import Clustering_Approach

from create_dataset import read_dataset


main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
path = main_path / Path('data_h5_files/10_ratio-n2t_50')


main_results_path = main_path / Path('data') / 'clusterings' / path.stem
main_results_path.mkdir(exist_ok=True, parents=True)

def get_embeddings(path, models):
    embeds_snr, umaps = {}, {}
    audio_dir = Path(path)

    for model_name in tqdm(models):#
        embeds_snr[model_name] = {}
        umaps[model_name] = {}
        for snr_dir in tqdm(audio_dir.iterdir()):
            snr_string = snr_dir.stem if snr_dir.is_dir() else False
            if not snr_string:
                continue
            # if not '0' in snr_string:
            #     continue
                # no need to work on other snr's for now

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

            try:
                files = list(loader_dr.embed_dir.rglob('*json'))
                if len(files) == 0:
                    raise FileNotFoundError("umaps haven't been computed yet")
                for file in files:
                    with open(file, 'r') as f:
                        umaps[model_name][file.stem.split('_')[0]] = json.load(f)
                # umaps[model_name][file.stem.split('_')[0]]['umap_model'] = load_umap_model(loader_dr.embed_dir / 'umap_model.pkl')
            except:
                umap_embed_obj = Embedder(model_name, loader=loader_dr, device='cuda', padding='wrap', dim_reduction_model='umap')
                umap_embed_obj.run_dimensionality_reduction_pipeline()
                loader_dr.write_metadata_file()
                save_umap_model(loader_dr.embed_dir / 'umap_model.pkl', umap_embed_obj)
            # break
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

def save_umap_model(path, umap_embed_obj):
    pickle.dump(umap_embed_obj.model.model, open(path, 'wb'))

def load_umap_model(path):
    # time passes
    return pickle.load((open(path, 'rb')))


def fetch_clustering(embeds, df, clustering_dict, overwrite=False):
    if overwrite:
        clust_df = df.copy()
        centroid_dict = dict()

        for model_name, embed_dict in tqdm(embeds.items(), 'iterating through models', total=len(embeds), position=0):
            if overwrite or not (main_results_path / f'{model_name}_df_with_clusters.csv').exists():
                for snr_file, embed_array in tqdm(embed_dict.items(), "iterating through snr's", total=len(embed_dict), position=1):
                    print('\n ')
                    
                    snr_val = float(snr_file.replace(',', '.').split('snr=')[-1])
                    indices = df[df.snr.isin([snr_val, -1]).values & (df.model==model_name).values].index.values

                    for clust_name, clust_algo in clustering_dict.items():
                    
                        clust_df[f"{model_name}_{snr_val}_{clust_name}"] = np.nan
                        final_cluster_labels, centroids, weights = clust_algo.fit_predict(embed_array)
                        centroid_dict[f"{model_name}_{snr_val}_{clust_name}"] = (centroids, weights)
                        clust_df.loc[indices, f"{model_name}_{snr_val}_{clust_name}"] = final_cluster_labels
                clust_df.to_csv(main_results_path / f'{model_name}_df_with_clusters.csv', index=False)
            else:
                clust_df = pd.read_csv(main_results_path / f'{model_name}_df_with_clusters.csv', index_col=False)
        
        clust_df.to_csv(main_results_path / f'clusters.csv', index=False)
        np.save(main_results_path / 'centroids.npy', centroid_dict, allow_pickle=True)
    else:
        centroid_dict = np.load(main_results_path / 'centroids.npy', allow_pickle=True).item()
        clust_df = pd.read_csv(main_results_path / f'clusters.csv', index_col=False)
        
    return clust_df, centroid_dict

def evaluate_clustering(df, clust_df, embeds, clustering_dict, overwrite=False):
    if overwrite:
        clust_results = {}
        cluster_booleans = df.copy()
        for model_name, embed_dict in tqdm(embeds.items(), 'iterating through models', total=len(embeds), position=0):
            clust_results[model_name] = {}

            # 1. Use a dictionary to collect boolean data instead of a fragmented DataFrame
            boolean_dict = {}

            for snr in tqdm(df.snr.unique(), 'evaluating clusters by snr', total=len(df.snr.unique())):
                if snr < 0:
                    continue
                else:
                    snr = str(snr)
                clust_results[model_name][snr] = {}
                
                for clust_name, clust_algo in clustering_dict.items():
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

        cluster_booleans.to_csv(main_results_path / 'cluster_booleans.csv', index=False)
        
        with open(main_results_path / 'clust_results.json', 'w') as f:
            json.dump(clust_results, f)
            
            
        if len(df.snr.unique()) > 3:
            for model in embeds.keys():
                for clust_name in clustering_dict.keys():
                    for eval_name in [
                        'species_vs_infile_noise',
                        'species_vs_all',
                        'species_vs_species',
                        'species_vs_other_noise'
                        ]:
                        save_path = main_results_path / f'{clust_name}_{model}'
                        save_path.mkdir(exist_ok=True)
                        plot_clusterings(clust_results, df, model, clust_name, eval_name, save_path)
    else:
        cluster_booleans = pd.read_csv(main_results_path / 'cluster_booleans.csv', index_col=False)
        
        with open(main_results_path / 'clust_results.json', 'r') as f:
            clust_results = json.load(f)
    return cluster_booleans, clust_results
        
def fetch_visualization_df(clust_df, path, clustering_dict, umaps, overwrite=True, overwrite_gt=False):
    if overwrite or not (path / 'visualization_dataframe.csv').exists():
        df_vis = clust_df.copy()
        
        cols2rename = [c for c in df_vis.columns if not c in ['audiofilename', 'start' ,'end', 'noise_start', 'noise_end', 'species_end', 'noise_filename', 'model', 'snr']]
        df_vis = df_vis.rename(columns={k: f'label:{k}' for k in cols2rename})
        
        for model in umaps:
            for snr_str in umaps[model]:
                snr_val = float(snr_str.split('=')[-1].replace(',', '.'))
                snr_model_df = load_df_same_order_as_embeddings(path, model, snr_val)
                
                for col in ['audiofilename', 'start', 'end']:
                    if all(
                        df_vis.loc[(df_vis.snr.isin([snr_val, -1])).values & (df_vis.model==model).values]['label:species'].fillna('')
                        ==snr_model_df.loc[:, 'species'].values
                        ):
                        df_vis.loc[(df_vis.snr.isin([snr_val, -1])).values & (df_vis.model==model).values, col] = snr_model_df.loc[:, col].values
                    else:
                        raise AssertionError('species values by row do not match')
                    
                df_vis.loc[(df_vis.snr.isin([snr_val, -1])).values & (df_vis.model==model).values, 'x'] = umaps[model][snr_str]['x']
                df_vis.loc[(df_vis.snr.isin([snr_val, -1])).values & (df_vis.model==model).values, 'y'] = umaps[model][snr_str]['y']

                key = list(clustering_dict.keys())[0]
                clust_df_equiv = clust_df[clust_df.snr.isin([snr_val, -1]).values & (clust_df[f'{model}_{snr_val}_{key}'] >-3).values]

                for col in clust_df_equiv.columns:
                    if model in col and str(snr_val) in col:
                        df_vis[f'label:{col}'] = clust_df_equiv[col]
                

                df_vis = df_vis.rename(columns={
                    'noise_filename': 'label:noise_filename',
                    'noise_start':'label:noise_start'
                })
                    
                        
                gt_df = df_vis.copy()
                drop_cols = [c for c in gt_df.columns if '.' in c and (not model in c or not str(snr_val) in c)]
                
                for col in drop_cols:
                    gt_df.pop(col)
                
                
                ground_truth_by_model(
                    model=model,
                    main_results_dir = Path(settings.main_results_dir) / path.stem,
                    audio_dir=path / snr_str,
                    label_df=gt_df,
                    overwrite=overwrite_gt,
                    only_embed_annotations=True
                )
        df_vis.to_csv(path / 'visualization_dataframe.csv', index=False)
    else:
        df_vis = pd.read_csv(path / 'visualization_dataframe.csv', index_col=False)
        
    return df_vis


def plot_clusterings(clust_results, df, model, clust_name, eval_name, save_path):
    plot_data= {}
    for species in df.species.unique():
        if species == '':
            continue
        plot_data[species] = {}
        for noise_env in df.noise_env.unique():
            plot_data[species][noise_env] = [clust_results[model][str(snr)][clust_name][eval_name][noise_env][species] for snr in np.sort(df.snr.unique()) if snr >= 0]


    fig = plt.figure(figsize=[14, 8])
    ax = fig.subplots(3, 4)#(len(df.species.unique())-1)//2, 4)
    idx = 0
    for species in df.species.unique():
        if species == '':
            continue
        for noise_env in df.noise_env.unique():
            ax[idx%3, idx//3].plot([s for s in np.sort(df.snr.unique()) if s >= 0], plot_data[species][noise_env], label=noise_env)
        ax[idx%3, idx//3].set_title(species)
        
        if (
            not eval_name == 'species_vs_infile_noise' 
            and not np.max(list(plot_data[species].values())) > 0.5
            ):
            
            ax[idx%3, idx//3].set_ylim([0, 0.5])
        else:
            ax[idx%3, idx//3].set_ylim([0, 1])
        
        ax[idx%3, idx//3].set_xticks([s for s in df.snr.unique() if s >= 0], [str(s) for s in df.snr.unique() if s >= 0])
        idx += 1
    ax[0, 0].set_ylabel('Homogeneity Score')
    ax[-1, 0].set_ylabel('Homogeneity Score')
    ax[-1, 0].legend()
    ax[-1, 0].set_xlabel('SNR')
    fig.suptitle(f'{clust_name} {eval_name} {model}')
    fig.tight_layout()
    fig.savefig(save_path / f'{clust_name}_{eval_name}_{model}.png')
    plt.close(fig)


def listen_to_index(idx, file_path = None):
    import sounddevice as sd
    import h5py
    from pathlib import Path
    import matplotlib.pyplot as plt
    import librosa as lb
    import numpy as np
    
    main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
    snr = 3
    if file_path is None:
        
        path = main_path / Path('data_h5_files/6_ratio-n2t_10_cleaned')
        
        file_name = f'snr={snr}/unknown_sounds_len_3_sr_32000_repetitions_{path.stem.split("_cleaned")[0]}_snr={snr}.h5'
        file_path = Path(path / file_name)
        
        file_path.exists()
    
    
    plt.figure(figsize=[10, 8])
    
    with h5py.File(file_path, 'r') as data:
        audio = data['audio'][idx]
    
    SR1 = 32_000
    SR = 48_000
    sd.play(audio, samplerate=SR1)
    S = lb.feature.melspectrogram(y=np.array(audio), sr=SR, n_mels=128,
                                    fmax=SR // 2)
    S_dB = lb.power_to_db(S, ref=np.max)
    img = lb.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=SR,
                            fmax=SR // 2)
    plt.colorbar(img, format='%+2.0f dB')

    path_snr = main_path / f'data/figures/snr_{snr}'

    path_snr.mkdir(exist_ok=True, parents=True)
    plt.savefig(path_snr / f'{idx}.png')
    plt.close()



def load_df_same_order_as_embeddings(audio_dir, model, snr):
    for snr_dir in tqdm(audio_dir.iterdir()):
        snr_string = snr_dir.stem if snr_dir.is_dir() else False
        try:
            if int(snr) == float(snr): snr = int(snr) 
        except ValueError:
            pass
        if not snr_string:
            continue
        elif not (snr_string in str(snr_dir) and str(snr).replace('.', ',') in snr_string):
            if not (snr_string in str(snr_dir) and str(snr).split('.')[-1] in snr_string):
                continue
    
        loader = Loader(snr_string, model, use_folder_structure=True, audio_suffixes=['.h5'], main_results_dir=f'bacpipe_results/{Path(audio_dir).stem}')
        
        
        snr_model_df = pd.DataFrame()
        for h5_file in loader.metadata_dict['files']['audio_files']:
            df_tmp, audio = read_dataset(path / snr_string.split('_')[0] / h5_file, return_audio=False)   
            df_tmp['audiofilename'] = h5_file
            input_length = (
                # umaps[model][snr_str]['metadata']['segment_length (samples)']
                # / umaps[model][snr_str]['metadata']['sample_rate (Hz)']
                
                3 # this needs to corrspond to the original .h5 file, not the model
            )
            df_tmp['start'] = np.arange(len(df_tmp)) * input_length
            df_tmp['end'] = df_tmp['start'] + input_length
            snr_model_df = pd.concat([snr_model_df, df_tmp])
    
    
    snr_model_df.index = range(len(snr_model_df))
    return snr_model_df