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

from cluster_algos import MiniBatchkMeans_w_DBSCAN

from create_dataset import read_dataset

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
            except:
                umap_embed_obj = Embedder(model_name, loader=loader_dr, device='cuda', padding='wrap', dim_reduction_model='umap')
                umap_embed_obj.run_dimensionality_reduction_pipeline()
                loader_dr.write_metadata_file()
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



def fetch_clustering(embeds, clustering_dict, overwrite=False):
    if overwrite:
        cluster_booleans = df.copy()
        clust_df = df.copy()


        clust_results = {}
        for model_name, embed_dict in tqdm(embeds.items(), 'iterating through models', total=len(embeds), position=0):
            if overwrite or not (main_results_path / f'{model_name}_df_with_clusters.csv').exists():
                for snr_file, embed_array in tqdm(embed_dict.items(), "iterating through snr's", total=len(embed_dict), position=1):
                    print('\n ')
                    
                    snr_val = float(snr_file.replace(',', '.').split('snr=')[-1])
                    indices = df[df.snr.isin([snr_val, -1]).values & (df.model==model_name).values].index.values

                    for clust_name, clust_algo in clustering_dict.items():
                    
                        clust_df[f"{model_name}_{snr_val}_{clust_name}"] = np.nan
                        clust_df.loc[indices, f"{model_name}_{snr_val}_{clust_name}"] = clust_algo.fit_predict(embed_array)
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

        clust_df.to_csv(main_results_path / 'clusters.csv', index=False)
        cluster_booleans.to_csv(main_results_path / 'cluster_booleans.csv', index=False)
        
        with open(main_results_path / 'clust_results.json', 'w') as f:
            json.dump(clust_results, f)
    else:
        cluster_booleans = pd.read_csv(main_results_path / 'cluster_booleans.csv', index_col=False)
        clust_df = pd.read_csv(main_results_path / 'clusters.csv', index_col=False)
        
        with open(main_results_path / 'clust_results.json', 'r') as f:
            clust_results = json.load(f)
    return clust_df, cluster_booleans, clust_results
        
def fetch_visualization_df(clust_df, path, clustering_dict, overwrite=True):
    if overwrite or not (path / 'visualization_dataframe.csv').exists():
        df_vis = clust_df.copy()
        df_vis = df_vis.sort_values(['noise_env', 'snr'])
        cols2rename = [c for c in df_vis.columns if not c in ['audiofilename', 'start' ,'end', 'noise_start', 'noise_end', 'species_end', 'noise_filename', 'model', 'snr']]
        df_vis = df_vis.rename(columns={k: f'label:{k}' for k in cols2rename})
        
        for model in umaps:
            for snr_str in umaps[model]:
                snr_val = float(snr_str.split('=')[-1].replace(',', '.'))
                snr_model_df = pd.DataFrame()
                for h5_file in umaps[model][snr_str]['metadata']['audio_files']:
                    df_tmp, audio = read_dataset(path / snr_str.split('_')[0] / h5_file, return_audio=False)   
                    df_tmp['audiofilename'] = h5_file
                    input_length = (
                        # umaps[model][snr_str]['metadata']['segment_length (samples)']
                        # / umaps[model][snr_str]['metadata']['sample_rate (Hz)']
                        3 # this needs to corrspond to the original file, not the model
                    )
                    df_tmp['start'] = np.arange(len(df_tmp)) * input_length
                    df_tmp['end'] = df_tmp['start'] + input_length
                    snr_model_df = pd.concat([snr_model_df, df_tmp])
                    
                snr_model_df = snr_model_df.sort_values(['noise_env', 'snr'])
                for col in ['audiofilename', 'start', 'end']:
                    df_vis.loc[(df_vis.snr.isin([snr_val, -1])).values & (df_vis.model==model).values, col] = snr_model_df.loc[:, col].values

                key = list(clustering_dict.keys())[-1]
                clust_df_equiv = clust_df[clust_df.snr.isin([snr_val, -1]).values & (clust_df[f'{model}_{snr_val}_{key}'] >-3).values]

                for col in clust_df_equiv.columns:
                    if model in col and str(snr_val) in col:
                        df_vis[f'label:{col}'] = clust_df_equiv[col]
                        
                gt_df = df_vis.copy()
                drop_cols = [c for c in gt_df.columns if '.' in c and (not model in c or not str(snr_val) in c)]
                
                for col in drop_cols:
                    gt_df.pop(col)
                
                
                ground_truth_by_model(
                    model=model,
                    main_results_dir = Path(settings.main_results_dir) / path.stem,
                    audio_dir=path / snr_str,
                    label_df=gt_df,
                    overwrite=overwrite,
                    only_embed_annotations=True
                    # label_idx_dict=None,
                    # label_column='label:species',
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
    ax = fig.subplots((len(df.species.unique())-1)//2, 2)
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


###############################################################################################################################################

# main_path = Path('/mnt/swap/Work/Data/identifying_unknown_sounds_data')
# main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
path = main_path / Path('data_h5_files/6_ratio-n2t_10')
# path = main_path / Path('data_h5_files/6_ratio-n2t_10_cleaned')
# file_name = 'unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_4.h5'
# file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem}.h5'
file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem+"_snr=0"}.h5'#.split("_cleaned")[0]+

models = ['birdnet'] #, 'audioprotopnet', 'perch_v2'


embeds, umaps = get_embeddings(path, models)


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


main_results_path = main_path / Path('data') / 'clusterings' / path.stem
main_results_path.mkdir(exist_ok=True, parents=True)



## compute clusterings
n_centroids = embeds['birdnet']['snr=12'].shape[0]//100
max_clust = 80
clustering_dict = {
    # 'kmeans': KMeans(n_clusters=80), # because 15 species + noise for the within and diff file ...?
    # 'hdb': HDBSCAN(min_cluster_size=10, min_samples=None),
    # 'spec': SpectralClustering(n_clusters=16),
    # f'minibatchkmeans_w_dbscan_{max_clust}': MiniBatchkMeans_w_DBSCAN(
    #     max_cluster_size=max_clust,
    #     n_centroids=n_centroids
    #     ),
    f'kmeans_umap-init_w_dbscan_{max_clust}': MiniBatchkMeans_w_DBSCAN(
        max_cluster_size=max_clust,
        n_centroids=n_centroids, 
        initial_clustering='kmeans'
        ),
    f'kmeans_umap-init_half_w_dbscan_{max_clust}': MiniBatchkMeans_w_DBSCAN(
        max_cluster_size=max_clust,
        n_centroids=2, 
        initial_clustering='kmeans'
        )
}

clust_df, cluster_booleans, clust_results = fetch_clustering(embeds, clustering_dict, overwrite=False)


print(clust_results)

for model in models:
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

df_vis = fetch_visualization_df(clust_df, path, clustering_dict, overwrite=False)

    

remaining_settings = {**vars(settings)}

snr_str = 'snr=12'
snr_val = 12.0
vis_settings = {
    'models':list(embeds.keys()), 
    'audio_dir':path / snr_str, 
    'audio_suffixes' : ['.h5'],
    'main_results_dir':Path(settings.main_results_dir) / path.stem, 
    'default_label_keys':settings.default_label_keys,#{}, 
    'evaluation_task':config.evaluation_task, 
    'dim_reduction_model':config.dim_reduction_model, 
    'dim_reduc_parent_dir':settings.dim_reduc_parent_dir,
    'only_embed_annotations':True,
    'annotations_df' : df_vis,#[df_vis.snr.isin([snr_val, -1])],
    'constant_sr' : 32_000
}


for k in vis_settings.keys():
    if k in remaining_settings:
        remaining_settings.pop(k)
        
        
visualize_using_dashboard(
    **vis_settings,
    **remaining_settings
    )
