from pathlib import Path
from clustering_utils import *

# main_path = Path('/mnt/swap/Work/Data/identifying_unknown_sounds_data')
# main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
# path = main_path / Path('data_h5_files/6_ratio-n2t_10_cleaned')
# file_name = 'unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_4.h5'
# file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem}.h5'
# file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem+"_snr=0"}.h5'#.split("_cleaned")[0]+

models = [
    'birdnet_v3', 
    # 'birdnet', 
    # 'perch_v2', 
    # 'insect459', 
    # 'aves_especies', 
    'naturebeats', 
    'audioprotopnet', 
    'avesecho_passt'
    ]

from bacpipe.core.workflows import ensure_models_exist
ensure_models_exist(model_names = models)

embeds, umaps = get_embeddings(path, models)

SNR = 0


df = pd.DataFrame()
for model in models:
    for idx, snr in enumerate(embeds[model].keys()):
        df_temp = load_df_same_order_as_embeddings(path, model, snr.split('=')[-1].replace(',','.'))
        df_temp['model'] = [model] * len(df_temp)
        if idx == 0:
            df = pd.concat([df, df_temp])
        else:
            df_temp = df_temp[df_temp.snr!=-1]
            df = pd.concat([df, df_temp])
df.index = range(len(df))

## compute clusterings
n_centroids = embeds[model][f'snr={SNR}'].shape[0]//100
max_clust = 50
clustering_dict = {
    # 'kmeans': KMeans(n_clusters=n_centroids), # because 15 species + noise for the within and diff file ...?
    # # 'hdb': HDBSCAN(min_cluster_size=10, min_samples=None),
    # # 'spec': SpectralClustering(n_clusters=16),
    
    f'kmeans_w_agg_{max_clust}': Clustering_Approach(
        max_cluster_size=max_clust,
        n_centroids=n_centroids,
        initial_clustering='kmeans',
        agglomerative_clustering=True
    )
}

import umap
def umap_kmeans(X, n_clusters, random_state): 
    clusterer = umap.UMAP(
    **{"n_neighbors": 15, 
        "min_dist": 0.1, 
        "n_components": n_clusters, 
        "metric": "euclidean", 
        "random_state": random_state}
    )
    # X = X.swapaxes(0, 1)
    centroids = clusterer.fit_transform(X)
    return centroids#.swapaxes(0, 1)

OVERWRITE = False

clust_df, centroids = fetch_clustering(embeds, df, clustering_dict, overwrite=OVERWRITE)
# filtered_labels = weights < max_cluster_size
# labels[~filtered_labels] = -2
cluster_booleans, clust_results = evaluate_clustering(df, clust_df, embeds, clustering_dict, overwrite=OVERWRITE)

# #### filter clusters


# print(clust_results)


df_vis = fetch_visualization_df(clust_df, path, clustering_dict, umaps, overwrite=OVERWRITE, overwrite_gt=False)

    

remaining_settings = {**vars(settings)}

snr_str = f'snr={SNR}'
snr_val = SNR
vis_settings = {
    'models':list(embeds.keys()), 
    'audio_dir':path / snr_str, 
    'audio_suffixes' : ['.h5'],
    'main_results_dir':Path(settings.main_results_dir) / path.stem, 
    'default_label_keys':settings.metadata_label_keys,#{}, 
    'evaluation_task':config.evaluation_task, 
    'dim_reduction_model':config.dim_reduction_model, 
    'dim_reduc_parent_dir':settings.dim_reduc_parent_dir,
    'only_embed_annotations':True,
    # 'annotations_df' : df_vis,#[df_vis.snr.isin([snr_val, -1])],
    'constant_sr' : 32_000
}


for k in vis_settings.keys():
    if k in remaining_settings:
        remaining_settings.pop(k)
        
        
visualize_using_dashboard(
    **vis_settings,
    **remaining_settings
    )


def show_spec_of_h5idx(idx, df_vis, model='birdnet_v3'):
    from bacpipe.embedding_evaluation.visualization.visualize_spectrograms import SpectrogramPlot
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    
    model_name = SimpleNamespace(**{'options': []})
    spec = SpectrogramPlot(
        audio_dir=path,
        loader=None,
        model_name=model_name,
        paths=None,
        panel_static_text=None
        )
    spec.sample_rate = umaps[model][snr_str]['metadata']['sample_rate (Hz)']
    spec.segment_length = umaps[model][snr_str]['metadata']['segment_length (samples)']
    
    h5_file = df_vis.iloc[idx].audiofilename
    h5_idx = idx % 1180
    
    import h5py
    with h5py.File(path / snr_str / h5_file, 'r') as data:
        audio = data['audio'][h5_idx]
    
    
    fig = spec.create_specs(audio)
    fig.show()

def check_emb_from_idx_matches_saved_emb(idx, df_vis, model='birdnet_v3'):
    from bacpipe import Embedder
    import torch
    import numpy as np
    
    emb = Embedder(model_name=model)
    
    h5_file = df_vis.iloc[idx].audiofilename
    h5_idx = idx % 1180
    
    import h5py
    with h5py.File(path / snr_str / h5_file, 'r') as data:
        audio = data['audio'][h5_idx]
    audio = torch.tensor(audio.reshape(1, -1))
    
    embedding1 = emb.get_embeddings_for_audio(audio)
    
    embed_dir = Path(umaps[model][snr_str]['metadata']['embed_dir'])
    embedding2 = np.load(embed_dir / h5_file.replace('.h5', f'_{model}.npy'), mmap_mode='r')[h5_idx]
    
    return np.all(np.isclose(embedding1, embedding2, rtol=1e-4))

def check_umap_from_idx_matches_saved_umap(idx, df_vis, model='birdnet_v3'):
    from bacpipe import Loader
    import numpy as np
    import json
    
    loader = Loader(
        umaps[model][snr_str]['metadata']['audio_dir'], 
        model, 
        use_folder_structure=True,
        audio_suffixes=['.h5'], 
        dim_reduction_model='umap',
        main_results_dir=f'bacpipe_results/{path.stem}'
        )
    
    h5_file = df_vis.iloc[idx].audiofilename
    h5_idx = idx % 1180
    
    embed_dir = Path(umaps[model][snr_str]['metadata']['embed_dir'])
    embedding = np.load(embed_dir / h5_file.replace('.h5', f'_{model}.npy'), mmap_mode='r')[h5_idx]
    
    umap_dir = Path(loader.embed_dir)
    umap_data = json.load(open(umap_dir / f'{snr_str}_{model}.json', 'r'))
    x_and_y = umap_data['x'][idx], umap_data['y'][idx]
    
    from umap import UMAP
    import pickle

    umap_func = pickle.load((open(loader.embed_dir / 'umap_model.pkl', 'rb')))
    calc_umap = umap_func.transform(embedding.reshape(1, -1))
    
    # umap transforming after the fact is not not exactly representing it
    # therefore we have to use quite a weak tolerance
    return np.all(np.isclose(x_and_y, calc_umap, rtol=1e-1)) 
    
    
    

if False:
    ### check association when inside update_spectrogram:
    # get h5 path
    p = '/media/siriussound/Extreme SSD/identifying_unknown_sounds/data_h5_files/10_ratio-n2t_50/snr=0/unknown_sounds_len_3_sr_32000_nr-target_10_ratio-n2t_50_germany_campsite.h5'

    # load h5 idx arry
    import h5py
    with h5py.File(p, 'r') as data:
        au = data[587]
        
    # ensure that is identical to audio

    ## load numpy embedding at that idx

    npp = '/home/siriussound/Code/identifying_unknown_species/bacpipe_results/10_ratio-n2t_50/snr=0/embeddings/2026-08-12_12-00___birdnet-snr=0/unknown_sounds_len_3_sr_32000_nr-target_10_ratio-n2t_50_germany_campsite_birdnet.npy'
    em = np.load(npp)
    em[587]


    # load umap model
    from umap import UMAP
    import pickle
    up = '/home/siriussound/Code/identifying_unknown_species/bacpipe_results/10_ratio-n2t_50/snr=0/dim_reduced_embeddings/2026-08-12_12-03___umap-snr=0-birdnet/umap_model.pkl'
    um = pickle.load((open(up, 'rb')))

    # transform embedding into umap and check it matches the point in the visualization
    um.transform(em[587])


    ## and ensure the start and end match the csv files data
    p_csv = '/media/siriussound/Extreme SSD/identifying_unknown_sounds/data_h5_files/10_ratio-n2t_50/snr=0/unknown_sounds_len_3_sr_32000_nr-target_10_ratio-n2t_50_germany_campsite.csv'
    df = pd.read_csv(p_csv)
    df.iloc[587]
