from pathlib import Path
from clustering_utils import *

# main_path = Path('/mnt/swap/Work/Data/identifying_unknown_sounds_data')
# main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
# path = main_path / Path('data_h5_files/6_ratio-n2t_10_cleaned')
# file_name = 'unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_4.h5'
# file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem}.h5'
# file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem+"_snr=0"}.h5'#.split("_cleaned")[0]+

models = ['birdnet']#, 'naturebeats', 'audioprotopnet']#, 'avesecho_passt']


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
max_clust = 100
clustering_dict = {
    # 'kmeans': KMeans(n_clusters=n_centroids), # because 15 species + noise for the within and diff file ...?
    # # 'hdb': HDBSCAN(min_cluster_size=10, min_samples=None),
    # # 'spec': SpectralClustering(n_clusters=16),
    
    f'kmeans_w_agg_{max_clust}': Clustering_Approach(
        max_cluster_size=max_clust,
        n_centroids=n_centroids,
        initial_clustering='kmeans',
        agglomerative_clustering=True
        ),
    # f'kmeans_w_agg_{max_clust+100}': Clustering_Approach(
    #     max_cluster_size=max_clust,
    #     n_centroids=n_centroids,
    #     initial_clustering='kmeans',
    #     agglomerative_clustering=True
    #     ),
    # f'kmeans_w_dbscan_{max_clust}': Clustering_Approach(
    #     max_cluster_size=max_clust,
    #     n_centroids=n_centroids,
    #     initial_clustering='kmeans'
    #     ),
    # f'kmeans_w_dbscan_{max_clust}_filt_cent': Clustering_Approach(
    #     max_cluster_size=max_clust,
    #     n_centroids=n_centroids,
    #     initial_clustering='kmeans',
    #     filter_centroids=True
    #     ),
    # f'kmeans_umap-init_w_dbscan_{max_clust}': Clustering_Approach(
    #     max_cluster_size=max_clust,
    #     n_centroids=n_centroids, 
    #     initial_clustering='kmeans+umap'
    #     ),
    # f'kmeans_umap-init_half_w_dbscan_{max_clust}': Clustering_Approach(
    #     max_cluster_size=max_clust,
    #     n_centroids=2, 
    #     initial_clustering='kmeans+umap'
    #     )
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

#### filter clusters


print(clust_results)


df_vis = fetch_visualization_df(clust_df, path, clustering_dict, umaps, overwrite=OVERWRITE, overwrite_gt=False)

    

remaining_settings = {**vars(settings)}

snr_str = f'snr={SNR}'
snr_val = SNR
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
