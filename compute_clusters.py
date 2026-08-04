from pathlib import Path
from clustering_utils import *

# main_path = Path('/mnt/swap/Work/Data/identifying_unknown_sounds_data')
# main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
# path = main_path / Path('data_h5_files/6_ratio-n2t_10_cleaned')
# file_name = 'unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_4.h5'
# file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem}.h5'
file_name = f'unknown_sounds_len_3_sr_32000_repetitions_{path.stem+"_snr=0"}.h5'#.split("_cleaned")[0]+

models = ['birdnet', 'naturebeats', 'audioprotopnet']#, 'avesecho_passt']


embeds, umaps = get_embeddings(path, models)



df = pd.DataFrame()
for model in models:
    for snr in embeds[model].keys():
        df_temp = load_df_same_order_as_embeddings(path, model, snr.split('=')[-1])
        df_temp['model'] = [model] * len(df_temp)
        df = pd.concat([df, df_temp])
df.index = range(len(df))

## compute clusterings
n_centroids = embeds['birdnet']['snr=12'].shape[0]//100
max_clust = 100
clustering_dict = {
    # 'kmeans': KMeans(n_clusters=n_centroids), # because 15 species + noise for the within and diff file ...?
    # # 'hdb': HDBSCAN(min_cluster_size=10, min_samples=None),
    # # 'spec': SpectralClustering(n_clusters=16),
    f'kmeans_w_dbscan_{max_clust}': MiniBatchkMeans_w_DBSCAN(
        max_cluster_size=max_clust,
        n_centroids=n_centroids,
        initial_clustering='kmeans'
        ),
    f'kmeans_w_dbscan_{max_clust}_filt_cent': MiniBatchkMeans_w_DBSCAN(
        max_cluster_size=max_clust,
        n_centroids=n_centroids,
        initial_clustering='kmeans',
        filter_centroids=True
        ),
    # f'kmeans_umap-init_w_dbscan_{max_clust}': MiniBatchkMeans_w_DBSCAN(
    #     max_cluster_size=max_clust,
    #     n_centroids=n_centroids, 
    #     initial_clustering='kmeans+umap'
    #     ),
    # f'kmeans_umap-init_half_w_dbscan_{max_clust}': MiniBatchkMeans_w_DBSCAN(
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

clust_df, centroids = fetch_clustering(embeds, df, clustering_dict, overwrite=True)
# filtered_labels = weights < max_cluster_size
# labels[~filtered_labels] = -2
cluster_booleans, clust_results = evaluate_clustering(df, clust_df, embeds, clustering_dict, overwrite=True)

#### filter clusters


print(clust_results)


df_vis = fetch_visualization_df(clust_df, path, clustering_dict, umaps, overwrite=True, overwrite_gt=True)

    

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
