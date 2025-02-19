from exploration.explore_embeds import compare, get_original_embeds, clust_bar_plot
import sklearn
from tqdm import tqdm

# models = [
#     "birdnet",
#     "insect66",
#     "protoclr",
#     "surfperch",
#     "perch_bird",
#     "biolingual",
#     "aves_especies",
#     "birdaves_especies",
#     "animal2vec_xc",
#     "avesecho_passt",
# ]
models = [
    "animal2vec_xc",
    "aves_especies",
    "avesecho_passt",
    "audiomae",
    "biolingual",
    "birdaves_especies",
    "birdnet",
    "google_whale",
    "hbdet",
    "insect66",
    "insect459",
    "rcl_fs_bsed",
    "perch_bird",
    "protoclr",
    "surfperch",
    "vggish",
]


##### CONFIGS #####

conf_dim_reduc = [
    {
        "name": "umap",
        "conf_1": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
        },
    }
]

conf_clust = [
    {
        "name": "hdbscan",
        "conf_1": {
            "min_cluster_size": 10,
            "min_samples": 5,
            "metric": "euclidean",
        },
    },
    {
        "name": "hdbscan",
        "conf_2": {
            "min_cluster_size": 30,
            "min_samples": 15,
            "metric": "euclidean",
        },
    },
    {
        "name": "hdbscan",
        "conf_3": {
            "min_cluster_size": 15,
            "min_samples": 15,
            "metric": "euclidean",
        },
    },
    {"name": "kmeans", "conf_1": {"n_clusters": 6}},
]

reducer_conf = conf_dim_reduc[0]
clust_conf = conf_clust[-2:]

###### RUN ########

# clust_bar_plot(
#     [
#         'normal',
#         'pca50',
#         'pca100',
#         # 'spca50',
#         # 'spca100',
#     ]
# )
if True:
    embed_dict = get_original_embeds(models)

    compare(embed_dict, "normal", reducer_conf=reducer_conf, clust_conf=clust_conf)

    # DOWNSAMPLE TO 100 DIMENSIONS
    # dict_reduced = embed_dict.copy()
    # pca = sklearn.decomposition.SparsePCA(n_components=50)
    # for model, embed in tqdm(embed_dict.items()):
    #     dict_reduced[model]['all'] = pca.fit_transform(embed['all'])

    # compare(dict_reduced, 'pca50',
    #         reducer_conf=reducer_conf,
    #         clust_conf=clust_conf)
