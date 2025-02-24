from exploration import explore_embeds
import sklearn
from tqdm import tqdm


models = [
    "birdnet",
    # "animal2vec_xc",
    "aves_especies",
    "avesecho_passt",
    "audiomae",
    "biolingual",
    "birdaves_especies",
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

# data_path = "id_task_data"
data_path = "colombia_soundscape"

label_file = (
    "/mnt/swap/Work/Data/neotropical coffee farms in "
    "Colombia and Costa Rica/colombia_soundscape/"
    "annotations_colombia_soundscape.csv"
)

##### CONFIGS #####

conf_2d_reduction = [
    {
        "name": "2dumap",
        "conf_1": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
        },
    }
]

reducer_conf = [
    {
        "name": "pca_50",
        "conf_1": {"n_components": 50},
    },
    {
        "name": "pca_100",
        "conf_1": {"n_components": 100},
    },
    {
        "name": "spca_50",
        "conf_1": {"n_components": 50},
    },
    {
        "name": "spca_100",
        "conf_1": {"n_components": 100},
    },
    {
        "name": "umap_50",
        "conf_1": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 50,
            "metric": "euclidean",
            "random_state": 42,
        },
    },
    {
        "name": "umap_100",
        "conf_1": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 100,
            "metric": "euclidean",
            "random_state": 42,
        },
    },
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
            "min_cluster_size": 7,
            "min_samples": 15,
            "metric": "euclidean",
        },
    },
    {"name": "kmeans", "conf_1": {"n_clusters": 6}},
]

clust_conf = conf_clust[-2:]

###### RUN ########

# clust_bar_plot(
#     [
#         'normal',
#         'pca50',
#         'pca100',
#         'spca50',
#         'spca100',
#     ]
# )
if True:
    explore_embeds.set_paths(data_path)
    embed_dict = explore_embeds.get_original_embeds(label_file=label_file)

    explore_embeds.compare(
        embed_dict,
        reducer_conf=reducer_conf,
        reducer_2d_conf=conf_2d_reduction[0],
        clust_conf=clust_conf,
        label_file=label_file,
    )
