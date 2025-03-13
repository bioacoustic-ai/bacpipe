from exploration import explore_embeds
import sklearn
from tqdm import tqdm


models = [
    "birdnet",
    "animal2vec_xc",
    "aves_especies",
    "avesecho_passt",
    "audiomae",
    "biolingual",
    "birdaves_especies",
    "google_whale",
    "hbdet",
    "insect66",
    "insect459",
    # "rcl_fs_bsed",
    "perch_bird",
    "protoclr",
    "surfperch",
    "vggish",
]

# data_path = "Evaluation_set_5shots"
# data_path = "id_task_data"
data_path = "neotropic_dawn_chorus"
# data_path = "colombia_soundscape"

label_file = (
    # None
    # "/mnt/swap/Work/Data/neotropical coffee farms in "
    # "Colombia and Costa Rica/colombia_soundscape/"
    # "annotations_colombia_soundscape.csv"
    "/home/siriussound/Code/bacpipe/exploration/neotropic_dawn_chorus/annotations.csv"
)
if False:
    from pathlib import Path
    import pandas as pd

    p = Path(
        "/home/siriussound/Code/bacpipe/exploration/neotropic_dawn_chorus/annotations.csv"
    )
    df = pd.read_csv(p)
    # df['label'] = file.parent.stem
    # df['start'] = df['Starttime']
    # df['end'] = df['Endtime']
    # df['Filename'] = f"{file.parent.stem}/{df['Audiofilename'][0]}"
    np.random.seed(42)
    dff = df
    dff["predefined_set"] = "train"
    for species in dff.label.unique():
        dd = dff[dff.label == species]
        ar = np.arange(len(dd))
        np.random.shuffle(ar)
        tr_ar = ar[: int(len(ar) * 0.65)]
        te_ar = ar[int(len(ar) * 0.65) : int(len(ar) * 0.85)]
        va_ar = ar[int(len(ar) * 0.85) :]
        dff.loc[dd.index[tr_ar], "predefined_set"] = "train"
        dff.loc[dd.index[te_ar], "predefined_set"] = "test"
        dff.loc[dd.index[va_ar], "predefined_set"] = "val"
    dff.rename(columns={"label": "species"}, inplace=True)
    dff.rename(columns={"Filename": "wavfilename"}, inplace=True)
    dff = dff[["wavfilename", "species", "predefined_set", "start", "end"]]
    dff.to_csv(p.parent.joinpath("task_annotations.csv"), index=False)

    model = "birdnet"
    for model in embed_dict.keys():
        inv = {v: k for k, v in embed_dict[model]["label_dict"].items()}
        labs = [inv[i] for i in embed_dict[model]["labels"]]
        # d = {}
        df = pd.DataFrame()
        df["species"] = labs
        df["predefined_set"] = "lollinger"
        for k, v in embed_dict[model]["split"].items():
            l = v.shape[0]
            # df.species = [k]*l
            ar = list(df[df.species == k].index)
            np.random.shuffle(ar)
            tr_ar = ar[: int(l * 0.65)]
            te_ar = ar[int(l * 0.65) : int(l * 0.85)]
            va_ar = ar[int(l * 0.85) :]
            df.predefined_set[tr_ar] = "train"
            df.predefined_set[te_ar] = "test"
            df.predefined_set[va_ar] = "val"
            # d[k] = df
        # dff = pd.concat(d.values())
        df.to_csv(p.parent.joinpath(f"{model}__task_annotations.csv"), index=False)


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
    # {
    #     "name": "pca_50",
    #     "conf_1": {"n_components": 50},
    # },
    {
        "name": "pca_300",
        "conf_1": {"n_components": 300},
    },
    # {
    #     "name": "spca_50",
    #     "conf_1": {"n_components": 50},
    # },
    # {
    #     "name": "spca_100",
    #     "conf_1": {"n_components": 100},
    # },
    # {
    #     "name": "umap_50",
    #     "conf_1": {
    #         "n_neighbors": 15,
    #         "min_dist": 0.1,
    #         "n_components": 50,
    #         "metric": "euclidean",
    #         "random_state": 42,
    #     },
    # },
    {
        "name": "umap_300",
        "conf_1": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 300,
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
    {"name": "kmeans", "conf_1": {"n_clusters": 11}},
    # {"name": "kmeans", "conf_1": {"n_clusters": 6}},
]

clust_conf = conf_clust[-1:]

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
    embed_dict = explore_embeds.get_original_embeds(
        label_file=label_file, remove_noise=True
    )

    explore_embeds.compare(
        embed_dict,
        reducer_conf=reducer_conf,
        reducer_2d_conf=conf_2d_reduction[0],
        clust_conf=clust_conf,
        label_file=label_file,
        remove_noise=True,
        distances=False,
    )
