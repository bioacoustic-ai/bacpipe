from exploration import explore_embeds
import sklearn
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path


# DATA_PATH = "Evaluation_set_5shots"
# DATA_PATH = "id_task_data"
# DATA_PATH = "neotropic_dawn_chorus"
# DATA_PATH = "colombia_soundscape"
DATA_PATH = "anuran_set"

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


def read_annotations():
    if not explore_embeds.main_embeds_path.parent.joinpath("annotations.csv").exists():
        p = Path("/mnt/swap/Work/Data/Amphibians/AnuranSet/strong_labels")
        df = pd.DataFrame()
        for file in tqdm(p.rglob("*.txt")):
            try:
                ann = pd.read_csv(file, sep="\t", header=None)
            except pd.errors.EmptyDataError:
                continue
            dff = pd.DataFrame()
            dff["start"] = ann[0]
            dff["end"] = ann[1]
            dff["label"] = [a.split("_")[0] for a in ann[2]]
            dff["audiofilename"] = file.stem + ".wav"
            df = pd.concat([df, dff], ignore_index=True)
        if True:
            short_to_species = pd.read_csv(
                "/mnt/swap/Work/Data/Amphibians/AnuranSet/species.csv"
            )
            for spe in df.label.unique():
                df.label[df.label == spe] = short_to_species.SPECIES[
                    short_to_species.CODE == spe
                ].values[0]
        df.to_csv(
            explore_embeds.main_embeds_path.parent.joinpath("annotations.csv"),
            index=False,
        )
    else:
        df = pd.read_csv(
            explore_embeds.main_embeds_path.parent.joinpath("annotations.csv")
        )
    if not explore_embeds.main_embeds_path.parent.joinpath(
        "annotations_single_label.csv"
    ).exists():
        df = remove_multilabel(df)
        # Filter annotations
        a, b = np.unique(df.label, return_counts=True)
        c = [aa for aa, bb in zip(a, b) if bb > 150]
        df_filtered = df[df.label.isin(c)]
        df_filtered.to_csv(
            explore_embeds.main_embeds_path.parent.joinpath(
                "annotations_single_label.csv"
            ),
            index=False,
        )
    else:
        df = pd.read_csv(
            explore_embeds.main_embeds_path.parent.joinpath(
                "annotations_single_label.csv"
            )
        )


def remove_multilabel(df_full):
    df = pd.DataFrame()
    for file in tqdm(
        df_full.audiofilename.unique(),
        desc="Removing multi-labels",
        total=len(df_full.audiofilename.unique()),
        leave=False,
    ):
        dff = df_full[df_full.audiofilename == file]
        if len(dff.label.unique()) > 1:
            for _ in range(len(dff)):
                dff = dff.sort_values("start")
                dff.index = range(len(dff))
                number_of_changes = 0

                one_overarching_sound = 0
                for idx in range(len(dff)):
                    if idx in dff.index:
                        row = dff.loc[idx]
                    else:
                        continue

                    begins_within = (
                        (dff.start > row.start)
                        & (dff.start < row.end)
                        & (dff.end > row.end)
                    )
                    ends_within = (
                        (dff.end > row.start)
                        & (dff.end < row.end)
                        & (dff.start < row.start)
                    )
                    complete_within = (
                        (dff.start >= row.start)
                        & (dff.end <= row.end)
                        & (dff.index != idx)
                    )

                    new_ends = dict()
                    new_starts = dict()

                    if all(
                        complete_within[dff.index != idx]
                    ):  # strict case, meaning as soon as there is a sound going from beg to end we skip
                        one_overarching_sound += 1
                        if one_overarching_sound > 1:
                            break

                    if any(begins_within.values):
                        new_ends["begins_within"] = dff[
                            begins_within
                        ].start.values.tolist()

                    if any(ends_within.values):
                        new_starts["ends_within"] = dff[ends_within].end.values.tolist()

                    if any(complete_within.values):
                        new_starts["complete_within"] = dff[
                            complete_within
                        ].end.values.tolist()
                        new_ends["complete_within"] = dff[
                            complete_within
                        ].start.values.tolist()
                        new_starts["complete_within"].insert(0, row.start)
                        new_ends["complete_within"].append(row.end)

                    if "ends_within" in new_starts.keys():
                        max_ind = dff.index.max()
                        for i in range(len(new_starts["ends_within"])):
                            row.name = max_ind + i + 1
                            dff = pd.concat([dff, pd.DataFrame(row).T])
                            index = dff.index[-1]
                            dff.loc[index, "start"] = new_starts["ends_within"][i]

                    if "begins_within" in new_ends.keys():
                        max_ind = dff.index.max()
                        for i in range(len(new_ends["begins_within"])):
                            row.name = max_ind + i + 1
                            dff = pd.concat([dff, pd.DataFrame(row).T])
                            index = dff.index[-1]
                            dff.loc[index, "end"] = new_ends["begins_within"][i]

                    if "complete_within" in new_starts.keys():
                        max_ind = dff.index.max()
                        for i in range(len(new_starts["complete_within"])):
                            if (
                                new_starts["complete_within"][i]
                                >= new_ends["complete_within"][i]
                            ):
                                continue
                                # this is the case if two overlapping sounds start exactly the same time
                            row.name = max_ind + i + 1
                            dff = pd.concat([dff, pd.DataFrame(row).T])
                            # index = max_ind + i
                            dff.loc[row.name, "start"] = new_starts["complete_within"][
                                i
                            ]
                            dff.loc[row.name, "end"] = new_ends["complete_within"][i]
                    bool_combination = begins_within ^ complete_within ^ ends_within
                    dff.drop(
                        dff.loc[bool_combination.index][bool_combination].index,
                        inplace=True,
                    )
                    if any(bool_combination):
                        number_of_changes += 1

                    if (
                        any(begins_within.values)
                        or any(ends_within.values)
                        or any(complete_within.values)
                    ):
                        dff.drop(idx, inplace=True)
                dff = dff[
                    dff.end - dff.start > 0.2
                ]  # minimum of 0.2 seconds vocalization
                if number_of_changes == 0:
                    break
                # dff.drop_duplicates(inplace=True)

        dff = dff.sort_values("start")
        if dff.isna().sum().sum() > 0:
            print(dff)
            print("NA values in the dataframe")
        df = pd.concat([df, dff], ignore_index=True)
    return df


def generate_annotations(embed_dict):
    for model in embed_dict.keys():
        inv = {v: k for k, v in embed_dict[model]["label_dict"].items()}
        labs = [inv[i] for i in embed_dict[model]["labels"]]
        df = pd.DataFrame()
        df["species"] = labs
        df["predefined_set"] = "lollinger"
        for k, v in embed_dict[model]["split"].items():
            l = v.shape[0]
            ar = list(df[df.species == k].index)
            np.random.shuffle(ar)
            tr_ar = ar[: int(l * 0.65)]
            te_ar = ar[int(l * 0.65) : int(l * 0.85)]
            va_ar = ar[int(l * 0.85) :]
            df.predefined_set[tr_ar] = "train"
            df.predefined_set[te_ar] = "test"
            df.predefined_set[va_ar] = "val"
        df.to_csv(
            explore_embeds.main_embeds_path.parent.joinpath("annotations")
            .joinpath("task_annotations")
            .joinpath(f"{model}__task_annotations.csv"),
            index=False,
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
    {"name": "kmeans", "conf_1": {"n_clusters": 18}},
    # {"name": "kmeans", "conf_1": {"n_clusters": 11}},
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
    explore_embeds.set_paths(DATA_PATH)
    read_annotations()
    label_file = explore_embeds.main_embeds_path.parent.joinpath(
        "annotations_single_label.csv"
    )

    embed_dict = explore_embeds.get_original_embeds(
        label_file=label_file, remove_noise=True
    )
    generate_annotations(embed_dict)

    explore_embeds.compare(
        embed_dict,
        reducer_conf=reducer_conf,
        reducer_2d_conf=conf_2d_reduction[0],
        clust_conf=clust_conf,
        label_file=label_file,
        remove_noise=True,
        distances=False,
    )
