import yaml
import json
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import datetime as dt

import umap

# import hdbscan

from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score as SS
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

import seaborn as sns
import matplotlib.pyplot as plt

# sns.set_theme(style="white")


import logging

logger = logging.getLogger("bacpipe")

import yaml

with open("bacpipe/settings.yaml", "rb") as p:
    settings = yaml.load(p, Loader=yaml.CLoader)


class DefaultLabels:
    def __init__(self, paths, model, **kwargs):
        self.model = model
        with open("bacpipe/settings.yaml", "r") as f:
            self.settings = yaml.safe_load(f)
        embed_path = model_specific_embedding_path(paths, model)
        self.metadata = yaml.safe_load(open(embed_path.joinpath("metadata.yml"), "r"))
        self.nr_embeds_per_file = self.metadata["files"]["nr_embeds_per_file"]
        self.nr_embeds_total = self.metadata["nr_embeds_total"]
        if not sum(self.nr_embeds_per_file) == self.nr_embeds_total:
            raise ValueError(
                "The number of embeddings per file does not match the total number of embeddings."
            )

    def generate(self):
        self.default_label_dict = {}
        for default_label in self.settings["default_labels"]:
            getattr(self, default_label)()

            self.default_label_dict.update(
                {default_label: getattr(self, f"{default_label}_per_embedding")}
            )

    @staticmethod
    def get_dt_filename(file):
        numbs = re.findall("[0-9]+", file)
        numbs = [n for n in numbs if len(n) % 2 == 0]

        i, datetime = 1, ""
        while len(datetime) < 12:
            if i > 1000:
                logger.warning(
                    f"Could not find a valid datetime in the filename {file}. "
                    "Please check the filename format."
                    "Creating a default datetime corresponding to 2000, 1, 1."
                )
                datetime = "20001010000000"
            datetime = "".join(numbs[-i:])
            i += 1

        i = 1
        while 12 <= len(datetime) > 14:
            datetime = datetime[:-i]

        for _ in range(2):
            try:
                if len(datetime) == 12:
                    file_date = dt.datetime.strptime(datetime, "%y%m%d%H%M%S")
                elif len(datetime) == 14:
                    file_date = dt.datetime.strptime(datetime, "%Y%m%d%H%M%S")
            except:
                i = 1
                while len(datetime) > 12:
                    datetime = datetime[:-i]
        return file_date

    def get_datetimes(self):
        if not hasattr(self, "timestamp_per_file"):
            self.timestamp_per_file = {}
            for file in self.metadata["files"]["audio_files"]:
                self.timestamp_per_file.update({file: self.get_dt_filename(file)})

    def time_of_day(self):
        self.get_datetimes()
        segment_s = (
            self.metadata["segment_length (samples)"]
            / self.metadata["sample_rate (Hz)"]
        )
        segment_s_dt = dt.timedelta(seconds=segment_s)
        time_of_day_per_file = {}
        for file, datetime_of_file in self.timestamp_per_file.items():
            timeofday = dt.datetime(
                2000,
                1,
                1,  # using a default day just to keep working with timestamps
                datetime_of_file.hour,
                datetime_of_file.minute,
                datetime_of_file.second,
            )
            time_of_day_per_file.update({file: timeofday})

        self.time_of_day_per_embedding = []
        for file_idx, (file, time_of_day) in enumerate(time_of_day_per_file.items()):
            for index_of_embedding in range(self.nr_embeds_per_file[file_idx]):
                self.time_of_day_per_embedding.append(
                    time_of_day + index_of_embedding * segment_s_dt
                )

    def day_of_year(self):
        self.get_datetimes()
        day_of_year_per_file = {}
        for file, datetime_of_file in self.timestamp_per_file.items():
            time_of_day = dt.datetime(
                2000, datetime_of_file.month, datetime_of_file.day
            )
            day_of_year_per_file.update({file: time_of_day})

        self.day_of_year_per_embedding = []
        for file_idx, (file, day_of_year) in enumerate(day_of_year_per_file.items()):
            self.day_of_year_per_embedding.extend(
                np.repeat(day_of_year, self.nr_embeds_per_file[file_idx])
            )

    def continuous_timestamp(self):
        self.get_datetimes()
        segment_s = (
            self.metadata["segment_length (samples)"]
            / self.metadata["sample_rate (Hz)"]
        )
        segment_s_dt = dt.timedelta(seconds=segment_s)

        self.continuous_timestamp_per_embedding = []
        for file_idx, (file, datetime_per_file) in enumerate(
            self.timestamp_per_file.items()
        ):
            for index_of_embedding in range(self.nr_embeds_per_file[file_idx]):
                self.continuous_timestamp_per_embedding.append(
                    datetime_per_file + index_of_embedding * segment_s_dt
                )

    def parent_directory(self):
        self.parent_directory_per_embedding = []
        for file_idx, file in enumerate(self.metadata["files"]["audio_files"]):
            self.parent_directory_per_embedding.extend(
                np.repeat(str(Path(file).parent), self.nr_embeds_per_file[file_idx])
            )

    def audio_file_name(self):
        self.audio_file_name_per_embedding = []
        for file_idx, file in enumerate(self.metadata["files"]["audio_files"]):
            self.audio_file_name_per_embedding.extend(
                np.repeat(file, self.nr_embeds_per_file[file_idx])
            )


def model_specific_embedding_path(paths, model):
    embed_paths_for_this_model = [
        d
        for d in paths.main_embeds_path.iterdir()
        if d.is_dir() and d.stem.split("___")[-1].split("-")[0] == model
    ]
    embed_paths_for_this_model.sort()
    if len(embed_paths_for_this_model) == 0:
        raise ValueError(
            f"No embeddings found for model {model} in {paths.main_embeds_path}. "
            "Please check the directory path."
        )
    elif len(embed_paths_for_this_model) > 1:
        logger.info(
            "Multiple embeddings found for model {model} in {main_embeds_path}. "
            "Using the mosr recent path."
        )
    return embed_paths_for_this_model[-1]


def create_default_labels(paths, model, audio_dir, overwrite=True, **kwargs):
    if overwrite or not paths.labels_path.joinpath("default_labels.npy").exists():

        default_labels = DefaultLabels(paths, model=model, **kwargs)
        default_labels.generate()

        np.save(
            paths.labels_path.joinpath("default_labels.npy"),
            default_labels,
        )
    else:
        default_labels = np.load(
            paths.labels_path.joinpath("default_labels.npy"), allow_pickle=True
        ).item()
    return default_labels.default_label_dict


def load_labels_and_build_dict(paths, label_file):
    label_df = pd.read_csv(paths.main_embeds_path.parent.joinpath(label_file))
    label_idx_dict = {label: idx for idx, label in enumerate(label_df.label.unique())}
    with open(paths.labels_path.joinpath("label_idx_dict.json"), "w") as f:
        json.dump(label_idx_dict, f)
    return label_df, label_idx_dict


def fit_labels_to_embedding_timestamps(
    df, label_idx_dict, num_embeds, segment_s, single_label=True, **kwargs
):
    file_labels = np.ones(num_embeds) * -1
    embed_timestamps = np.arange(num_embeds) * segment_s
    if single_label:
        single_label_arr = [True] * len(embed_timestamps)

    # assert (
    #     df.end.values[-1] <= embed_timestamps[-1] + segment_s*2
    # ), f"Timestamps do not match for {audio_file}"

    for _, row in df.iterrows():
        em_start = np.argmin(np.abs(embed_timestamps - row.start))
        em_end = np.argmin(np.abs(embed_timestamps - row.end))
        if single_label:
            if (
                not all(file_labels[em_start:em_end] == -1)
                and not label_idx_dict[row.label] in file_labels[em_start:em_end]
            ):
                single_label_arr[em_start:em_end] = [False] * (em_end - em_start)
        if (
            row.end - row.start > 0.65
        ):  # at least 0.65 seconds of the bbox have to be in the embedding timestamp window
            file_labels[em_start:em_end] = label_idx_dict[row.label]
    if single_label:
        file_labels[~np.array(single_label_arr)] = -2
    return file_labels


def build_ground_truth_labels_by_file(
    paths,
    ind,
    model,
    num_embeds,
    segment_s,
    metadata,
    all_labels,
    label_df=None,
    label_idx_dict=None,
    **kwargs,
):
    audio_file = metadata["files"]["audio_files"][ind]
    if "/" in audio_file:
        audio_file = Path(audio_file).stem + Path(audio_file).suffix
    df = label_df[label_df.audiofilename == audio_file]

    if df.empty:
        all_labels = np.concatenate((all_labels, np.ones(num_embeds) * -1))
        return all_labels

    file_labels = fit_labels_to_embedding_timestamps(
        df, label_idx_dict, num_embeds, segment_s, **kwargs
    )
    all_labels = np.concatenate((all_labels, file_labels))

    if np.unique(file_labels).shape[0] > 2:
        embed_timestamps = np.arange(num_embeds) * segment_s
        path = paths.labels_path.joinpath("raven_tables_for_sanity_check")
        path.mkdir(exist_ok=True, parents=True)
        if (
            len(list(path.iterdir())) < 10
        ):  # make sure to only do this a handful of times
            df_file_gt = label_df[label_df.audiofilename == audio_file]
            df_file_fit = pd.DataFrame()
            df_file_fit["start"] = embed_timestamps[file_labels > -1]
            df_file_fit["end"] = embed_timestamps[file_labels > -1] + segment_s
            inv = {v: k for k, v in label_idx_dict.items()}
            df_file_fit["label"] = [inv[i] for i in file_labels[file_labels > -1]]
            raven_gt = create_Raven_annotation_table(df_file_gt)
            raven_fit = create_Raven_annotation_table(df_file_fit)
            raven_fit["Low Freq (Hz)"] = 1500
            raven_fit["High Freq (Hz)"] = 2000
            raven_gt.to_csv(
                path.joinpath(f"{audio_file}_gt.txt"), sep="\t", index=False
            )
            raven_fit.to_csv(
                path.joinpath(f"{audio_file}_fit.txt"), sep="\t", index=False
            )
    return all_labels


def create_Raven_annotation_table(df):
    df.index = np.arange(1, len(df) + 1)
    raven_df = pd.DataFrame()
    raven_df["Selection"] = df.index
    raven_df.index = np.arange(1, len(df) + 1)
    raven_df["View"] = 1
    raven_df["Channel"] = 1
    raven_df["Begin Time (s)"] = df.start
    raven_df["End Time (s)"] = df.end
    raven_df["High Freq (Hz)"] = 1000
    raven_df["Low Freq (Hz)"] = 0
    raven_df["Label"] = df.label
    return raven_df


def collect_ground_truth_labels_by_file(
    paths, files, model, segment_s, metadata, label_df, label_idx_dict, **kwargs
):

    ground_truth = np.array([])

    for ind, file in tqdm(
        enumerate(files),
        desc=f"Loading {model} embeddings and split by labels",
        leave=False,
    ):
        assert (
            Path(metadata["files"]["audio_files"][ind]).stem
            == file.stem.split(f"_{model}")[0]
        ), (
            f"File names do not match for {file} and "
            f"{metadata['files']['audio_files'][ind]}"
        )

        num_embeds = metadata["files"]["nr_embeds_per_file"][ind]
        ground_truth = build_ground_truth_labels_by_file(
            paths,
            ind,
            model,
            num_embeds,
            segment_s,
            metadata,
            ground_truth,
            label_df,
            label_idx_dict,
            **kwargs,
        )
    return ground_truth


def ground_truth_by_model(
    paths,
    model,
    label_file=None,
    overwrite=False,
    single_label=True,
    remove_noise=False,
    **kwargs,
):
    if overwrite or not paths.labels_path.joinpath("ground_truth.npy").exists():

        path = model_specific_embedding_path(paths, model)

        label_df, label_idx_dict = load_labels_and_build_dict(paths, label_file)

        files = list(path.rglob("*.npy"))
        files.sort()

        metadata = yaml.safe_load(open(path.joinpath("metadata.yml"), "r"))
        segment_s = metadata["segment_length (samples)"] / metadata["sample_rate (Hz)"]

        ground_truth = collect_ground_truth_labels_by_file(
            paths,
            files,
            model,
            segment_s,
            metadata,
            label_df,
            label_idx_dict,
            single_label=single_label,
        )

        if remove_noise:
            if single_label:
                ground_truth = ground_truth[ground_truth > -1]
            else:
                ground_truth = ground_truth[ground_truth != -1]

        ground_truth_dict = {
            "labels": ground_truth,
            "label_dict": label_idx_dict,
        }
        np.save(paths.labels_path.joinpath("ground_truth.npy"), ground_truth_dict)
    else:
        ground_truth_dict = np.load(
            paths.labels_path.joinpath("ground_truth.npy"), allow_pickle=True
        ).item()
    return ground_truth_dict


def generate_annotations_for_classification_task(paths):
    if not paths.labels_path.joinpath("ground_truth.npy").exists():
        raise ValueError(
            "The ground truth label file ground_truth.npy does not exist. "
            "Please create it first."
        )
    ground_truth = np.load(
        paths.labels_path.joinpath("ground_truth.npy"), allow_pickle=True
    ).item()

    inv = {v: k for k, v in ground_truth["label_dict"].items()}
    labels = ground_truth["labels"][ground_truth["labels"] > -1]
    labs = [inv[i] for i in labels]
    df = pd.DataFrame()
    df["label"] = labs
    df["predefined_set"] = "lollinger"
    for v in inv.values():
        l = labs.count(v)
        ar = list(df[df.label == v].index)
        np.random.shuffle(ar)
        tr_ar = ar[: int(l * 0.65)]
        te_ar = ar[int(l * 0.65) : int(l * 0.85)]
        va_ar = ar[int(l * 0.85) :]
        df.predefined_set[tr_ar] = "train"
        df.predefined_set[te_ar] = "test"
        df.predefined_set[va_ar] = "val"
    df.to_csv(
        paths.labels_path.joinpath("classification_dataframe.csv"),
        index=False,
    )
