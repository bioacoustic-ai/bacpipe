import yaml
import json
import re
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import datetime as dt

import logging

logger = logging.getLogger("bacpipe")


class DefaultLabels:
    def __init__(self, paths, model, default_label_keys, **kwargs):
        self.model = model
        self.default_label_keys = default_label_keys
        embed_path = model_specific_embedding_path(paths.main_embeds_path, model)
        self.metadata = yaml.safe_load(open(embed_path.joinpath("metadata.yml"), "r"))
        self.nr_embeds_per_file = self.metadata["files"]["nr_embeds_per_file"]
        self.nr_embeds_total = self.metadata["nr_embeds_total"]
        if not sum(self.nr_embeds_per_file) == self.nr_embeds_total:
            raise ValueError(
                "The number of embeddings per file does not match the total number of embeddings."
            )

    def generate(self):
        self.default_label_dict = {}
        for default_label in self.default_label_keys:
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
                    (time_of_day + index_of_embedding * segment_s_dt)
                    .time()
                    .replace(microsecond=0)
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
                    (datetime_per_file + index_of_embedding * segment_s_dt).replace(
                        microsecond=0
                    )
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


def make_set_paths_func(
    audio_dir,
    main_results_dir=None,
    dim_reduc_parent_dir="dim_reduced_embeddings",
    testing=False,
    **kwargs,
):
    if testing:
        main_results_dir = Path("bacpipe/tests/results_files")
        dim_reduc_parent_dir = "dimensionality_reduction"
    global get_paths

    def get_paths(model_name):
        """
        Generate model specific paths for the results of the embedding evaluation.
        This includes paths for the embeddings, labels, clustering, classification,
        distances, and plots. The paths are created based on the audio directory,
        and model name.

        Parameters
        ----------
        audio_dir : string
            full path to audio files
        model_name : string
            name of the model used for embedding
        main_results_dir : string
            top level directory for the results of the embedding evaluation

        Returns
        -------
        paths : SimpleNamespace
            object containing the paths for the results of the embedding evaluation
        """
        dataset_path = Path(main_results_dir).joinpath(Path(audio_dir).stem)
        task_path = dataset_path.joinpath("task_results").joinpath(model_name)

        paths = {
            "dataset_path": dataset_path,
            "dim_reduc_parent_dir": dataset_path.joinpath(dim_reduc_parent_dir),
            "main_embeds_path": dataset_path.joinpath("embeddings"),
            "labels_path": task_path.joinpath("labels"),
            "clust_path": task_path.joinpath("clustering"),
            "class_path": task_path.joinpath("classification"),
            "distances_path": task_path.joinpath("distances"),
            "plot_path": task_path.joinpath("plots"),
        }

        paths = SimpleNamespace(**paths)

        paths.main_embeds_path.mkdir(exist_ok=True, parents=True)
        paths.labels_path.mkdir(exist_ok=True, parents=True)
        paths.clust_path.mkdir(exist_ok=True)
        paths.class_path.mkdir(exist_ok=True)
        paths.distances_path.mkdir(exist_ok=True)
        paths.plot_path.mkdir(exist_ok=True)
        return paths

    return get_paths


def get_dim_reduc_path_func(model_name):
    return model_specific_embedding_path(
        get_paths(model_name).dim_reduc_parent_dir, model_name
    )


def get_default_labels(model_name, **kwargs):
    return create_default_labels(get_paths(model_name), model_name, **kwargs)


def get_ground_truth(model_name):
    return np.load(
        get_paths(model_name).labels_path.joinpath("ground_truth.npy"),
        allow_pickle=True,
    ).item()


def model_specific_embedding_path(path, model):
    if not isinstance(model, str):
        model = str(model)
    embed_paths_for_this_model = [
        d
        for d in path.iterdir()
        if d.is_dir() and model in d.stem.split("___")[-1].split("-")
    ]
    embed_paths_for_this_model.sort()
    if len(embed_paths_for_this_model) == 0:
        raise ValueError(
            f"No embeddings found for model {model} in {path}. "
            "Please check the directory path."
        )
    elif len(embed_paths_for_this_model) > 1:
        logger.info(
            "Multiple embeddings found for model {model} in {main_embeds_path}. "
            "Using the mosr recent path."
        )
    return embed_paths_for_this_model[-1]


def create_default_labels(paths, model, overwrite=True, **kwargs):
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


def concatenate_annotation_files(
    annotation_src,
    appendix=".txt",
    acodet_annotations=False,
    start_col_name="start",
    end_col_name="end",
    lab_col_name="label",
):
    # TODO needs testing
    p = Path(annotation_src)
    if acodet_annotations:
        ## This should always work for acodet combined annotations
        dfc = pd.read_csv(p.joinpath("combined_annotations.csv"))
        dfn = pd.read_csv(p.joinpath("explicit_noise.csv"))
        dfall = pd.concat([dfc, dfn])
        aud = dfall["filename"]
        auds = [Path(a).stem + ".wav" for a in aud]
        dfall["audiofilename"] = auds
        df = dfall[["start", "end", "label", "audiofilename"]]
    else:
        df = pd.DataFrame()
        for file in tqdm(
            p.rglob(f"*{appendix}"), desc="Loading annotations", leave=False
        ):
            try:
                ann = pd.read_csv(file, sep="\t", header=None)
            except pd.errors.EmptyDataError:
                continue
            df = pd.concat([df, dff], ignore_index=True)

        dff = pd.DataFrame()
        dff["start"] = ann[start_col_name]
        dff["end"] = ann[end_col_name]
        dff["label"] = ann[lab_col_name]
        dff["audiofilename"] = file.stem + ".wav"

    if True:
        short_to_species = pd.read_csv(
            "/mnt/swap/Work/Data/Amphibians/AnuranSet/species.csv"
        )
        for spe in df.label.unique():
            df.label[df.label == spe] = short_to_species.SPECIES[
                short_to_species.CODE == spe
            ].values[0]

    df.to_csv(
        p.joinpath("annotations.csv"),
        index=False,
    )


def filter_annotations_by_minimum_number_of_occurrences(
    df, min_occurrences=150, min_duration=0.65
):
    """
    Filter the annotations to have at least a minimum number of occurrences
    and a minimum duration.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the annotations.
    min_occurrences : int, optional
        Minimum number of occurrences for each label, by default 150.
    min_duration : float, optional
        Minimum duration for each label, by default 0.65.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing the annotations.
    """
    label_counts = df["label"].value_counts()
    labels_to_keep = label_counts[label_counts >= min_occurrences].index

    filtered_df = df[
        (df["label"].isin(labels_to_keep)) & ((df["end"] - df["start"]) >= min_duration)
    ]

    return filtered_df


def load_labels_and_build_dict(paths, label_file):
    # TODO filter to have at least 150 labels
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

    for _, row in df.iterrows():
        em_start = np.argmin(np.abs(embed_timestamps - row.start))
        em_end = np.argmin(np.abs(embed_timestamps - row.end))
        if single_label:
            if (
                not all(file_labels[em_start:em_end] == -1)
                and not label_idx_dict[row.label] in file_labels[em_start:em_end]
            ):
                single_label_arr[em_start:em_end] = [False] * (em_end - em_start)

        # at least 0.65 seconds of the bbox have
        # to be in the embedding timestamp window
        if row.end - row.start > 0.65:
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
    label_file="annotations.csv",
    overwrite=False,
    single_label=True,
    **kwargs,
):
    if overwrite or not paths.labels_path.joinpath("ground_truth.npy").exists():

        path = model_specific_embedding_path(paths.main_embeds_path, model)

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


def turn_multilabel_into_singlelabel(df_full):

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
