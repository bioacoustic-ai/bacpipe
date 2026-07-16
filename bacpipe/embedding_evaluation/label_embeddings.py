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

from importlib import import_module
from librosa import get_duration

import logging
import bacpipe

logger = logging.getLogger("bacpipe")


class DefaultLabels:
    def __init__(self, paths, model, default_label_keys, **kwargs):
        """
        Class to generate default labels based on audio files and
        number of generated embeddings per file.

        Parameters
        ----------
        paths : SimpleNamespace
            convenient object for path handling
        model : str
            model name
        default_label_keys : list
            list of default labels, see settings.yaml

        Raises
        ------
        ValueError
            if no embeddings were found
        """
        self.model = model
        self.default_label_keys = default_label_keys
        self.paths = paths
        if kwargs.get("only_embed_annotations"):
            self.only_embed_annotations = True
            self.df, _ = load_labels_and_build_dict(
                paths,
                kwargs.get("annotations_filename"),
                self.paths.audio_dir,
                bool_filter_labels=False,
            )

        if (self.paths.preds_path / "original_classifier_outputs").exists():
            if not "default_classifier" in self.default_label_keys:
                self.default_label_keys += ["default_classifier"]
        elif "default_classifier" in self.default_label_keys:
            self.default_label_keys.remove("default_classifier")

        try:
            embed_path = model_specific_embedding_path(
                paths.main_embeds_path, model
            )
            self.metadata = load_metadata_file(embed_path)
            self.nr_embeds_per_file = self.metadata["files"][
                "nr_embeds_per_file"
            ]
            self.nr_embeds_total = self.metadata["nr_embeds_total"]
        except ValueError as e:
            logger.info(
                "No embeddings found. Gathering files and nr of embeddings "
                "per file from audio files."
            )
            _, _, metadata = get_files_if_no_embeds(paths.audio_dir, model)
            self.metadata = metadata
            self.nr_embeds_per_file = metadata["files"]["nr_embeds_per_file"]
            self.nr_embeds_total = sum(metadata["files"]["nr_embeds_per_file"])
        if not sum(self.nr_embeds_per_file) == self.nr_embeds_total:
            error = (
                "\nThe number of embeddings per file does not match "
                "the total number of embeddings."
            )
            logger.exception(error)
            raise ValueError(error)

    def generate(self):
        self.default_label_dict = {}
        for default_label in tqdm(
            self.default_label_keys, "Building default labels"
        ):
            getattr(self, default_label)()

            if hasattr(self, f"{default_label}_per_embedding"):
                self.default_label_dict.update(
                    {
                        default_label: getattr(
                            self, f"{default_label}_per_embedding"
                        )
                    }
                )

    def get_datetimes(self):
        if not hasattr(self, "timestamp_per_file"):
            self.timestamp_per_file = {}
            for file in tqdm(
                self.metadata["files"]["audio_files"], "collecting datetimes"
            ):
                file_stem = Path(file).stem
                self.timestamp_per_file.update(
                    {file: get_dt_filename(file_stem)}
                )

    def time_of_day(self):
        self.get_datetimes()
        segment_s = (
            self.metadata["segment_length (samples)"]
            / self.metadata["sample_rate (Hz)"]
        )
        segment_s_dt = dt.timedelta(seconds=float(segment_s))
        time_of_day_per_file = {}
        for file, datetime_of_file in tqdm(
            self.timestamp_per_file.items(), "getting time of day"
        ):
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
        for file_idx, (file, time_of_day) in tqdm(
            enumerate(time_of_day_per_file.items()),
            "getting time per embeddings",
        ):
            for index_of_embedding in range(self.nr_embeds_per_file[file_idx]):

                if hasattr(self, "only_embed_annotations") and getattr(
                    self, "only_embed_annotations"
                ):
                    from bacpipe import Loader

                    df = Loader.filter_df_by_file(
                        self.paths.audio_dir,
                        self.df,
                        Path(self.paths.audio_dir) / file,
                    )
                    starts = df.start.values
                    timestamp = (
                        (
                            time_of_day
                            + dt.timedelta(
                                seconds=float(starts[index_of_embedding])
                            )
                        )
                        .time()
                        .replace(microsecond=0)
                    )
                else:
                    timestamp = (
                        (time_of_day + index_of_embedding * segment_s_dt)
                        .time()
                        .replace(microsecond=0)
                    )
                self.time_of_day_per_embedding.append(
                    timestamp.strftime("%H-%M-%S")
                )

    def day_of_year(self):
        self.get_datetimes()
        day_of_year_per_file = {}
        for file, datetime_of_file in tqdm(
            self.timestamp_per_file.items(), "getting day of year"
        ):
            time_of_day = dt.datetime(
                2000, datetime_of_file.month, datetime_of_file.day
            )
            day_of_year_per_file.update({file: time_of_day})

        self.day_of_year_per_embedding = []
        for file_idx, (file, day_of_year) in enumerate(
            day_of_year_per_file.items()
        ):
            self.day_of_year_per_embedding.extend(
                np.repeat(
                    day_of_year.strftime("%Y-%m-%d"),
                    self.nr_embeds_per_file[file_idx],
                )
            )

    def continuous_timestamp(self):
        self.get_datetimes()
        segment_s = (
            self.metadata["segment_length (samples)"]
            / self.metadata["sample_rate (Hz)"]
        )
        segment_s_dt = dt.timedelta(seconds=segment_s)

        self.continuous_timestamp_per_embedding = []
        for file_idx, (file, datetime_per_file) in tqdm(
            enumerate(self.timestamp_per_file.items()),
            "getting continuous timestamps",
        ):
            for index_of_embedding in range(self.nr_embeds_per_file[file_idx]):

                if hasattr(self, "only_embed_annotations") and getattr(
                    self, "only_embed_annotations"
                ):
                    from bacpipe import Loader

                    df = Loader.filter_df_by_file(
                        self.paths.audio_dir,
                        self.df,
                        Path(self.paths.audio_dir) / file,
                    )
                    starts = df.start.values
                    timestamp = (
                        (
                            datetime_per_file
                            + dt.timedelta(
                                seconds=float(starts[index_of_embedding])
                            )
                        )
                        .time()
                        .replace(microsecond=0)
                    )
                else:
                    timestamp = (
                        datetime_per_file + index_of_embedding * segment_s_dt
                    ).replace(microsecond=0)
                self.continuous_timestamp_per_embedding.append(
                    timestamp.strftime("%Y-%m-%d_%H:%M:%S")
                )

    def parent_directory(self):
        self.parent_directory_per_embedding = []
        for file_idx, file in tqdm(
            enumerate(self.metadata["files"]["audio_files"]),
            "getting parent directory",
        ):
            self.parent_directory_per_embedding.extend(
                np.repeat(
                    str(Path(file).parent), self.nr_embeds_per_file[file_idx]
                )
            )

    def audio_file_name(self):
        self.audio_file_name_per_embedding = []
        for file_idx, file in tqdm(
            enumerate(self.metadata["files"]["audio_files"]),
            "getting audio file names",
        ):
            self.audio_file_name_per_embedding.extend(
                np.repeat(file, self.nr_embeds_per_file[file_idx])
            )

    def default_classifier(self):
        clfier_paths = list(
            self.paths.preds_path.rglob("*_classifier_annotations.csv")
        )
        if len(clfier_paths) == 0:
            self.default_label_keys.remove("default_classifier")
        else:
            path = clfier_paths[0]
            df = pd.read_csv(path)
            if not len(self.parent_directory_per_embedding) == len(df):
                df = self.fill_remaining_labels(df)
            self.default_classifier_per_embedding = df[
                "label:default_classifier"
            ].values.tolist()

    def fill_remaining_labels(self, df):
        from bacpipe import Loader

        seg_len = (
            self.metadata["segment_length (samples)"]
            / self.metadata["sample_rate (Hz)"]
        )
        df_new = {
            "start": [],
            "end": [],
            "audiofilename": [],
            "label:default_classifier": [],
        }
        for file, nr_embeds in zip(
            self.metadata["files"]["audio_files"],
            self.metadata["files"]["nr_embeds_per_file"],
        ):
            df_part = Loader.filter_df_by_file(
                self.paths.audio_dir, df, Path(self.paths.audio_dir) / file
            )
            # df_part = df[df.audiofilename == file]
            if hasattr(self, "only_embed_annotations") and getattr(
                self, "only_embed_annotations"
            ):
                df_tmp = Loader.filter_df_by_file(
                    self.paths.audio_dir,
                    self.df,
                    Path(self.paths.audio_dir) / file,
                )
                starts = df_tmp.start.values
                # starts = self.df.start[self.df.audiofilename == file]
                all_time_bins = np.round(starts, 4).tolist()
            else:
                all_time_bins = np.round(
                    np.arange(nr_embeds) * seg_len, 4
                ).tolist()

            try:
                [all_time_bins.remove(l) for l in np.round(df_part.start, 4)]
            except ValueError:
                exception_label = (
                    "\nThe timestamps from the precomputed predictions do not match those "
                    "of the generated embeddings. This is the case if a run has previously "
                    "been created for `only_embed_annotations=True` and now you are running "
                    "bacpipe with the setting False. Or the other way around. In this case "
                    "you have to recompute the embeddings. Please rename or delete the created "
                    "embeddings and evaluations folder to avoid problems."
                )
                logger.exception(exception_label)
                self.default_label_keys.remove("default_classifier")
                raise ValueError(exception_label)
                # import sys
                # sys.exit(1)
            df_new["start"].extend(all_time_bins)
            df_new["end"].extend((np.array(all_time_bins) + seg_len).tolist())
            df_new["audiofilename"].extend([file] * len(all_time_bins))
            df_new["label:default_classifier"].extend(
                ["below_thresh"] * len(all_time_bins)
            )

        df = pd.concat([df, pd.DataFrame(df_new)], ignore_index=True)
        if not len(df) == self.metadata["nr_embeds_total"]:
            raise AssertionError(
                "The number of points does not match the total number of embeddings."
            )
        return df.sort_values(["audiofilename", "start"])


def make_set_paths_func(
    audio_dir,
    main_results_dir=None,
    dim_reduc_parent_dir="dim_reduced_embeddings",
    testing=False,
    **kwargs,
):
    global get_paths

    def get_paths(model_name):
        """
        Generate model specific paths for the results of the embedding evaluation.
        This includes paths for the embeddings, labels, clustering, classification,
        and plots. The paths are created based on the audio directory,
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
        dataset_path = Path(main_results_dir).joinpath(
            Path(audio_dir).parts[-1]
        )
        
        task_path = dataset_path.joinpath(
            bacpipe.settings.evaluations_dir
            ).joinpath(
            model_name
        )  

        paths = {
            "audio_dir": audio_dir,
            "dataset_path": dataset_path,
            "dim_reduc_parent_dir": dataset_path.joinpath(
                dim_reduc_parent_dir
            ),
            "main_embeds_path": dataset_path.joinpath("embeddings"),
            "labels_path": task_path.joinpath("labels"),
            "clust_path": task_path.joinpath("clustering"),
            "probe_path": task_path.joinpath("probing"),
            "preds_path": task_path.joinpath("predictions"),
            "plot_path": task_path.joinpath("plots"),
        }

        paths = SimpleNamespace(**paths)

        paths.main_embeds_path.mkdir(exist_ok=True, parents=True)
        paths.labels_path.mkdir(exist_ok=True, parents=True)
        paths.clust_path.mkdir(exist_ok=True)
        paths.probe_path.mkdir(exist_ok=True)
        paths.plot_path.mkdir(exist_ok=True)
        return paths

    return get_paths


def get_dim_reduc_path_func(model_name, dim_reduction_model="umap", **kwargs):
    if dim_reduction_model in [None, "None", "", []]:
        dim_reduction_model = "umap"
        logger.warning(
            f"Dimensionality reduction model not specified. "
            f"Search for default dim_reduction_model: {dim_reduction_model}."
        )
    return model_specific_embedding_path(
        get_paths(model_name).dim_reduc_parent_dir,
        model_name,
        dim_reduction_model=dim_reduction_model,
        **kwargs,
    )


def ensure_windoof_path_to_posix(path):
    if "\\" in path:
        from pathlib import PureWindowsPath

        return str(PureWindowsPath(path).as_posix())
    else:
        return str(path)


def load_metadata_file(folder):
    with open(folder.joinpath("metadata.yml"), "r") as f:
        metadata_dict = yaml.load(f, Loader=yaml.CLoader)

    metadata_dict["audio_dir"] = ensure_windoof_path_to_posix(
        metadata_dict["audio_dir"]
    )
    metadata_dict["embed_dir"] = ensure_windoof_path_to_posix(
        metadata_dict["embed_dir"]
    )
    return metadata_dict


def get_default_labels(model_name, **kwargs):
    """
    Return dictionary of the default labels based on the files that were
    already processed and saved. This is model dependent, as the input length is
    model dependent and therefore this function requires a model name as input.
    The default labels are calculated based on the default labels specified in the
    settings.yaml file.

    Parameters
    ----------
    model_name : str
        model name

    Returns
    -------
    dict
        dictionary of default labels
    """
    paths = get_paths(model_name)
    return create_default_labels(paths.audio_dir, model_name, paths, **kwargs)


def get_ground_truth(model_name, file_path=None, return_type="dataframe"):
    """
    Return dictionary of the ground truth labels based on the files that were
    already processed and saved. This is model dependent, as the input length is
    model dependent and therefore this function requires a model name as input.

    Parameters
    ----------
    model_name : str
        model name

    Returns
    -------
    dict
        dictionary of ground truth labels
    """
    if return_type == "dataframe" and not file_path is None:
        return pd.read_csv(file_path, index_col=False)
    elif return_type == "array":
        return np.load(
            get_paths(model_name).labels_path.joinpath("ground_truth.npy"),
            allow_pickle=True,
        ).item()


def get_dt_filename(file):
    """
    Return the timestamp within a filename as a datetime object based on
    the most common naming conventions in bioacoustics. This is not bullet
    proof but it works with the vast majority of naming conventions for files.

    Parameters
    ----------
    file : str
        filename as string

    Returns
    -------
    dt.datetime object
        datetime object of the filename
    """
    if "+" in file:
        file = file.split("+")[0]
    numbs = re.findall("[0-9]+", file)
    numbs = [n for n in numbs if len(n) % 2 == 0]
    file_date = None

    i, datetime = 1, ""
    while len(datetime) < 12:
        if i > 1000:
            logger.warning(
                f"Could not find a valid datetime in the filename {file}. "
                "Please check the filename format."
                "Creating a default datetime corresponding to 2000, 1, 1."
            )
            datetime = "20001010000000"
            break
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

    # add fix if file_date is never created as a datetime object
    if file_date is None:
        logger.warning(
            f"Could not find a valid datetime in the filename {file}. "
            "Please check the filename format."
            "Creating a default datetime corresponding to 2000, 1, 1."
        )
        file_date = dt.datetime.strptime("20001010000000", "%y%m%d%H%M%S")
    return file_date


def model_specific_embedding_path(
    path, model, dim_reduction_model=None, **kwargs
):
    """
    Get the path to the model specific embeddings.
    This function searches for the most recent directory
    containing the embeddings for the specified model and
    dimensionality reduction model.

    Parameters
    ----------
    path : Path
        Path to the main embeddings directory.
    model : str
        Name of the model used for embedding.
    dim_reduction_model : str
        Name of the dimensionality reduction model used. Default is 'umap'.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    Path
        Path to the model specific embeddings directory.

    Raises
    -------
    ValueError
        If no embeddings are found for the specified model.
    """
    if not isinstance(model, str):
        model = str(model)
    embed_paths_for_this_model = [
        d
        for d in path.iterdir()
        if d.is_dir() and model in d.stem.split("___")[-1].split("-")
    ]
    if not dim_reduction_model in [None, "None", "", []]:
        embed_paths_for_this_model = [
            d
            for d in embed_paths_for_this_model
            if dim_reduction_model in d.stem
        ]
    embed_paths_for_this_model.sort()
    if len(embed_paths_for_this_model) == 0:
        error = (
            f"\nNo embeddings found for model {model} in {path}. "
            "Please check the directory path."
        )
        logger.exception(error)
        raise ValueError(error)
    elif len(embed_paths_for_this_model) > 1:
        logger.info(
            f"Multiple embeddings found for model {model} in {path}. "
            "Using the most recent path."
        )
    return embed_paths_for_this_model[-1]


def create_default_labels(
    audio_dir=None, model=None, paths=None, overwrite=True, **kwargs
):
    """
    Create default labels based on audio files and model timestamps to
    match the number of embeddings created per file for visualization
    and clustering purposes.

    Parameters
    ----------
    audio_dir : str, optional
        path to audio data, by default None
    model : str, optional
        model name, by default None
    paths : SimpleNamespace, optional
        convenient object for path handling, by default None
    overwrite : bool, optional
        if True labels are overwritten, by default True

    Returns
    -------
    dict
        dictionary with default labels
    """
    if paths is None:
        assign_global_get_paths_function(audio_dir)
        paths = get_paths(model)
    if (
        overwrite
        or not paths.labels_path.joinpath("default_labels.npy").exists()
    ):
        if not kwargs.get("default_label_keys"):
            from bacpipe import settings as bacpipe_settings

            kwargs["default_label_keys"] = bacpipe_settings.default_label_keys
        default_labels = DefaultLabels(
            paths, model=model, audio_dir=audio_dir, **kwargs
        )
        default_labels.generate()

        def_labels = default_labels.default_label_dict
        np.save(
            paths.labels_path.joinpath("default_labels.npy"),
            def_labels,
        )
    else:
        def_labels = np.load(
            paths.labels_path.joinpath("default_labels.npy"), allow_pickle=True
        ).item()
    return def_labels

def fetch_annotation_file(audio_dir, annotations_filename, paths):
    if annotations_filename is None:
        annotations_filename = bacpipe.settings.annotations_filename

    try:
        try:
            return pd.read_csv(
                Path(audio_dir).joinpath(annotations_filename)
            )
        except FileNotFoundError as e:
            try:
                return pd.read_csv(
                    Path(audio_dir).joinpath(annotations_filename)
                )
            except FileNotFoundError as e:
                logger.warning(
                    "No annotations file found, not able to create ground_truth.npy file. "
                    "bacpipe should still work, but you will not be able to label by ground truth. "
                    "You also will not be able to evaluate using classification."
                )
                raise FileNotFoundError("No annotations file found.")
    except FileNotFoundError as e:
        logger.warning(
            f"No annotations file found in {audio_dir}, trying in "
            f"{str(paths.dataset_path.resolve())}."
        )
        try:
            return pd.read_csv(
                paths.dataset_path.joinpath(annotations_filename)
            )
        except:
            logger.warning(
                "No annotations file found, not able to create ground_truth.npy file. "
                "bacpipe should still work, but you will not be able to label by ground truth. "
                "You also will not be able to evaluate using classification."
            )
            raise FileNotFoundError("No annotations file found.")
        
def filter_annotations(
    label_df,
    main_label_column, 
    min_label_occurrences,
    bool_filter_labels
    ):
    filtered_labels = [
        lab
        for lab in set(label_df[main_label_column])
        if len(label_df[label_df[main_label_column] == lab])
        > min_label_occurrences
    ]
    if not filtered_labels:
        logger.info(
            "\nBy filtering the annotations.csv file using the "
            f"{min_label_occurrences=}, no labels are left. In "
            "case you are just testing, the labels will not be filtered"
            f" and {bool_filter_labels=} will be ignored. If this "
            "a serious probing task, you will need more annotations. "
            "This might cause the probing or clustering to crash.\n"
        )
    else:
        return label_df[
            label_df[main_label_column].isin(filtered_labels)
        ]


def load_labels_and_build_dict(
    paths,
    annotations_filename,
    audio_dir,
    audio_files=[],
    bool_filter_labels=True,
    min_label_occurrences=150,
    main_label_column=None,
    testing=False,
    **kwargs,
):
    label_df = fetch_annotation_file(audio_dir, annotations_filename, paths)
    
    if bool_filter_labels and not testing:
        label_df = filter_annotations(
            label_df, main_label_column, min_label_occurrences, bool_filter_labels
        )
    
    if len(audio_files) > 0:
        from bacpipe import Loader

        filtered_df = pd.DataFrame()
        for file in audio_files:
            df_temp = Loader.filter_df_by_file(
                paths.audio_dir, 
                label_df, 
                Path(paths.audio_dir) / file
            )
            filtered_df = pd.concat([filtered_df, df_temp])
        label_df = filtered_df
        
    return label_df


def fit_labels_to_embedding_timestamps(
    df,
    df_fitted_gt,
    num_embeds,
    segment_s,
    label_column=None,
    min_annotation_length=0.65,
    only_embed_annotations=False,
    **kwargs,
):
    for col in df_fitted_gt.columns:
        df_fitted_gt[col] = np.zeros(num_embeds, dtype=np.int8)
    df = df.sort_values("start")

    if not only_embed_annotations:
        df_fitted_gt["starts"] = np.arange(num_embeds) * segment_s
        df_fitted_gt["ends"] = df_fitted_gt["starts"] + segment_s
    else:
        df_fitted_gt["starts"] = df["start"].values
        df_fitted_gt["ends"] = df["end"].values

    df.index = range(len(df))
    for _, row in df.iterrows():
        start_at_embed_nr = np.where(
            df_fitted_gt["starts"] - row.start <= 0
            )[0][-1]
        end_at_embed_nr = np.where(df_fitted_gt["starts"] - row.end >= 0)[0]
        if len(end_at_embed_nr) > 0:
            end_at_embed_nr = end_at_embed_nr[0]
        else:
            end_at_embed_nr = len(df_fitted_gt["starts"])
        for idx in range(start_at_embed_nr, end_at_embed_nr):

            # check if the annotation length is longer that the specified min_annotation_length
            if (row.end - row.start > min_annotation_length):
                df_fitted_gt.loc[idx, row[f"label:{label_column}"]] = 1
            else:
                logger.info(
                    f"\nSkipping annotation from {row.start} to {row.end} with "
                    f"label {row['label:species']} because the annotation is "
                    f"shorter than {min_annotation_length=}. To change this, "
                    "modify the value in the settings file."
                )
                
    df_fitted_gt["species_richness"] = df_fitted_gt.drop(
        columns=["starts", "ends", "audiofilename", "species_richness"]
        ).sum(axis=1)
    if df_fitted_gt["species_richness"].max() > 1:
        logger.warning(
            "The species richness column of the ground truth has "
            "values exceeding 1. This means you have multi-label "
            "ground truth annotations. If this should not be "
            "happening ensure the ground truth is created correcly."
        )
    elif df_fitted_gt["species_richness"].max() == 0:
        logger.warning(
            "The species richness column of the ground truth has a "
            "maximum value of 0. This means no annotations have been"
            "found for your data. Something failed in building the "
            "ground truth array. Please ensure the audio filenames "
            "match the names in the names in the annotations file."
        )
    return df_fitted_gt



def build_ground_truth_labels_by_file(
    ind,
    model,
    num_embeds,
    segment_s,
    metadata,
    all_labels,
    label_df=None,
    label_column=None,
    only_embed_annotations=False,
    **kwargs,
):
    audio_file = metadata["files"]["audio_files"][ind]
    df = filter_df_by_filename(label_df, audio_file, model=model)

    file_labels = pd.DataFrame(columns=all_labels.columns)
    file_labels = fit_labels_to_embedding_timestamps(
        df,
        file_labels,
        num_embeds,
        segment_s,
        label_column=label_column,
        only_embed_annotations=only_embed_annotations,
        **kwargs,
    )
    file_labels["audiofilename"] = audio_file
    all_labels = pd.concat([all_labels, file_labels])
    return all_labels


def filter_df_by_filename(
    df_to_filter, file_name, file_name_column="audiofilename", model=None
):
    df = df_to_filter[
        df_to_filter[file_name_column] == Path(file_name).as_posix()
    ]
    if len(df) == 0:
        df = df_to_filter[
            df_to_filter[file_name_column]
            == (Path(file_name).stem + Path(file_name).suffix)
        ]
        
    # if no files are found, ensure parent path is not the cause
    if len(df) == 0:
        df = df_to_filter[
            np.array([
                Path(f).stem + Path(f).suffix 
                for f in df_to_filter[file_name_column]
                ]) 
            == file_name
        ]
        
    # if no files are found, ensure parent path is not the cause
    if len(df) == 0:
        df = df_to_filter[
            np.array([
                Path(f).stem + Path(f).suffix 
                for f in df_to_filter[file_name_column]
                ]) 
            == (Path(file_name).stem + Path(file_name).suffix)
        ]
        
    # if no files are found, match by classifier_prediction files
    if len(df) == 0:
        df = df_to_filter[
            df_to_filter[file_name_column]
            == Path(file_name).parent
            / (Path(file_name).stem + f"_{model}.json")
        ]
    
    return df


def create_Raven_annotation_table(df, label_column, high_freq=1000):
    df.index = np.arange(1, len(df) + 1)
    raven_df = pd.DataFrame()
    raven_df["Selection"] = df.index
    raven_df.index = np.arange(1, len(df) + 1)
    raven_df["View"] = "Spectrogram 1"
    raven_df["Channel"] = 1
    raven_df["Begin Time (s)"] = df.start
    raven_df["End Time (s)"] = df.end
    raven_df["Low Freq (Hz)"] = 0
    raven_df["High Freq (Hz)"] = high_freq
    raven_df["Label"] = df[f"label:{label_column}"]
    return raven_df

def ensure_file_names_match(metadata, ind, file, model):
    assert (
        Path(metadata["files"]["audio_files"][ind]).stem
        == file.stem.split(f"_{model}")[0]
    ), (
        f"File names do not match for {file} and "
        f"{metadata['files']['audio_files'][ind]}"
    )

def initialize_ground_truth_df(label_df, label_column):
    # Get species names
    species_cols = label_df[f"label:{label_column}"].unique().tolist()

    # This ensures all species columns and numeric columns are floats from the start
    return pd.DataFrame(
        {
            **{col: pd.Series(dtype="int8") for col in species_cols},
            "species_richness": pd.Series(dtype="int8"),
            "audiofilename": pd.Series(
                dtype="string"
            ),
            "ends": pd.Series(dtype="int8"),
            "starts": pd.Series(dtype="int8"),
        }
    )

def collect_ground_truth_labels(
    files,
    model,
    segment_s,
    metadata,
    label_df,
    label_column,
    **kwargs,
):
    ground_truth = initialize_ground_truth_df(label_df, label_column)
    
    for ind, file in tqdm(
        enumerate(files),
        desc=f"Collecting annotations and fitting to embeddings timestamps",
        total=len(files),
        leave=False,
    ):
        ensure_file_names_match(metadata, ind, file, model)
        num_embeds = metadata["files"]["nr_embeds_per_file"][ind]
        
        ground_truth = build_ground_truth_labels_by_file(
            ind,
            model,
            num_embeds,
            segment_s,
            metadata,
            ground_truth,
            label_df,
            label_column=label_column,
            **kwargs,
        )
    return ground_truth


def assign_global_get_paths_function(audio_dir):
    if not "get_paths" in globals():
        from bacpipe import settings as bapcipe_settings

        make_set_paths_func(audio_dir, bapcipe_settings.main_results_dir)


def ground_truth_by_model(
    model,
    audio_dir,
    label_df=None,
    label_column="label:species",
    paths=None,
    annotations_filename="annotations.csv",
    only_embed_annotations=False,
    overwrite=True,
    bool_filter_labels=False,
    **kwargs,
):
    """
    Generate ground truth labels that are mapped onto the
    timestamps of a model, based on the model-specific
    input lengths. This way the embeddings and ground truth
    labels have the same lengths, and can be used for downstream
    evaluation like probing or clustering.
    This function supports single or multi-label generation
    of ground truth labels.
    A dictionary is created with a numpy array for the labels
    and a dictionary to associate the int values with the
    corresponding label class.
    The labels are processed based on a single annotation file
    which requires predefined column names:
    `audiofilename`, `start`, `end`, `label:species` (species
    can be replaced with other things but the `label:` needs to
    be consistent). See 'bacpipe/tests/test_data/annotations.csv'
    for an example.
    After processing the ground truth, the dictionary is saved
    as a numpy file and upon reexecution is simply loaded for
    shorter runtime.

    Parameters
    ----------
    model : str
        model name
    audio_dir : str
        path to audio data
    label_df : pandas.DataFrame, optional
        ground truth annotations in specified format, by default None
    label_column : str, optional
        name of column in annotation file, by default 'label:species'
    paths : SimpleNamespace, optional
        convenient object for path handling, by default None
    annotations_filename : str, optional
        path to annotations csv file, by default "annotations.csv"
    only_embed_annotations : bool, optional
        If True the time stamps from the existing annotations are used
        rather than creating a grid based on the model specific
        input length, defaults to False
    overwrite : bool, optional
        If True, the dict will be generated again and saved
        rather than loaded from a file if already
        processed, by default True
    bool_filter_labels : bool, optional
        set to True, if you want a minimum number of occurrence
        for labels to be included in the ground truth. See
        settings file for more options and descriptions, by default False

    Returns
    -------
    dict
        dictionary of ground truth labels with numpy array
        and dict to link int values to class labels

    Raises
    ------
    ValueError
        if gorund truth file is not found
    """
    if paths is None:
        assign_global_get_paths_function(audio_dir)
        paths = get_paths(model)

    if (
        overwrite
        or not paths.labels_path.joinpath(f"ground_truth_species.csv").exists()
    ):

        # check if embeddings exist
        try:
            path = model_specific_embedding_path(paths.main_embeds_path, model)
        except Exception as e:
            logger.warning(f"No embeddings directory seems to exist. {e}")
            path = None

        # get annotations is not provided
        if label_df is None:
            if not "label:" in label_column:
                label_column = "label:" + label_column
            if kwargs.get("testing"):
                annotations_filename = "annotations.csv"
            label_df = load_labels_and_build_dict(
                paths,
                annotations_filename,
                main_label_column=label_column,
                audio_dir=audio_dir,
                bool_filter_labels=bool_filter_labels,
                **kwargs,
            )

        # build files, segment_s and metadata variables
        # depending if embeddings exist or not
        if path is not None and len(list(path.iterdir())) > 0:
            files = list(path.rglob("*.npy"))
            files.sort()
            
            try:
                metadata = load_metadata_file(path)
                segment_s = (
                    metadata["segment_length (samples)"]
                    / metadata["sample_rate (Hz)"]
                )
            except:
                files, segment_s, metadata = get_files_if_no_embeds(
                    audio_dir, model, label_df, only_embed_annotations
                )    
        else:
            files, segment_s, metadata = get_files_if_no_embeds(
                audio_dir, model, label_df, only_embed_annotations
            )

        # find all label columns
        label_columns = [col for col in label_df.columns if "label:" in col]

        # collect all the ground truth for all the label columns
        for label_col in label_columns:
            clean_label_column = label_col.split("label:")[-1]
            ground_truth = collect_ground_truth_labels(
                files,
                model,
                segment_s,
                metadata,
                label_df,
                label_column=clean_label_column,
                only_embed_annotations=only_embed_annotations,
                **kwargs,
            )
            cols = list(ground_truth.columns)[::-1]
            ground_truth = ground_truth[cols]
            ground_truth = ground_truth.sort_values(
                by=["audiofilename", "starts"]
            )
            ground_truth.to_csv(
                paths.labels_path.joinpath(
                    f"ground_truth_{clean_label_column}.csv"
                ),
                index=False,
            )
        if (
            not clean_label_column == label_column
            and not clean_label_column in label_column
        ):
            if ":" in label_column:
                label_column = label_column.split(":")[-1]

            ground_truth = pd.read_csv(
                paths.labels_path.joinpath(f"ground_truth_{label_column}.csv"),
                index_col=False,
            )

    else:
        clean_label_column = label_column.split("label:")[-1]
        ground_truth = pd.read_csv(
            paths.labels_path.joinpath(
                f"ground_truth_{clean_label_column}.csv"
            ),
            index_col=False,
        )
    return ground_truth


def ensure_audio_files(found_audio_files, annotated_audio_files, audio_dir):
    if not annotated_audio_files:
        return found_audio_files
    matching = set(found_audio_files).intersection(set(annotated_audio_files))
    if len(matching) < len(annotated_audio_files) or len(matching) == 0:
        relative_to_audio_dir = [
            Path(f).relative_to(audio_dir) for f in found_audio_files
        ]
        matching = set(relative_to_audio_dir).intersection(
            set(annotated_audio_files)
        )

    if len(matching) < len(annotated_audio_files) or len(matching) == 0:
        annotated_stems = [Path(f).stem for f in annotated_audio_files]
        found_stems = [Path(f).stem for f in found_audio_files]
        matching = set(annotated_stems).intersection(set(found_stems))

    if len(matching) < len(annotated_audio_files) or len(matching) == 0:
        not_found = []
        found_annotated_audio_files = [
            (
                list(Path(audio_dir).rglob(f"*{f.stem + f.suffix}"))[0]
                if list(Path(audio_dir).rglob(f"*{f.stem + f.suffix}"))
                else not_found.append(f)
            )
            for f in annotated_audio_files
        ]
        if not_found:
            logger.warning(
                f"{not_found} were not found in {audio_dir}. "
                "Are you sure you entered the correct path to the audio data?"
            )
        if len(found_annotated_audio_files) > 0:
            found_annotated_audio_files = found_audio_files

    return [str(f) for f in found_audio_files]


def get_files_if_no_embeds(audio_dir, model, label_df=None, only_embed_annotations=False):
    if label_df is None:
        annotated_audio_files = []
    else:
        annotated_audio_files = label_df.audiofilename.unique()
        annotated_audio_files = [Path(f) for f in annotated_audio_files]

    module = import_module(
        f"bacpipe.model_pipelines.feature_extractors.{model}"
    )
    segment_s = module.LENGTH_IN_SAMPLES / module.SAMPLE_RATE

    metadata = {}
    metadata["files"] = {}
    from bacpipe import get_audio_files

    found_audio_files = get_audio_files(audio_dir)
    matching_audio_files = ensure_audio_files(
        found_audio_files, annotated_audio_files, audio_dir
    )
    matching_audio_files.sort()

    metadata["segment_length (samples)"] = module.LENGTH_IN_SAMPLES
    metadata["sample_rate (Hz)"] = module.SAMPLE_RATE
    metadata["files"]["audio_files"] = matching_audio_files
    if only_embed_annotations:
        metadata["files"]["nr_embeds_per_file"] = [
            len(filter_df_by_filename(label_df, f, model=model)) 
            for f in matching_audio_files
        ]
    else:
        metadata["files"]["nr_embeds_per_file"] = [
            int(get_duration(path=f) / segment_s) for f in matching_audio_files
        ]
    files = [Path(f"{Path(d).stem}_{model}") for d in matching_audio_files]

    return files, segment_s, metadata
