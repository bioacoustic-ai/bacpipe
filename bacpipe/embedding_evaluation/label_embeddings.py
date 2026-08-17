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
        Class to generate metadata labels based on audio files and
        number of generated embeddings per file.

        Parameters
        ----------
        paths : SimpleNamespace
            convenient object for path handling
        model : str
            model name
        default_label_keys : list
            list of metadata labels, see settings.yaml

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
            self.df = load_labels_and_build_dict(
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
        """
        Generate all default metadata labels.

        For each key in ``default_label_keys`` the corresponding method is
        called and the per-embedding labels are collected in
        ``default_label_dict``.
        """
        self.default_label_dict = {}
        for default_label in tqdm(
            self.default_label_keys, "Building metadata labels"
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
        """
        Collect the datetime for each audio file based on its file name.

        The datetimes are stored in ``timestamp_per_file``.
        """
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
        """
        Calculate the time of day for each embedding.

        The results are stored in ``time_of_day_per_embedding``. If
        ``only_embed_annotations`` is set, the times are based on the start
        times of the annotations.
        """
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

    def week_of_year(self):
        """
        Calculate the week of the year for each embedding.

        The results are stored in ``week_of_year_per_embedding`` in the format
        ``year--week``.
        """
        self.get_datetimes()
        week_of_year_per_file = {}
        for file, datetime_of_file in tqdm(
            self.timestamp_per_file.items(), "getting week of year"
        ):
            date = datetime_of_file.date()
            week_of_day = (
                date.year, date.isocalendar().week
                )
            week_of_year_per_file.update({file: week_of_day})

        self.week_of_year_per_embedding = []
        for file_idx, (file, week_of_year) in enumerate(
            week_of_year_per_file.items()
        ):
            self.week_of_year_per_embedding.extend(
                np.repeat(
                    "--".join([str(a) for a in week_of_year]),
                    self.nr_embeds_per_file[file_idx],
                )
            )

    def day_of_year(self):
        """
        Calculate the day of the year for each embedding.

        The results are stored in ``day_of_year_per_embedding`` in the format
        ``YYYY-MM-DD``.
        """
        self.get_datetimes()
        day_of_year_per_file = {}
        for file, datetime_of_file in tqdm(
            self.timestamp_per_file.items(), "getting day of year"
        ):
            time_of_day = dt.datetime(
                datetime_of_file.year, datetime_of_file.month, datetime_of_file.day
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
        """
        Calculate a continuous timestamp for each embedding.

        The results are stored in ``continuous_timestamp_per_embedding``. If
        ``only_embed_annotations`` is set, the timestamps are based on the
        start times of the annotations.
        """
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
        """
        Store the parent directory of each audio file for each embedding.

        The results are stored in ``parent_directory_per_embedding``.
        """
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
        """
        Store the audio file name for each embedding.

        The results are stored in ``audio_file_name_per_embedding``.
        """
        self.audio_file_name_per_embedding = []
        for file_idx, file in tqdm(
            enumerate(self.metadata["files"]["audio_files"]),
            "getting audio file names",
        ):
            self.audio_file_name_per_embedding.extend(
                np.repeat(file, self.nr_embeds_per_file[file_idx])
            )

    def default_classifier(self):
        """
        Load the predictions of the default classifier for each embedding.

        The results are stored in ``default_classifier_per_embedding``. If no
        classifier annotations exist, the key is removed from
        ``default_label_keys``.
        """
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
        """
        Fill the embeddings without classifier annotations with a
        below-threshold label.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe with the predictions of the default classifier

        Returns
        -------
        pandas.DataFrame
            dataframe with the predictions, extended with a
            ``"below_thresh"`` label for the remaining embeddings

        Raises
        ------
        ValueError
            if the timestamps of the predictions do not match those of the
            generated embeddings
        AssertionError
            if the number of points does not match the total number of
            embeddings
        """
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
    """
    Create a function that generates model specific paths for the results of
    the embedding evaluation.

    The returned ``get_paths`` function creates the paths for the embeddings,
    labels, clustering, probing, predictions, and plots based on the audio
    directory and the model name.

    Parameters
    ----------
    audio_dir : str
        full path to the directory containing the audio files
    main_results_dir : str, optional
        top level directory for the results of the embedding evaluation,
        by default None
    dim_reduc_parent_dir : str, optional
        name of the folder containing the dimensionality reduced embeddings,
        by default "dim_reduced_embeddings"
    testing : bool, optional
        if True, no directories are created, by default False

    Returns
    -------
    get_paths : callable
        function that returns a SimpleNamespace with the model specific paths
    """
    global get_paths

    def get_paths(model_name):
        """
        Generate model specific paths for the results of the embedding evaluation.
        This includes paths for the embeddings, labels, clustering, classification,
        and plots. The paths are created based on the audio directory,
        and model name.

        Parameters
        ----------
        model_name : str
            name of the model used for embedding

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
    """
    Return the path to the dimensionality reduced embeddings of a model.

    Parameters
    ----------
    model_name : str
        name of the model used for embedding
    dim_reduction_model : str, optional
        name of the dimensionality reduction model, by default "umap"

    Returns
    -------
    Path
        path to the dimensionality reduced embeddings of the model
    """
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
    """
    Convert a path with windows separators to a posix path.

    Parameters
    ----------
    path : str
        path that may contain windows separators

    Returns
    -------
    str
        path converted to posix separators
    """
    if "\\" in path:
        from pathlib import PureWindowsPath

        return str(PureWindowsPath(path).as_posix())
    else:
        return str(path)


def load_metadata_file(folder):
    """
    Load the ``metadata.yml`` file of a folder and normalize its paths.

    Parameters
    ----------
    folder : Path
        path to the folder containing the ``metadata.yml`` file

    Returns
    -------
    dict
        dictionary with the metadata content

    Raises
    ------
    AssertionError
        if the metadata file does not contain any audio files
    """
    with open(folder.joinpath("metadata.yml"), "r") as f:
        metadata_dict = yaml.load(f, Loader=yaml.CLoader)

    metadata_dict["audio_dir"] = ensure_windoof_path_to_posix(
        metadata_dict["audio_dir"]
    )
    metadata_dict["embed_dir"] = ensure_windoof_path_to_posix(
        metadata_dict["embed_dir"]
    )
    if len(metadata_dict['files']['audio_files']) == 0:
        raise AssertionError(
            f"The metadata file {folder.joinpath('metadata.yml')} is empty. "
            f"Please manually remove the folder {folder}."
        )
    return metadata_dict


def get_metadata_labels(model_name, **kwargs):
    """
    Return dictionary of the metadata labels based on the files that were
    already processed and saved. This is model dependent, as the input length is
    model dependent and therefore this function requires a model name as input.
    The metadata labels are calculated based on the metadata labels specified in the
    settings.yaml file.

    Parameters
    ----------
    model_name : str
        model name

    Returns
    -------
    dict
        dictionary of metadata labels
    """
    paths = get_paths(model_name)
    return create_metadata_labels(paths.audio_dir, model_name, paths, **kwargs)


def get_ground_truth(model_name, file_path=None, return_type="dataframe"):
    """
    Return dictionary of the ground truth labels based on the files that were
    already processed and saved. This is model dependent, as the input length is
    model dependent and therefore this function requires a model name as input.

    Parameters
    ----------
    model_name : str
        model name
    file_path : str, optional
        path to a ground truth csv file, by default None
    return_type : str, optional
        return the ground truth as a dataframe or as a numpy array,
        either "dataframe" or "array", by default "dataframe"

    Returns
    -------
    pandas.DataFrame or dict
        dictionary or dataframe of ground truth labels
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
        from bacpipe.core.experiment_manager import return_reduced_dimensions
        embed_paths_for_this_model = [
            d
            for d in embed_paths_for_this_model
            if (
                dim_reduction_model in d.stem
                and return_reduced_dimensions(d) == bacpipe.settings.visualization_dimensions
                )
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


def create_metadata_labels(
    audio_dir=None, model=None, paths=None, 
    overwrite=True, return_type='dataframe', **kwargs
):
    """
    Create metadata labels based on audio files and model timestamps to
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
    return_type : string, optional
        return data as dict or dataframe, defaults to dataframe

    Returns
    -------
    dict
        dictionary with metadata labels
    """
    if paths is None:
        assign_global_get_paths_function(audio_dir)
        paths = get_paths(model)
    if (
        overwrite
        or (
            not (paths.labels_path / "metadata_labels.parquet").exists()
            and not (paths.labels_path / "metadata_labels.csv").exists()
            # these two are old versions of bacpipe that will 
            # still be supported until the next major version
            and not (paths.labels_path / "metadata_labels.npy").exists()
            and not (paths.labels_path / "default_labels.npy").exists()
            )
    ):
        if not kwargs.get("default_label_keys"):
            from bacpipe import settings as bacpipe_settings

            kwargs["default_label_keys"] = bacpipe_settings.default_label_keys
        metadata_labels = DefaultLabels(
            paths, model=model, audio_dir=audio_dir, **kwargs
        )
        metadata_labels.generate()

        df_labels = pd.DataFrame(metadata_labels.default_label_dict)
        input_length = (
            metadata_labels.metadata['segment_length (samples)']
            / metadata_labels.metadata['sample_rate (Hz)']
            )
        if not bacpipe.settings.only_embed_annotations:
            start = []
            [
                start.extend([embed_idx * input_length for embed_idx in 
                np.arange(nr_of_embeds)])
                for nr_of_embeds in 
                metadata_labels.nr_embeds_per_file
            ]
            df_labels['start'] = start
            df_labels['end'] = df_labels['start'] + input_length
        else:
            df_gt = ground_truth_by_model(
                model, 
                audio_dir, 
                annotations_filename=bacpipe.settings.annotations_filename, 
                only_embed_annotations=True, 
                overwrite=False
                )
            df_labels['start'] = df_gt['start']
            df_labels['end'] = df_gt['end']
        
        if len(df_labels) * len(df_labels.T) > 3_000_000:
            df_labels.to_parquet(paths.labels_path / "metadata_labels.parquet", index=False)
        else:
            df_labels.to_csv(paths.labels_path / "metadata_labels.csv", index=False)

        def_labels = df_labels.to_dict('list')
    else:
        if (paths.labels_path / "metadata_labels.parquet").exists():
            df_labels  = pd.read_parquet(paths.labels_path / "metadata_labels.parquet", index_col=False)
            def_labels = df_labels.to_dict('list')
            
        elif (paths.labels_path / "metadata_labels.csv").exists():
            df_labels  = pd.read_csv(paths.labels_path / "metadata_labels.csv", index_col=False)
            def_labels = df_labels.to_dict('list')
            
        elif paths.labels_path.joinpath("metadata_labels.npy").exists():
            def_labels = np.load(
                paths.labels_path.joinpath("metadata_labels.npy"), allow_pickle=True
            ).item()
            df_labels = pd.DataFrame(def_labels)
            
        elif paths.labels_path.joinpath("default_labels.npy").exists():
            def_labels = np.load(
                paths.labels_path.joinpath("default_labels.npy"), allow_pickle=True
            ).item()
            df_labels = pd.DataFrame(def_labels)
    if return_type == 'dict':
        return def_labels
    elif return_type == 'dataframe':
        return df_labels

def fetch_annotation_file(audio_dir, annotations_filename, paths):
    """
    Fetch the annotations file from the audio directory or the dataset
    directory.

    Parameters
    ----------
    audio_dir : str
        full path to the directory containing the audio files
    annotations_filename : str
        name of the annotations file
    paths : SimpleNamespace
        convenient object for path handling

    Returns
    -------
    pandas.DataFrame
        dataframe with the annotations

    Raises
    ------
    FileNotFoundError
        if no annotations file could be found
    """
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
    """
    Filter labels that occur fewer times than a minimum number of
    occurrences.

    Parameters
    ----------
    label_df : pandas.DataFrame
        dataframe with the annotations
    main_label_column : str
        column name of the main label
    min_label_occurrences : int
        minimum number of occurrences required to keep a label
    bool_filter_labels : bool
        if True, the labels are filtered

    Returns
    -------
    pandas.DataFrame
        dataframe with only the labels that occur at least the minimum
        number of times, or None if no labels remain after filtering
    """
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
    """
    Load the annotations file and optionally filter the labels and the
    audio files.

    Parameters
    ----------
    paths : SimpleNamespace
        convenient object for path handling
    annotations_filename : str
        name of the annotations file
    audio_dir : str
        full path to the directory containing the audio files
    audio_files : list, optional
        list of audio files to keep, by default []
    bool_filter_labels : bool, optional
        if True, the labels are filtered by their number of occurrences,
        by default True
    min_label_occurrences : int, optional
        minimum number of occurrences required to keep a label,
        by default 150
    main_label_column : str, optional
        column name of the main label, by default None
    testing : bool, optional
        if True, no label filtering is applied, by default False

    Returns
    -------
    pandas.DataFrame
        dataframe with the annotations, optionally filtered
    """
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
    only_embed_annotations=False,
    **kwargs,
):
    """
    Fit the annotations of a single file onto the timestamps of the
    embeddings.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with the annotations of a single audio file
    df_fitted_gt : pandas.DataFrame
        empty ground truth dataframe to be filled
    num_embeds : int
        number of embeddings for the audio file
    segment_s : float
        length of one segment in seconds
    label_column : str, optional
        column name of the label, by default None
    only_embed_annotations : bool, optional
        if True, only the annotated parts of the audio files are embedded,
        by default False

    Returns
    -------
    pandas.DataFrame
        ground truth dataframe fitted to the embedding timestamps
    """
    for col in df_fitted_gt.columns:
        df_fitted_gt[col] = np.zeros(num_embeds, dtype=np.int8)
    df = df.sort_values("start")

    if not only_embed_annotations:
        df_fitted_gt["start"] = np.arange(num_embeds) * segment_s
        df_fitted_gt["end"] = df_fitted_gt["start"] + segment_s
    else:
        df_fitted_gt["start"] = df["start"].values
        df_fitted_gt["end"] = df["end"].values

    df.index = range(len(df))
    for _, row in df.iterrows():
        start_at_embed_nr = np.where(
            df_fitted_gt["start"] - row.start <= 0
            )[0][-1]
        end_at_embed_nr = np.where(df_fitted_gt["start"] - row.end >= 0)[0]
        if len(end_at_embed_nr) > 0:
            end_at_embed_nr = end_at_embed_nr[0]
        else:
            end_at_embed_nr = len(df_fitted_gt["start"])
        for idx in range(start_at_embed_nr, end_at_embed_nr):

            # check if the annotation length is longer that the specified min_annotation_length
            if (row.end - row.start > bacpipe.settings.min_annotation_length):
                df_fitted_gt.loc[idx, row[f"label:{label_column}"]] = 1
            else:
                logger.info(
                    f"\nSkipping annotation from {row.start} to {row.end} with "
                    f"label {row['label:species']} because the annotation is "
                    f"shorter than {bacpipe.settings.min_annotation_length=}. To change this, "
                    "modify the value in the settings file."
                )
                
    df_fitted_gt["simultaneous_labels"] = df_fitted_gt.drop(
        columns=["start", "end", "audiofilename", "simultaneous_labels"]
        ).sum(axis=1)
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
    filename_array=None,
    only_embed_annotations=False,
    **kwargs,
):
    """
    Build the ground truth labels for a single audio file and add them to
    the ground truth labels of all files.

    Parameters
    ----------
    ind : int
        index of the audio file in the metadata
    model : str
        model name
    num_embeds : int
        number of embeddings for the audio file
    segment_s : float
        length of one segment in seconds
    metadata : dict
        dictionary with the metadata content
    all_labels : pandas.DataFrame
        dataframe with the ground truth labels collected so far
    label_df : pandas.DataFrame, optional
        dataframe with the annotations, by default None
    label_column : str, optional
        column name of the label, by default None
    filename_array : numpy.ndarray, optional
        array with the file names, by default None
    only_embed_annotations : bool, optional
        if True, only the annotated parts of the audio files are embedded,
        by default False

    Returns
    -------
    pandas.DataFrame
        dataframe with the ground truth labels including the new file
    """
    audio_file = metadata["files"]["audio_files"][ind]
    df = filter_df_by_filename(label_df, audio_file, filename_array=filename_array, model=model)
    if len(df) == 0:
        logger.info(
            f"\nNo annotations found for {audio_file=}. "
            "Continuing with next file."
        )
    else:
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
        if file_labels["simultaneous_labels"].max() == 0:
            logger.warning(
                "The simultaneous labels column of the ground truth has a "
                "maximum value of 0 for annotations corresponding to "
                f"{audio_file=}. This means no annotations have been"
                "found for your data. Something failed in building the "
                "ground truth array. Please ensure the audio filenames "
                "match the names in the names in the annotations file."
            )
        file_labels["audiofilename"] = audio_file
        all_labels = pd.concat([all_labels, file_labels])
    return all_labels


        
        
def filter_df_by_filename(
    df_to_filter, file_name, filename_array=None,
    file_name_column="audiofilename", model=None
):
    """
    Filter a dataframe by the file name of an audio file.

    Multiple matching strategies are applied so that paths, stems, and
    classifier prediction files are matched as well.

    Parameters
    ----------
    df_to_filter : pandas.DataFrame
        dataframe to be filtered
    file_name : str
        name of the audio file
    filename_array : numpy.ndarray, optional
        array with the file names including the file extension,
        by default None
    file_name_column : str, optional
        column name of the file names, by default "audiofilename"
    model : str, optional
        model name, by default None

    Returns
    -------
    pandas.DataFrame
        filtered dataframe
    """
    if filename_array is None:
        filename_array = get_filename_array(df_to_filter, file_name_column)
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
        df = df_to_filter[filename_array == file_name]
        
    # if no files are found, ensure parent path is not the cause
    if len(df) == 0:
        df = df_to_filter[
            filename_array 
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
    """
    Create a Raven annotation table from a dataframe with annotations.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with the annotations
    label_column : str
        column name of the label
    high_freq : int, optional
        high frequency of the Raven table in Hz, by default 1000

    Returns
    -------
    pandas.DataFrame
        dataframe formatted as a Raven annotation table
    """
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
    """
    Ensure that the name of an embedding file matches the name of the
    corresponding audio file in the metadata.

    Parameters
    ----------
    metadata : dict
        dictionary with the metadata content
    ind : int
        index of the audio file in the metadata
    file : Path
        path to the embedding file
    model : str
        model name

    Raises
    ------
    AssertionError
        if the file names do not match
    """
    assert (
        Path(metadata["files"]["audio_files"][ind]).stem
        == file.stem.split(f"_{model}")[0]
    ), (
        f"File names do not match for {file} and "
        f"{metadata['files']['audio_files'][ind]}"
    )

def initialize_ground_truth_df(label_df, label_column):
    """
    Initialize an empty ground truth dataframe based on the labels in the
    annotations file.

    Parameters
    ----------
    label_df : pandas.DataFrame
        dataframe with the annotations
    label_column : str
        column name of the label

    Returns
    -------
    pandas.DataFrame
        empty dataframe with one column per species, a column for the
        simultaneous labels, the audio file name, and the start and end times
    """
    # Get species names
    species_cols = label_df[f"label:{label_column}"].unique().tolist()

    # This ensures all species columns and numeric columns are floats from the start
    return pd.DataFrame(
        {
            **{col: pd.Series(dtype="int8") for col in species_cols},
            "simultaneous_labels": pd.Series(dtype="int8"),
            "audiofilename": pd.Series(
                dtype="string"
            ),
            "end": pd.Series(dtype="int8"),
            "start": pd.Series(dtype="int8"),
        }
    )

def get_filename_array(label_df, label_column):
    """
    Return an array with the file names of the annotations including the
    file extension.

    Parameters
    ----------
    label_df : pandas.DataFrame
        dataframe with the annotations
    label_column : str
        column name of the file names

    Returns
    -------
    numpy.ndarray
        array with the file names
    """
    return np.array([
        Path(f).stem + Path(f).suffix for f in label_df[label_column]
        ]) 

def collect_ground_truth_labels(
    files,
    model,
    segment_s,
    metadata,
    label_df,
    label_column,
    **kwargs,
):
    """
    Collect the ground truth labels for all audio files and fit them to the
    embedding timestamps.

    Parameters
    ----------
    files : list
        list of embedding files
    model : str
        model name
    segment_s : float
        length of one segment in seconds
    metadata : dict
        dictionary with the metadata content
    label_df : pandas.DataFrame
        dataframe with the annotations
    label_column : str
        column name of the label

    Returns
    -------
    pandas.DataFrame
        dataframe with the ground truth labels
    """
    ground_truth = initialize_ground_truth_df(label_df, label_column)
    filename_array = get_filename_array(label_df, 'audiofilename')
    
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
            filename_array=filename_array,
            label_column=label_column,
            **kwargs,
        )
    
    if ground_truth["simultaneous_labels"].max() > 1:
        logger.warning(
            "The simultaneous labels column of the ground truth has "
            "values exceeding 1. This means you have multi-label "
            "ground truth annotations. If this should not be "
            "happening ensure the ground truth is created correcly."
        )
    return ground_truth


def assign_global_get_paths_function(audio_dir):
    """
    Assign the global ``get_paths`` function based on the audio directory.

    Parameters
    ----------
    audio_dir : str
        full path to the directory containing the audio files
    """
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
            logger.warning(f"No embeddings directory seems to exist. {str(e)}")
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
                by=["audiofilename", "start"]
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
    """
    Ensure that the annotated audio files are present in the found audio
    files by trying multiple matching strategies.

    Parameters
    ----------
    found_audio_files : list
        list of audio files found in the audio directory
    annotated_audio_files : list
        list of audio files that are annotated
    audio_dir : str
        full path to the directory containing the audio files

    Returns
    -------
    list
        list of the found audio files
    """
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
    """
    Get the files and metadata for a model when no embeddings exist yet.

    Parameters
    ----------
    audio_dir : str
        full path to the directory containing the audio files
    model : str
        model name
    label_df : pandas.DataFrame, optional
        dataframe with the annotations, by default None
    only_embed_annotations : bool, optional
        if True, only the annotated parts of the audio files are embedded,
        by default False

    Returns
    -------
    files : list
        list of embedding file names
    segment_s : float
        length of one segment in seconds
    metadata : dict
        dictionary with the metadata content
    """
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
            int(np.ceil(get_duration(path=f) / segment_s)) for f in matching_audio_files
        ]
    files = [Path(f"{Path(d).stem}_{model}") for d in matching_audio_files]

    return files, segment_s, metadata
