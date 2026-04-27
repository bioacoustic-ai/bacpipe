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
        if kwargs.get('only_embed_annotations'):
            self.only_embed_annotations = True
            self.df, _ = load_labels_and_build_dict(
                paths, 
                kwargs.get('annotations_filename'), 
                self.paths.audio_dir,
                bool_filter_labels=False
            )
        
        if (self.paths.preds_path / "original_classifier_outputs").exists():
            if not "default_classifier" in self.default_label_keys:
                self.default_label_keys += ["default_classifier"]
        elif "default_classifier" in self.default_label_keys:
            self.default_label_keys.remove("default_classifier")
        
        try:
            embed_path = model_specific_embedding_path(paths.main_embeds_path, model)
            self.metadata = yaml.safe_load(open(embed_path.joinpath("metadata.yml"), "r"))
            self.nr_embeds_per_file = self.metadata["files"]["nr_embeds_per_file"]
            self.nr_embeds_total = self.metadata["nr_embeds_total"]
        except ValueError as e:
            logger.info(
                "No embeddings found. Gathering files and nr of embeddings "
                "per file from audio files."
            )
            _, _, metadata = get_files_if_no_embeds(paths.audio_dir, model)
            self.metadata = metadata
            self.nr_embeds_per_file = metadata['files']['nr_embeds_per_file']
            self.nr_embeds_total = sum(
                metadata['files']['nr_embeds_per_file']
            )
        if not sum(self.nr_embeds_per_file) == self.nr_embeds_total:
            error = (
                "\nThe number of embeddings per file does not match "
                "the total number of embeddings.")
            logger.exception(error)
            raise ValueError(error)

    def generate(self):
        self.default_label_dict = {}
        for default_label in self.default_label_keys:
            getattr(self, default_label)()

            if hasattr(self, f"{default_label}_per_embedding"):
                self.default_label_dict.update(
                    {default_label: getattr(self, f"{default_label}_per_embedding")}
                )

    def get_datetimes(self):
        if not hasattr(self, "timestamp_per_file"):
            self.timestamp_per_file = {}
            for file in self.metadata["files"]["audio_files"]:
                file_stem = Path(file).stem
                self.timestamp_per_file.update({file: get_dt_filename(file_stem)})

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
                
                if hasattr(self, 'only_embed_annotations') and getattr(self, 'only_embed_annotations'):
                    timestamp = (
                        (time_of_day + dt.timedelta(seconds=self.df.start[index_of_embedding]))
                        .time()
                        .replace(microsecond=0)
                    )
                else:
                    timestamp = (
                        (time_of_day + index_of_embedding * segment_s_dt)
                        .time()
                        .replace(microsecond=0)
                    )
                self.time_of_day_per_embedding.append(timestamp.strftime("%H-%M-%S"))

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
                np.repeat(
                    day_of_year.strftime("%Y-%m-%d"), self.nr_embeds_per_file[file_idx]
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
        for file_idx, (file, datetime_per_file) in enumerate(
            self.timestamp_per_file.items()
        ):
            for index_of_embedding in range(self.nr_embeds_per_file[file_idx]):
                
                if hasattr(self, 'only_embed_annotations') and getattr(self, 'only_embed_annotations'):
                    timestamp = (
                        (datetime_per_file + dt.timedelta(seconds=self.df.start[index_of_embedding]))
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

    def default_classifier(self):
        clfier_paths = list(self.paths.preds_path.rglob("*_classifier_annotations.csv"))
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
        seg_len = self.metadata['segment_length (samples)'] / self.metadata['sample_rate (Hz)']
        df_new = {
            'start': [],
            'end': [],
            'audiofilename': [],
            'label:default_classifier': []
        }
        for file, nr_embeds in zip(
            self.metadata['files']['audio_files'], 
            self.metadata['files']['nr_embeds_per_file']
            ):
            df_part = df[df.audiofilename == file]
            if hasattr(self, 'only_embed_annotations') and getattr(self, 'only_embed_annotations'):
                all_time_bins = np.round(self.df.start.values, 4).tolist()
            else:
                all_time_bins = np.round(np.arange(nr_embeds) * seg_len, 4).tolist()
            [all_time_bins.remove(l) for l in np.round(df_part.start, 4)]
            df_new['start'].extend(all_time_bins)
            df_new['end'].extend((np.array(all_time_bins) + seg_len).tolist())
            df_new['audiofilename'].extend([file] * len(all_time_bins))
            df_new['label:default_classifier'].extend(['below_thresh'] * len(all_time_bins))
            
        df = pd.concat([df, pd.DataFrame(df_new)], ignore_index=True)
        if not len(df) == self.metadata['nr_embeds_total']:
            raise AssertionError(
                "The number of points does not match the total number of embeddings."
            )
        return df.sort_values(['audiofilename', 'start'])

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
        dataset_path = Path(main_results_dir).joinpath(Path(audio_dir).parts[-1])
        task_path = dataset_path.joinpath("evaluations").joinpath(model_name)

        paths = {
            "audio_dir": audio_dir,
            "dataset_path": dataset_path,
            "dim_reduc_parent_dir": dataset_path.joinpath(dim_reduc_parent_dir),
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


def get_ground_truth(model_name):
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

def model_specific_embedding_path(path, model, dim_reduction_model=None, **kwargs):
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
            d for d in embed_paths_for_this_model if dim_reduction_model in d.stem
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
    if overwrite or not paths.labels_path.joinpath("default_labels.npy").exists():
        if not kwargs.get('default_label_keys'):
            from bacpipe import settings as bacpipe_settings
            kwargs['default_label_keys'] = bacpipe_settings.default_label_keys
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


def load_labels_and_build_dict(
    paths,
    annotations_filename,
    audio_dir,
    bool_filter_labels=True,
    min_label_occurrences=150,
    main_label_column=None,
    testing=False,
    **kwargs,
):
    try:
        try:
            label_df = pd.read_csv(Path(audio_dir).joinpath(annotations_filename))
        except FileNotFoundError as e:
            label_df = pd.read_csv(list(Path(audio_dir).rglob(annotations_filename))[0])
    except FileNotFoundError as e:
        logger.warning(
            f"No annotations file found in {audio_dir}, trying in "
            f"{str(paths.dataset_path.resolve())}."
        )
        try:
            label_df = pd.read_csv(paths.dataset_path.joinpath(annotations_filename))
        except:
            logger.warning(
                "No annotations file found, not able to create ground_truth.npy file. "
                "bacpipe should still work, but you will not be able to label by ground truth. "
                "You also will not be able to evaluate using classification."
            )
            raise FileNotFoundError("No annotations file found.")
    if bool_filter_labels and not testing:
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
            label_df = label_df[label_df[main_label_column].isin(filtered_labels)]
    label_idx_dict = {}
    for label_column in [l for l in label_df.columns if 'label:' in l]:
        label_idx_dict[label_column] = {
            label: idx
            for idx, label in enumerate(label_df[label_column].unique())
        }
    if paths.labels_path.exists():
        with open(paths.labels_path.joinpath("label_idx_dict.json"), "w") as f:
            json.dump(label_idx_dict, f, indent=1)
    return label_df, label_idx_dict


def fit_labels_to_embedding_timestamps(
    df,
    label_idx_dict,
    num_embeds,
    segment_s,
    label_column=None,
    single_label=True,
    min_annotation_length=0.65,
    **kwargs,
):
    file_labels = np.ones(num_embeds) * -1
    embed_timestamps = np.arange(num_embeds) * segment_s
    if single_label:
        single_label_arr = [True] * len(embed_timestamps)
    else:
        file_labels = file_labels.reshape([len(file_labels), 1])

    for _, row in df.iterrows():
        em_start = np.where(embed_timestamps - row.start <= 0)[0][-1]
        em_end = np.where(embed_timestamps - row.end >= 0)[0]
        if len(em_end) > 0:
            em_end = em_end[0]
        else:
            em_end = len(embed_timestamps)
            
        # if not all of the values are noise, meaning there are already
        # some labels in this segment
        if not np.all(file_labels[em_start:em_end] == -1):
            if single_label:
                single_label_arr[em_start:em_end] = [False] * (em_end - em_start)
            else:
                for idx in range(em_start, em_end):
                    
                    # if there is any noise in this segment, we'll write into 
                    # those places
                    if np.any(file_labels[idx:idx+1] == -1):
                        file_labels[idx:idx+1][
                                file_labels[idx:idx+1]==-1
                                ] = label_idx_dict[row[f"label:{label_column}"]]
                    
                    # if the current label is already written in that segment
                    # skip. This assumes that we don't have two annotations of the 
                    # same class of varying length overlaying each other
                    elif (
                        label_idx_dict[row[f"label:{label_column}"]]
                        in file_labels[idx:idx+1]
                        ):
                        continue
                    
                    # if all labels in this segment are the same. meaning that
                    # if we have a 2d array already but for each timestamp the
                    # labels are the same, we can just overwrite one. we won't 
                    # loose anything and we don't have to create any new columns
                    elif len(np.unique(file_labels[idx:idx+1])) == 1:
                        file_labels[idx:idx+1, -1] = label_idx_dict[row[f"label:{label_column}"]]
                        
                    # We only go here, if there is no place we can write our new 
                    # class into the array without loosing information. we therefore
                    # create a new column, which is created with noise (-1) values and
                    # we then write our current label index into that column
                    else:
                        new_column = np.ones(len(file_labels)) * -1
                        new_column = new_column.reshape([len(file_labels), 1])
                        file_labels = np.hstack([file_labels, new_column])
                        file_labels[idx:idx+1, -1] = label_idx_dict[row[f"label:{label_column}"]]

        # check if the annotation length is longer that the specified min_annotation_length
        elif row.end - row.start > min_annotation_length:
            if single_label:
                file_labels[em_start:em_end] = label_idx_dict[row[f"label:{label_column}"]]
            else:
                file_labels[em_start:em_end, 0] = label_idx_dict[row[f"label:{label_column}"]]
                
    if len(file_labels.shape) > 1 and (
        file_labels.shape[0] > 1 or file_labels.shape[-1] > 1
        ):
        file_labels = file_labels.squeeze()
            
    if single_label:
        file_labels[~np.array(single_label_arr)] = -2
        return file_labels
    else:
        if len(file_labels.shape) == 1:
            array = np.ones([len(file_labels), 2]) * -1
            array[:, 0] = file_labels
            return array
        else:
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
    label_column=None,
    **kwargs,
):

    audio_file = metadata["files"]["audio_files"][ind]
    df = filter_df_by_filename(label_df, audio_file, model=model)
        
    if df.empty:
        logger.info(
            f'df is empty for {audio_file}, meaning no annotations. '
            "If that's incorrect, ensure the audiofilename column has the correct "
            "file names."
            )
        label_dimensions = all_labels.shape[-1] if len(all_labels.shape) > 1 else 1
        all_labels = np.concatenate(
            (all_labels, np.ones([num_embeds, label_dimensions]).squeeze() * -1)
            )
        return all_labels

    if kwargs.get('only_embed_annotations'):
        values = df[f'label:{label_column}']
        file_labels = np.array([label_idx_dict[v] for v in values])
    else:
        file_labels = fit_labels_to_embedding_timestamps(
            df, label_idx_dict, num_embeds, segment_s, 
            label_column=label_column, **kwargs
        )
    
        
    
    all_labels = fill_all_labels_array(file_labels, all_labels)

    if np.unique(file_labels).shape[0] > 2 and kwargs.get('testing'):
        raven_tables_sanity_check(
            num_embeds, segment_s, paths, audio_file, 
            label_df, label_idx_dict, label_column, file_labels
        )
    return all_labels

def filter_df_by_filename(
    df_to_filer, file_name, file_name_column = 'audiofilename', model=None
    ):
    df = df_to_filer[df_to_filer[file_name_column] == Path(file_name).as_posix()]
    if len(df) == 0:
        df = df_to_filer[
            df_to_filer[file_name_column] == (
                Path(file_name).stem + Path(file_name).suffix
                )
        ]
    # if no files are found, match by classifier_prediction files
    if len(df) == 0:
            df = df_to_filer[
                df_to_filer[file_name_column]
                == Path(file_name).parent / (
                    Path(file_name).stem + f'_{model}.json'
                    )
            ]
    return df

def fill_all_labels_array(file_labels, all_labels):
    if len(file_labels.shape) > 1:
        if len(all_labels) == 0:
            all_labels = file_labels
        else:
            # if file_labels got columns that all_labels don't have
            # add noise columns to ensure both can be stacked
            if all_labels.shape[-1] < file_labels.shape[-1]:
                new_column = np.ones([
                        len(all_labels), 
                        file_labels.shape[-1] - all_labels.shape[-1]
                    ]) * -1
                all_labels = np.hstack([all_labels, new_column])
            # if all_labels got columns that file_labels don't have
            # add noise columns to ensure both can be stacked
            elif all_labels.shape[-1] > file_labels.shape[-1]:
                new_column = np.ones([
                    len(file_labels), 
                    all_labels.shape[-1] - file_labels.shape[-1]
                    ]) * -1
                file_labels = np.hstack([file_labels, new_column])
                
            all_labels = np.concatenate((all_labels, file_labels))
    else:
        all_labels = np.concatenate((all_labels, file_labels))
    return all_labels

def raven_tables_sanity_check(
    num_embeds, segment_s, paths, audio_file, 
    label_df, label_idx_dict, label_column, file_labels
    ):
    if len(file_labels.shape) > 1:
        file_labels = file_labels[:, 0]
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
        df_file_fit[f"label:{label_column}"] = [
            inv[i] for i in file_labels[file_labels > -1]
        ]
        raven_gt = create_Raven_annotation_table(df_file_gt, label_column)
        raven_fit = create_Raven_annotation_table(df_file_fit, label_column)
        raven_fit["Low Freq (Hz)"] = 1500
        raven_fit["High Freq (Hz)"] = 2000
        raven_gt.to_csv(
            path.joinpath(f"{Path(audio_file).stem}_gt.txt"), sep="\t", index=False
        )
        raven_fit.to_csv(
            path.joinpath(f"{Path(audio_file).stem}_fit.txt"), sep="\t", index=False
        )

def create_Raven_annotation_table(df, label_column, high_freq=1000):
    df.index = np.arange(1, len(df) + 1)
    raven_df = pd.DataFrame()
    raven_df["Selection"] = df.index
    raven_df.index = np.arange(1, len(df) + 1)
    raven_df["View"] = 'Spectrogram 1'
    raven_df["Channel"] = 1
    raven_df["Begin Time (s)"] = df.start
    raven_df["End Time (s)"] = df.end
    raven_df["Low Freq (Hz)"] = 0
    raven_df["High Freq (Hz)"] = high_freq
    raven_df["Label"] = df[f"label:{label_column}"]
    return raven_df


def collect_ground_truth_labels(
    paths, files, model, segment_s, metadata, 
    label_df, label_idx_dict, **kwargs
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
        if kwargs.get('only_embed_annotations'):
            num_embeds = int(
                metadata["files"]["file_lengths (s)"][ind] 
                / (metadata['segment_length (samples)'] / metadata['sample_rate (Hz)'])
                )
        else:
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

def assign_global_get_paths_function(audio_dir):
    if not 'get_paths' in globals():
        from bacpipe import settings as bapcipe_settings
        make_set_paths_func(
            audio_dir, bapcipe_settings.main_results_dir
            )

def ground_truth_by_model(
    model,
    audio_dir,
    label_df=None,
    label_idx_dict=None,
    label_column='label:species',
    paths=None,
    annotations_filename="annotations.csv",
    overwrite=True,
    single_label=True,
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
    label_idx_dict : dict, optional
        link between int values and class labels
        can be auto generated, by default None
    label_column : str, optional
        name of column in annotation file, by default 'label:species'
    paths : SimpleNamespace, optional
        convenient object for path handling, by default None
    annotations_filename : str, optional
        path to annotations csv file, by default "annotations.csv"
    overwrite : bool, optional
        If True, the dict will be generated again and saved
        rather than loaded from a file if already
        processed, by default True
    single_label : bool, optional
        set False if you want multi-label, by default True
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
        or not paths.labels_path.joinpath("ground_truth.npy").exists()
        ):

        # check if embeddings exist
        try:    
            path = model_specific_embedding_path(paths.main_embeds_path, model)
        except Exception as e:
            logger.warning(
                f"No embeddings directory seems to exist. {e}"
            )
            path = None

        # get annotations is not provided
        if label_df is None or label_idx_dict is None:
            if not 'label:' in label_column:
                label_column = 'label:' + label_column
            if kwargs.get('testing'):
                annotations_filename='annotations.csv'
            label_df, label_idx_dict = load_labels_and_build_dict(
                paths, annotations_filename, 
                main_label_column=label_column, 
                audio_dir=audio_dir, 
                bool_filter_labels=bool_filter_labels,
                **kwargs
            )

        # build files, segment_s and metadata variables
        # depending if embeddings exist or not
        if path is not None and len(list(path.iterdir())) > 0:
            files = list(path.rglob("*.npy"))
            files.sort()

            metadata = yaml.safe_load(
                open(list(path.rglob("metadata.yml"))[0], "r")
                )
            segment_s = (
                metadata["segment_length (samples)"] 
                / metadata["sample_rate (Hz)"]
                )
        else:
            files, segment_s, metadata = get_files_if_no_embeds(
                audio_dir, model, label_df
                )
            
        # find all label columns
        label_columns = [col for col in label_df.columns if "label:" in col]
        ground_truth_dict = {}
        
        # collect all the ground truth for all the label columns 
        for label_col in label_columns:
            labels = label_col.split("label:")[-1]
            ground_truth = collect_ground_truth_labels(
                paths,
                files,
                model,
                segment_s,
                metadata,
                label_df,
                label_idx_dict[label_col],
                single_label=single_label,
                label_column=labels,
                **kwargs,
            )

            ground_truth_dict.update({
                f"label:{labels}": ground_truth,
                f"label_dict:{labels}": label_idx_dict[label_col],
            })
        np.save(paths.labels_path.joinpath("ground_truth.npy"), ground_truth_dict)
    else:
        if not paths.labels_path.joinpath("ground_truth.npy").exists():
            error = (
                "\nThe ground truth label file ground_truth.npy does not exist. "
                "Please create it first by rerunning with `overwrite=True`."
            )
            logger.exception(error)
            raise ValueError(error)
        ground_truth_dict = np.load(
            paths.labels_path.joinpath("ground_truth.npy"), allow_pickle=True
        ).item()
    return ground_truth_dict

def ensure_audio_files(found_audio_files, annotated_audio_files, audio_dir):
    if not annotated_audio_files:
        return found_audio_files
    matching = set(found_audio_files).intersection(set(annotated_audio_files))
    if len(matching) < len(annotated_audio_files) or len(matching) == 0:
        relative_to_audio_dir = [
            Path(f).relative_to(audio_dir) for f in found_audio_files
        ]
        matching = set(relative_to_audio_dir).intersection(set(annotated_audio_files))
        
    if len(matching) < len(annotated_audio_files) or len(matching) == 0:
        annotated_stems = [
            Path(f).stem for f in annotated_audio_files
        ]
        found_stems = [
            Path(f).stem for f in found_audio_files
        ]
        matching = set(annotated_stems).intersection(set(found_stems))
        
    # TODO maybe a case where they are nested but have duplicate filenames
    
    if len(matching) < len(annotated_audio_files) or len(matching) == 0:
        not_found = []
        found_annotated_audio_files = [
            list(Path(audio_dir).rglob(f'*{f.stem + f.suffix}'))[0]
            if list(Path(audio_dir).rglob(f'*{f.stem + f.suffix}'))
            else not_found.append(f)
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
    

def get_files_if_no_embeds(audio_dir, model, label_df=None):
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
    metadata['files'] = {}
    from bacpipe import get_audio_files
    found_audio_files = get_audio_files(audio_dir)
    matching_audio_files = ensure_audio_files(
        found_audio_files, annotated_audio_files, audio_dir
        )
    matching_audio_files.sort()

    metadata["segment_length (samples)"] = module.LENGTH_IN_SAMPLES
    metadata["sample_rate (Hz)"] = module.SAMPLE_RATE
    metadata['files']['audio_files'] = matching_audio_files
    metadata['files']['nr_embeds_per_file'] = [
        int(
            get_duration(path=f) / segment_s 
        )
        for f in matching_audio_files
    ]
    files = [Path(f'{Path(d).stem}_{model}') for d in matching_audio_files]
    
    return files, segment_s, metadata
