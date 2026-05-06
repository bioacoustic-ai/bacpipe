from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path

import logging

logger = logging.getLogger("bacpipe")


class ProbeDatasetLoader(Dataset):
    def __init__(self, class_df, embeds, label2index, set_name=None, **kwargs):
        """
        Class to initialize and iterate through classification dataset.

        Parameters
        ----------
        class_df : pd.DataFrame
            classification dataframe
        embeds : np.array
            embeddings
        label2index : dict
            linking labels to integers
        set_name : string, optional
            train, test or val set, by default None
        """
        if set_name is not None:
            self.dataset = class_df[class_df.predefined_set == set_name]
        else:
            self.dataset = class_df

        logger.info(
            f"Found {len(self.dataset)} samples in the {set_name} set with "
            f"{len(self.dataset.label.unique())} unique labels."
        )
        self.embeds = embeds

        self.label2index = label2index

        self.dataset = self.dataset.sample(frac=1, random_state=42)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Iterate through dataset.

        Parameters
        ----------
        idx : int
            index of training step

        Returns
        -------
        tuple
            (embedding, true label)
        """
        X = self.embeds[self.dataset.index[idx]]
        X = X.reshape(1, -1)

        if X.shape[0] > 1:
            X = np.mean(X, axis=0)
        else:
            X = X.flatten()
        y = self.label2index[self.dataset.label.values[idx]]

        return X, y



def probe_dataset_loader(
    set_name, clean_df, embeds, label2index, batch_size=64, shuffle=False, **kwargs
):
    """
    Create dataset loader object for classification.

    Parameters
    ----------
    set_name : string
        train, test of val set
    clean_df : pd.DataFrame
        classification dataframe
    embeds : np.array
        embeddings
    label2index : dict
        link labels to ints
    batch_size : int, optional
        number of embeddings per batch, by default 64
    shuffle : bool, optional
        shuffle or not, by default False

    Returns
    -------
    DataLoader obj
        dataset loader object to iterate over during training
    """
    loader = ProbeDatasetLoader(
        class_df=clean_df,
        set_name=set_name,
        embeds=embeds,
        label2index=label2index,
        **kwargs,
    )

    loader_generator = DataLoader(
        loader, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
    return loader_generator

def generate_annotations_for_probing_task(
    ground_truth, paths, label_column, 
    dataset_csv_path='probe_annotations.csv', 
    train_ratio=None, test_ratio=None, **kwargs
    ):
    import bacpipe
    if train_ratio is None:
        train_ratio = bacpipe.settings.probe_configs['config_1']['train_ratio']
    if test_ratio is None:
        test_ratio = bacpipe.settings.probe_configs['config_1']['test_ratio']
    
    if (
        paths is None
        or not Path(dataset_csv_path).exists()
        or not paths.labels_path.joinpath(dataset_csv_path).exists()
        ):
        inv = {v: k for k, v in ground_truth[f"label_dict:{label_column}"].items()}
        labels = ground_truth[f"label:{label_column}"][
            ground_truth[f"label:{label_column}"] > -1
        ]
        labs = [inv[i] for i in labels]
        df = pd.DataFrame()
        
        if not paths is None:
            filenames, starts = get_filenames_and_starts_for_probe_df(paths, ground_truth, label_column)
            df["filenames"] = filenames 
            df["starts"] = starts
            
        df["label"] = labs
        df["predefined_set"] = "undefined"
        for v in inv.values():
            l = labs.count(v)
            ar = list(df[df.label == v].index)
            np.random.shuffle(ar) # TODO have seed option
            tr_ar = ar[: int(l * train_ratio)] 
            te_ar = ar[int(l * train_ratio) : int(l * (train_ratio + test_ratio))]
            va_ar = ar[int(l * (train_ratio + test_ratio)) :]
            if not all([tr_ar, te_ar, va_ar]):
                continue
            df.loc[tr_ar, "predefined_set"] = "train"
            df.loc[te_ar, "predefined_set"] = "test"
            df.loc[va_ar, "predefined_set"] = "val"
        df = df[df.predefined_set.isin(["train", "val", "test"])]
        
        if paths is None:
            df.to_csv(dataset_csv_path, index=False)
        else:
            df.to_csv(
                paths.labels_path.joinpath("probing_dataframe.csv"),
                index=False,
            )
    else:
        df = pd.read_csv(paths.labels_path.joinpath(dataset_csv_path))
    return df

def get_filenames_and_starts_for_probe_df(paths, ground_truth, label_column):
    from bacpipe.embedding_evaluation.label_embeddings import get_default_labels, get_dt_filename
    import datetime as dt
    model_name = paths.labels_path.parent.stem
    default_labels = get_default_labels(model_name, overwrite=False)
    filenames = np.array(default_labels['audio_file_name'])[
        ground_truth[f"label:{label_column}"] > -1
    ]
    times_of_day = np.array(default_labels['time_of_day'])[
        ground_truth[f"label:{label_column}"] > -1
    ]
    starts = [
        (
            dt.datetime.strptime(tod_e, '%H-%M-%S')
            - get_dt_filename(tod_f)
        ).seconds
        for tod_e, tod_f in 
        zip(times_of_day, filenames)
        ]
    return filenames, starts