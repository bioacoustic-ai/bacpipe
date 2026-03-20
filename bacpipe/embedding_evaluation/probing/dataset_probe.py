from torch.utils.data import Dataset, DataLoader
import numpy as np

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
