import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import yaml

with open("bacpipe/path_settings.yaml", "rb") as f:
    path_settings = yaml.load(f, Loader=yaml.CLoader)

REDUCED_EMBEDS = path_settings["use_reduced_dim_embeds_for_tasks"]


class EmbeddingTaskLoader(Dataset):
    """dataset of precomputed embeddings with hierarchical labels.
    embeddings_model refers to the name of the pretrained model that generated the embeddings: vggish, openl3, wav2vec, Birdnet....
    """

    def __init__(
        self,
        partition_dataframe,
        pretrained_model_name,
        embed2wavfile_mapper,
        loader_object,
        target_labels,
        label2index={},
        set_name=None,
    ):
        if set_name is not None:
            self.dataset = partition_dataframe[
                partition_dataframe.predefined_set == set_name
            ]
        else:
            self.dataset = partition_dataframe

        print(
            f"Found {len(self.dataset)} samples in the {set_name} set with "
            f"{len(self.dataset[target_labels].unique())} unique labels."
        )

        self.features_folder = loader_object.embed_dir
        self.embed2wavfile_mapper = embed2wavfile_mapper
        self.sound_files = list(self.dataset.wavfilename)
        self.pretrained_model_name = pretrained_model_name
        self.labels = list(self.dataset[target_labels])
        self.label2index = label2index
        if REDUCED_EMBEDS:
            p = list(
                Path(path_settings["dim_reduc_parent_dir"]).glob(
                    f"*{pretrained_model_name}*/*.npy"
                )
            )[-1]
            self.all_embeds = np.load(p, allow_pickle=True).item()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sound_file = self.sound_files[idx]
        embd_idx = np.where(self.embed2wavfile_mapper[:, 1] == sound_file)[0][0]
        embed_file = self.embed2wavfile_mapper[embd_idx, 0]

        if not REDUCED_EMBEDS:
            X = np.load(embed_file)
        else:
            index_file = sound_file.replace(".wav", f"_{self.pretrained_model_name}")
            X = self.all_embeds[index_file]

        if X.shape[0] > 1:
            X = np.mean(X, axis=0)
        else:
            X = X.flatten()
        y = self.label2index[self.labels[idx]]

        return X, y, self.labels[idx], embed_file.stem


# TODO handle different embedding sizes
