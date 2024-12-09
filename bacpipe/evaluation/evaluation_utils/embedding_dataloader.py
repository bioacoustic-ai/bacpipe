import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EmbeddingTaskLoader(Dataset):
    """dataset of precomputed embeddings with hierarchical labels.
    embeddings_model refers to the name of the pretrained model that generated the embeddings: vggish, openl3, wav2vec, Birdnet....
    """

    def __init__(
        self,
        partition_dataframe,
        pretrained_model_name,
        loader_object,
        target_labels,
        label2index={},
    ):
        self.dataset = partition_dataframe
        self.features_folder = loader_object.embed_dir
        all_embedding_files = loader_object.files
        if not len(all_embedding_files) == len(self.dataset):
            self.embed_files = [
                f
                for f in all_embedding_files
                if f.stem.replace(f"_{pretrained_model_name}", ".wav")
                in list(self.dataset.wavfilename)
            ]
        self.pretrained_model_name = pretrained_model_name
        self.labels = list(self.dataset[target_labels])
        self.label2index = label2index

    def __len__(self):
        return len(self.embed_files)

    def __getitem__(self, idx):
        X = np.load(self.embed_files[idx])
        y = self.label2index[self.labels[idx]]
        return X, y, self.labels[idx], self.embed_files[idx].stem


# TODO handle different embedding sizes
