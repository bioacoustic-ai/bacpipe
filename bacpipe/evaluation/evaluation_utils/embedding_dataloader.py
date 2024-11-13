
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EmbeddingsWith3LevelsHierarchicalLabels(Dataset):

    ''' dataset of precomputed embeddings with hierarchical labels.
    embeddings_model refers to the name of the pretrained model that generated the embeddings: vggish, openl3, wav2vec, Birdnet....'''
    def __init__(self, partition_dataframe, pretrained_model_name, embeddings_path, target_labels, label2index={}):
        self.dataset = partition_dataframe 
        self.items_list = list(self.dataset.wavfilename)
        self.features_folder = embeddings_path
        self.pretrained_model_name = pretrained_model_name
        self.labels = target_labels
        self.label2index = label2index

    def __len__(self):
        return len(self.items_list)

    def __getitem__(self,idx):
        filename = self.items_list[idx][0:-4] + '_' + self.pretrained_model_name + '.npy'     #TODO check if we are using the name of the pretrained model on the file name
        X = np.load(os.path.join(self.features_folder, filename))
        y = self.label2index[self.labels[idx]]
        return X, y, self.labels[idx], filename
    