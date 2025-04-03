import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import yaml

with open("bacpipe/settings.yaml", "rb") as f:
    settings = yaml.load(f, Loader=yaml.CLoader)

REDUCED_EMBEDS = settings["use_reduced_dim_embeds_for_tasks"]


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
        if False:
            self.sound_files = list(self.dataset.wavfilename)
        self.pretrained_model_name = pretrained_model_name
        self.labels = list(self.dataset[target_labels])
        self.label2index = label2index
        if REDUCED_EMBEDS:
            if False:
                p = list(
                    Path(settings["dim_reduc_parent_dir"]).glob(
                        f"*{pretrained_model_name}*/*.npy"
                    )
                )[-1]
                self.all_embeds = np.load(p, allow_pickle=True).item()
            else:
                # p = '/home/siriussound/Code/bacpipe/exploration/neotropic_dawn_chorus/numpy_embeddings/embed_dict.npy'
                # all_embeds = np.load(p, allow_pickle=True).item()[pretrained_model_name]['split']
                # self.all_embeds = np.concatenate(list(all_embeds.values()))
                # all_labels = np.concatenate(
                #     [np.array([k] * v.shape[0]) for k, v in all_embeds.items()]
                # )
                # assert np.all(all_labels == partition_dataframe.species), (
                #     "The labels in the embeddings file do not match the labels in the task annotations."
                # )

                # p = '/home/siriussound/Code/bacpipe/exploration/neotropic_dawn_chorus/numpy_embeddings/embed_dict.npy'
                # all_embeds = np.load(p, allow_pickle=True).item()[pretrained_model_name]
                # p = f"/home/siriussound/Code/bacpipe/exploration/neotropic_dawn_chorus/numpy_embeddings/umap_300/umap_300_{pretrained_model_name}.npy"
                # p = '/home/siriussound/Code/bacpipe/exploration/anuran_set/numpy_embeddings/embed_dict.npy'
                # all_embeds = np.load(p, allow_pickle=True).item()[pretrained_model_name]
                p = f"/home/siriussound/Code/bacpipe/exploration/anuran_set/numpy_embeddings/pca_300/pca_300_{pretrained_model_name}.npy"
                all_embeds = np.load(p, allow_pickle=True).item()
                self.all_embeds = all_embeds["all"]
                ind2lab = {v: k for k, v in all_embeds["label_dict"].items()}
                labs = [ind2lab[i] for i in all_embeds["labels"]]
                assert np.all(
                    labs == partition_dataframe.species
                ), "The labels in the embeddings file do not match the labels in the task annotations."
                self.dataset = self.dataset.sample(frac=1, random_state=42)
                self.labels = list(self.dataset[target_labels])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if True:
            X = self.all_embeds[self.dataset.index[idx]]
            X = X.reshape(1, -1)
        else:
            sound_file = self.sound_files[idx]
            embd_idx = np.where(self.embed2wavfile_mapper[:, 1] == sound_file)[0][0]
            embed_file = self.embed2wavfile_mapper[embd_idx, 0]

            if not REDUCED_EMBEDS:
                X = np.load(embed_file)
            else:
                index_file = sound_file.replace(
                    ".wav", f"_{self.pretrained_model_name}"
                )
                X = self.all_embeds[index_file]

        if X.shape[0] > 1:
            X = np.mean(X, axis=0)
        else:
            X = X.flatten()
        y = self.label2index[self.labels[idx]]

        return X, y


# TODO handle different embedding sizes
