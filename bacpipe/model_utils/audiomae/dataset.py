# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# AST: https://github.com/YuanGongND/ast
# --------------------------------------------------------
import csv, os, sys
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
import torch.distributed as dist
import random
import math


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
        self, sampler, dataset, num_replicas=None, rank=None, shuffle: bool = True
    ):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle
        )
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


class DistributedWeightedSampler(Sampler):
    # dataset_train, samples_weight,  num_replicas=num_tasks, rank=global_rank
    def __init__(
        self,
        dataset,
        weights,
        num_replicas=None,
        rank=None,
        replacement=True,
        shuffle=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.weights = torch.from_numpy(weights)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # targets = self.dataset.targets
        # # select only the wanted targets for this subsample
        # targets = torch.tensor(targets)[indices]
        # assert len(targets) == self.num_samples
        # # randomly sample this subset, producing balanced classes
        # weights = self.calculate_weights(targets)
        weights = self.weights[indices]

        subsample_balanced_indicies = torch.multinomial(
            weights, self.num_samples, self.replacement
        )
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]
        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row["index"]] = row["display_name"]
            line_count += 1
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


class AudiosetDataset(Dataset):
    def __init__(self, sr, audio_conf, use_fbank=False, fbank_dir=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        self.sr = sr
        self.audio_conf = audio_conf
        print(
            "---------------the {:s} dataloader---------------".format(
                self.audio_conf.get("mode")
            )
        )
        if "multilabel" in self.audio_conf.keys():
            self.multilabel = self.audio_conf["multilabel"]
        else:
            self.multilabel = False
        print(f"multilabel: {self.multilabel}")
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm")
        self.timem = self.audio_conf.get("timem")
        print(
            "using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        self.mixup = self.audio_conf.get("mixup")
        print("using mix-up with rate {:f}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        print(
            "Dataset: {}, mean {:.3f} and std {:.3f}".format(
                self.dataset, self.norm_mean, self.norm_std
            )
        )

    def _wav2fbank(self, waveform, sr):

        waveform = waveform - waveform.mean()
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.melbins,
            dither=0.0,
            frame_shift=10,
        )
        # 512
        target_length = self.audio_conf.get("target_length")
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank, 0

    def _fbank(self, filename, filename2=None):
        if filename2 == None:
            fn1 = os.path.join(
                self.fbank_dir, os.path.basename(filename).replace(".wav", ".npy")
            )
            fbank = np.load(fn1)
            return torch.from_numpy(fbank), 0
        else:
            fn1 = os.path.join(
                self.fbank_dir, os.path.basename(filename).replace(".wav", ".npy")
            )
            fn2 = os.path.join(
                self.fbank_dir, os.path.basename(filename2).replace(".wav", ".npy")
            )
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            fbank = mix_lambda * np.load(fn1) + (1 - mix_lambda) * np.load(fn2)
            return torch.from_numpy(fbank), mix_lambda

    def process(self, data):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        fbank, mix_lambda = self._wav2fbank(data, self.sr)

        fbank = torch.transpose(fbank.squeeze(), 0, 1)  # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank

    def __len__(self):
        return len(self.data)
