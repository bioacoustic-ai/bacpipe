# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This defines the audio pretraining task
"""
import io
import os
import h5py
import logging
import re
import warnings
from abc import ABC
import torch
import numpy as np

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II
from scipy import interpolate

from fairseq import metrics, utils
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from fairseq.data.audio.audio_utils import parse_path, read_from_stored_zip, is_sf_audio_data
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
import torch.multiprocessing
from ievad.animal2vec_nn.nn.utils import get_conv_size

logger = logging.getLogger("animal2vec.hydra_train")


# torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class AudioConfigCCAS(AudioPretrainingConfig):
    sample_rate: int = field(
        default=8_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    do_focal_prediction: bool = field(
        default=True,
        metadata={"help": "if set, try to predict if the vocalization was 'focal' or not."
                          "That is, being issued by the animal that is tracked (focal) or by a"
                          "nearby animal (non focal)."},
    )
    with_labels: bool = field(
        default=False,
        metadata={"help": "if set, we have labels available that were created using "
                          "one of the preparation scripts in the 'scripts' folder"},
    )
    verbose_tensorboard_logging: bool = field(
        default=False,
        metadata={"help": "if set, we log precision and recall curves as well as embeddings"},
    )
    min_label_size: int = field(
        default=0,
        metadata={"help": "if set, we only load files from the manifest whose label file bytesize "
                          "is larger then this value. Hint: Empty label files have Bytesize 1780."
                          "Only used when with_labels is True"},
    )
    split: Optional[str] = field(
        default="pretrain",
        metadata={"help": "use this dataset split for getting embeddings during eval"},
    )
    unique_labels: Optional[str] = field(
        default="['beep', 'synch', 'eating', 'cc', 'oth', 'sn', 'ld', 'mo', 'agg', 'al', 'soc', 'focal']",
        metadata={"help": "a 'string list' of all the unique labels."
                          "Only used when with_labels is True"},
    )
    conv_feature_layers: str = field(
        default='[(63, 125, 1)] +[(512, 10, 5)] + [(512, 3, 2)] * 3 + [(512, 3, 1)] + [(512, 2, 1)] * 2',
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
                    "[(dim, kernel_size, stride), ...]"
        },
    )
    train_subset: str = II("dataset.train_subset")
    valid_subset: str = II("dataset.valid_subset")
    use_focal_loss: bool = II("criterion.use_focal_loss")
    segmentation_metrics: bool = II("criterion.segmentation_metrics")
    rebuild_batches: bool = False
    multi_corpus_keys: Optional[bool] = False
    multi_corpus_sampling_weights: Optional[bool] = False
    precompute_mask_config: Optional[bool] = False
    post_save_script: Optional[bool] = False
    subsample: Optional[float] = 1.0
    seed: Optional[bool] = False



@register_task("audio_ccas", dataclass=AudioConfigCCAS)
class AudioTaskCCAS(AudioPretrainingTask):
    """ """

    def __init__(self, cfg: FairseqDataclass, **kwargs):
        super().__init__(cfg, **kwargs)
        self.unique_labels = None
        self.use_focal_loss = None
        self.segmentation_metrics = None
        # print(cfg)
        if getattr(cfg, "unique_labels") and getattr(cfg, "with_labels"):
            self.unique_labels = eval(getattr(cfg, "unique_labels"))
        if getattr(cfg, "use_focal_loss") and getattr(cfg, "with_labels"):
            self.use_focal_loss = getattr(cfg, "use_focal_loss")
        if getattr(cfg, "segmentation_metrics") and getattr(cfg, "with_labels"):
            self.segmentation_metrics = getattr(cfg, "segmentation_metrics")

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        do_focal_prediction = getattr(task_cfg, "do_focal_prediction")
        return_labels = getattr(task_cfg, "with_labels")
        min_label_size = getattr(task_cfg, "min_label_size")

        self.datasets[split] = FileAudioLabelDataset(
            manifest_path=manifest_path,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            max_sample_size=task_cfg.max_sample_size,
            min_sample_size=task_cfg.min_sample_size,
            pad=task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
            # compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
            text_compression_level=text_compression_level,
            return_labels=return_labels,
            unique_labels=self.unique_labels,
            use_focal_loss=self.use_focal_loss,
            min_label_size=min_label_size,
            conv_feature_layers=task_cfg.conv_feature_layers,
            segmentation_metrics=self.segmentation_metrics,
            do_focal_prediction=do_focal_prediction,
            # **self._get_mask_precompute_kwargs(task_cfg),
        )

        if self.cfg.tpu and task_cfg.inferred_w2v_config.mask_channel_prob == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs
        self_func = getattr(self, "aggregate_logging_outputs").__func__
        if self_func is not base_func:
            utils.deprecation_warning(
                "Tasks should implement the reduce_metrics API. "
                "Falling back to deprecated aggregate_logging_outputs API."
            )
            agg_logging_outputs = self.aggregate_logging_outputs(
                logging_outputs, criterion
            )
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("misc/wpb", ntokens, priority=180, round=1)
            metrics.log_speed("misc/wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("misc/bsz", nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)


class FileAudioLabelDataset(RawAudioDataset, ABC):
    """
    This modifies the fairseq FileAudioDataset to return labels as well.
    The labels have to be created using one of the
    preparation scripts in the 'scripts' folder.
    """

    def __init__(
            self,
            manifest_path,
            sample_rate,
            max_sample_size=None,
            min_sample_size=0,
            shuffle=True,
            pad=False,
            normalize=False,
            num_buckets=0,
            text_compression_level=TextCompressionLevel.none,
            unique_labels=None,
            use_focal_loss=None,
            return_labels=True,
            conv_feature_layers=None,
            min_label_size=0,
            segmentation_metrics=None,
            do_focal_prediction=True,
            **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            feature_encoder_spec=conv_feature_layers,
            **mask_compute_kwargs,
        )
        self.text_compressor = TextCompressor(level=text_compression_level)

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            # Infer the parent of the label directory which might be in self.root_dir,
            # although some configurations provide a path to the audio directory which
            # we assume is one of the following:
            audio_dirs = ["wav", "flac", "audio"]
            parents, last = os.path.split(self.root_dir)
            if last in audio_dirs:
                self.label_dir = parents
            else:
                self.label_dir = self.root_dir

            # Each line of the manifest contains a filename relative to self.root_dir and the number of samples
            # Determine examples that should be added based on presence of label files and duration
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                label_file = self.filename_audio2label(os.path.join(self.root_dir, items[0]), lblext="h5")
                label_file_check = os.path.isfile(label_file)
                if label_file_check:
                    label_size = os.path.getsize(label_file)
                else:
                    label_size = 0.
                if (min_sample_size is not None and sz < min_sample_size) or label_size <= min_label_size:
                    # print("\n min_sample_size is not None", min_sample_size is not None)
                    # if min_sample_size is not None:
                    #     print("sz < min_sample_size", sz < min_sample_size, sz, min_sample_size)
                    # print("label_size <= min_label_size\n ", label_size <= min_label_size, label_size, min_label_size)
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)
        # The unique labels from the Meerkat dataset
        self.do_focal_prediction = do_focal_prediction
        self.unique_labels = unique_labels
        self.use_focal_loss = use_focal_loss
        self.segmentation_metrics = segmentation_metrics
        self.return_labels = return_labels
        self.conv_feature_layers = eval(conv_feature_layers)

    # Used to transform audio file path to label file path
    # Assumes that audio is in a "wav" directory, file extension can be arbitrary (e.g., .flac)
    audio2label_re = re.compile(r"(?P<pre>.*)(?P<dir>wav)(?P<post>/.*\.)(?P<ext>[a-z]+)$", re.IGNORECASE)

    def filename_audio2label(self, audiofile, lbldir="lbl", lblext="npz"):
        """
        Given a path to an audio file w/ case independent format:
        .../wav/.../filename.wav,  convert to the expected label name path:
        .../lbldir/.../filename.lblext where lbldir/lblext are parameters

        :param audiofile: path to audio file
        :param lbldir:  Name of label directory
        :param lblext:  Label file extension
        :return:  labelfile path
        """

        # Look for pattern .../wav/.../file.ext
        m = self.audio2label_re.match(audiofile)
        if m is None:
            raise RuntimeError(f"Cannot derive label file from: {audiofile}")
        # Replace wav and ext with label directory & extension
        labelfile = m.expand(f"\g<pre>{lbldir}\g<post>{lblext}")
        return labelfile

    def __getitem__(self, index):
        import soundfile as sf
        fn = self.fnames[index]  # binary representation or list
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)

        # Load the audio data
        path_or_fp = os.path.join(self.root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        data_dict = {"id": index, "source": feats}
        # print("\n feats.size", feats.size())

        # Load the label data
        if self.return_labels:
            lbl_file = self.filename_audio2label(os.path.join(self.label_dir, fn), lblext="h5")
            with h5py.File(lbl_file, "r") as f:
                start_frame_lbl = list(f["start_frame_lbl"])
                end_frame_lbl = list(f["end_frame_lbl"])
                lbl_cat = list(f["lbl_cat"])
                if self.use_focal_loss and self.do_focal_prediction:
                    foc = list(f["foc"])

            ft_out_size = [len(wav.squeeze())]
            for xx in self.conv_feature_layers:
                ft_out_size = [get_conv_size(ft_out_size, [min(10, xx[1])], [0], [1], [xx[2]], dim=1)[0]]

            ft_out_size = np.array(ft_out_size).squeeze().astype(int)
            # print("\n conv_feature_layers", self.conv_feature_layers)
            # print("\n ft_out_size", ft_out_size)
            wav_len = len(wav.squeeze())
            # print("\n wav_len, len(self.unique_labels)", wav_len, len(self.unique_labels))
            if self.use_focal_loss:
                source_vector = np.zeros(shape=[wav_len, len(self.unique_labels)],
                                         dtype=np.int64)
            else:
                source_vector = np.zeros(shape=[wav_len],
                                         dtype=np.int64)
            source_idx_vector = np.arange(wav_len)
            target_vector_idx_vector = np.round(np.linspace(0, wav_len,
                                                            ft_out_size,
                                                            endpoint=False)).astype(np.int64)

            if len(start_frame_lbl) > 0 and len(end_frame_lbl) > 0 and len(lbl_cat) > 0:
                for ii, (s, e, l) in enumerate(zip(start_frame_lbl, end_frame_lbl, lbl_cat)):
                    if self.use_focal_loss:
                        source_vector[int(s):int(e), int(l)] = 1
                        if self.do_focal_prediction and self.unique_labels[-1].lower() == "focal":
                            # is it a focal call or not
                            focal_lbl = foc[ii]
                            if focal_lbl == 1:
                                source_vector[int(s):int(e), -1] = focal_lbl
                    else:
                        source_vector[int(s):int(e)] = int(l) + 1

            f = interpolate.interp1d(source_idx_vector, source_vector,
                                     axis=0, kind="linear")
            target_vector = np.round(f(target_vector_idx_vector), decimals=0).astype(np.int64)
            # print("\n target_vector", target_vector)
            # print("\n target_vector shape", target_vector.shape)
            data_dict.update({"target": target_vector})

        return data_dict

    def collate_samples(self, samples):
        sizes = [len(s) for s in samples]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)  # Padding to max size
        else:
            target_size = min(min(sizes), self.max_sample_size)  # Cropping to min size

        if samples[0].dim() > 1:
            collated_samples = samples[0].new_zeros(len(samples), target_size, samples[0].size(-1))
        else:
            collated_samples = samples[0].new_zeros(len(samples), target_size)
        # print("\n\n collated_samples.size()", collated_samples.size())

        padding_mask = (torch.BoolTensor(
            collated_samples.shape).fill_(False) if self.pad else None)

        for i, (source, size) in enumerate(zip(samples, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_samples[i] = source
            elif diff < 0:
                assert self.pad
                if source.dim() > 1:
                    targets = [torch.cat([x, x.new_full((-diff,), 0.0)]) for x in source.t()]
                    collated_samples[i] = torch.cat(targets, dim=1).view(-1, target_size).t()
                else:
                    collated_samples[i] = torch.cat([source, source.new_full((-diff,), 0.0)])

                padding_mask[i, diff:] = True
            else:
                if source.dim() > 1:
                    # print("\n\n diff", diff)
                    # print("\n\n target_size", target_size)
                    # print("\n\n source.size()", source.size())
                    targets = [self.crop_to_max_size(x.squeeze(), target_size) for x in source.t()]
                    # print("\n\n targets len", [len(x) for x in targets])
                    # targets_numpy = np.array(targets).T
                    collated_samples[i] = torch.cat(targets).view(-1, target_size).t()
                    # print("\n\n collated_samples[i].size()", collated_samples[i].size())
                else:
                    # print("\n\n source_size with dim =< 1", source_size)
                    collated_samples[i] = self.crop_to_max_size(source, target_size)
        return collated_samples, padding_mask

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        collated_sources, padding_mask = self.collate_samples(sources)
        inp = {"source": collated_sources}

        if self.return_labels:
            targets = [torch.tensor(s["target"]) for s in samples]
            collated_targets, _ = self.collate_samples(targets)
            inp["target"] = collated_targets
            # inp["target"] = torch.tensor(np.array(targets), dtype=torch.long)
            # print("TASK", inp["target"].shape, flush=True)

        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            inp["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                inp["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                inp["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if self.return_labels:
            out["target"] = inp["target"]
            out["ntokens"] = sum([len(t) for t in out["target"]])
        # if self.segmentation_metrics:
        #     seg_target_idx = [s["seg_target_idx"] for s in samples]
        #     out["seg_target_idx"] = seg_target_idx
        #     inp["seg_target_idx"] = seg_target_idx
        out["net_input"] = inp
        return out
