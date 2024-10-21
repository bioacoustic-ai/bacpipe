# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This defines the animal2vec model
"""
import re
import logging
from omegaconf import II, open_dict
import contextlib
from dataclasses import dataclass, field

import torch
import numpy as np
import torch.nn as tnn
from fairseq.tasks import FairseqTask
from fairseq.models import register_model, FairseqEncoder
from fairseq.models.wav2vec import (
    Wav2Vec2CtcConfig,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2VecCtc
)
from argparse import Namespace
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from bacpipe.animal2vec_nn.nn import FusedSegmentationMixin

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

logger = logging.getLogger("animal2vec.hydra_train")


@dataclass
class Wav2Vec2CcasFinetuneConfig(Wav2Vec2CtcConfig):
    unique_labels: str = II("task.unique_labels")
    average_top_k_layers: int = field(
        default=16, metadata={"help": "how many layers to average"}
    )
    use_focal_loss: bool = II("criterion.use_focal_loss")
    sample_rate: int = II("task.sample_rate")
    mixup_prob: float = 0.5
    mixing_window_length: float = 0.1
    source_mixup: float = -1.  # Setting to negative values, effectively disables BC-Learning
    same_mixup: bool = True
    target_mixup: bool = True
    gain_mode: str = "A_weighting"
    load_pretrain_weights: bool = False


@register_model("wav2vec_ccas_finetune", dataclass=Wav2Vec2CcasFinetuneConfig)
class Wav2VecCcasFinetune(Wav2VecCtc, FusedSegmentationMixin):
    @classmethod
    def build_model(cls, cfg: Wav2Vec2CcasFinetuneConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoderModOut(
            cfg,
            len(eval(cfg.unique_labels))
        )
        return cls(cfg, w2v_encoder)

    def get_logits(self, net_output, reshape=True):
        y = net_output["encoder_out"]
        if reshape:
            y = self.prepare_shapes(y, False)
        return y

    @staticmethod
    def prepare_shapes(out, transpose=True):
        if transpose:
            out = out.transpose(0, 2)
        out = out.reshape(-1, out.size(-1))
        return out

    def get_targets(self, sample, net_output, expand_steps=True, reshape=True):
        y = net_output["target"] if "target" in net_output else sample["target"]
        if reshape:
            if self.cfg.use_focal_loss:
                y = self.prepare_shapes(y, transpose=False)
            else:
                y = y.reshape(-1)
        return y


class Wav2VecEncoderModOut(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2CcasFinetuneConfig, output_size=None):
        self.apply_mask = cfg.apply_mask
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            # d2v multi args
            "encoder_dropout": cfg.dropout,
            # "drop_path": getattr(cfg, "drop_path", 0),
            "mask_dropout": getattr(cfg, "mask_dropout", 0),
            # "zero_mask": getattr(cfg, "zero_mask", False),
            "local_grad_mult": cfg.feature_grad_mult,
            "layerdrop": cfg.layerdrop,
            "prenet_layerdrop": cfg.layerdrop,
            "prenet_dropout": cfg.dropout,
            "post_mlp_drop": cfg.dropout,
            # "encoder_zero_mask": getattr(cfg, "zero_mask", False),
            "inverse_mask": False,
            # "learned_alibi_scale": getattr(cfg, "update_alibi", True),
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None

            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        self.modality_type = cfg.w2v_args.model.supported_modality
        self.is_d2v_multi = "data2vec_multi" in w2v_args.model.get("_name", None)

        if not self.is_d2v_multi:
            model_normalized = w2v_args.task.get(
                "normalize", w2v_args.model.get("normalize", False)
            )
            assert cfg.normalize == model_normalized, (
                "Fine-tuning works best when data normalization is the same. "
                "Please check that --normalize is set or unset for both pre-training and here"
            )

            if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
                with open_dict(w2v_args):
                    w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

            w2v_args.task.data = cfg.data
            task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
            model = task.build_model(w2v_args.model, from_checkpoint=True)

            model.remove_pretraining_modules()
            d = w2v_args.model.encoder_embed_dim
        else:
            assert cfg.normalize

            if hasattr(w2v_args.task, "audio"):
                w2v_args.task.audio.data = cfg.data
            else:
                w2v_args.task.data = cfg.data
            task = tasks.setup_task(w2v_args.task, from_checkpoint=True)

            model = task.build_model(w2v_args.model, from_checkpoint=True)

            model.remove_pretraining_modules(modality=self.modality_type.lower())
            d = w2v_args.model.embed_dim

        if state is not None and not cfg.no_pretrained_weights:
            if cfg.load_ema:
                assert "_ema" in state["model"]
                for k in state["model"]["_ema"]:
                    mk = "encoder." + k
                    assert mk in state["model"], mk
                    state["model"][mk] = state["model"]["_ema"][k]
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        self.w2v_model = model

        self.final_dropout = tnn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim

        if targ_d is not None:
            self.proj = Linear(d, targ_d)

        layer_decay = getattr(cfg, "layer_decay", 1)
        if layer_decay < 1:
            mod_encs = list(model.modality_encoders.values())
            assert len(mod_encs) == 1, len(mod_encs)
            blocks = list(mod_encs[0].context_encoder.blocks) + list(model.blocks)
            num_layers = len(blocks) + 1
            layer_scales = list(
                layer_decay ** (num_layers - i) for i in range(num_layers + 1)
            )

            for i, b in enumerate(blocks):
                lid = i + 1
                if layer_scales[lid] == 1.0:
                    continue

                for n, p in b.named_parameters():
                    optim_override = getattr(p, "optim_overrides", {})
                    if "optimizer" not in optim_override:
                        optim_override["optimizer"] = {}
                    if "p_swish" in n:
                        p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
                    optim_override["optimizer"]["lr_scale"] = layer_scales[lid]
                    p.optim_overrides = optim_override

        self.cfg = cfg

        if cfg.w2v_args is not None:
            model_use_focal_loss = getattr(cfg.w2v_args.criterion, "use_focal_loss",
                                           getattr(cfg.w2v_args.model, "use_focal_loss",
                                                   False))
            if cfg.load_pretrain_weights:
                if model_use_focal_loss == cfg.use_focal_loss:
                    if getattr(cfg.w2v_args.task, "with_labels", False):
                        logger.info(
                            "We are re-using the linear eval projection from pretrain "
                            "and resetting freeze_finetune_updates from {} to {}\n".format(
                                self.freeze_finetune_updates,
                                int(.1 * self.freeze_finetune_updates))
                        )
                        self.proj = self.w2v_model.linear_eval_projection
                        self.freeze_finetune_updates = int(.1 * self.freeze_finetune_updates)
                else:
                    logger.info(
                        "We are NOT re-using the linear eval projection from pretrain, as "
                        "model_use_focal_loss is {} and cfg.use_focal_loss is {}\n".format(
                            model_use_focal_loss, cfg.use_focal_loss
                        )
                    )

    # adapted from https://github.com/mil-tokyo/bc_learning_sound/blob/master/utils.py
    def compute_gain_torch(self, sound, fs=8_000, wl=0.1, min_db=-80.0, mode="A_weighting"):
        n_fft = round(fs * wl)

        if mode == "A_weighting":
            if not hasattr(self, f"a_weight"):
                self.a_weight = {}

            if fs not in self.a_weight:
                def a_weight(fs, n_fft, min_db=-80.0):
                    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
                    freq_sq = freq ** 2
                    freq_sq[0] = 1.0
                    weight = 2.0 + 20.0 * (
                            2 * np.log10(12194)
                            + 2 * np.log10(freq_sq)
                            - np.log10(freq_sq + 12194 ** 2)
                            - np.log10(freq_sq + 20.6 ** 2)
                            - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                            - 0.5 * np.log10(freq_sq + 737.9 ** 2)
                    )
                    weight = np.maximum(weight, min_db)

                    return weight

                self.a_weight[fs] = torch.from_numpy(
                    np.power(10, a_weight(fs, n_fft, min_db) / 10)
                ).to(device=sound.device)

        sound = sound.unfold(-1, n_fft, n_fft // 2)

        if mode == "RMSE":
            sound = sound ** 2
            g = sound.mean(-1)
        elif mode == "A_weighting":
            w = torch.hann_window(n_fft, device=sound.device) * sound
            spec = torch.fft.rfft(w)
            power_spec = spec.abs() ** 2
            a_weighted_spec = power_spec * self.a_weight[fs]
            g = a_weighted_spec.sum(-1)
        else:
            raise Exception("Invalid mode {}".format(mode))

        gain = torch.maximum(g, torch.tensor(10 ** (min_db / 10), device=g.device))
        gain_db = 10 * torch.log10(gain)

        return gain_db

    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }
            load_dict = new_big_dict
            model.load_state_dict(new_big_dict, strict=False)
        else:
            to_delete = {"_ema", "target_proj", "decoder"}
            for k in to_delete:
                if k in state["model"]:
                    del state["model"][k]
            # print("\n\n\n ########## W2V ARGS ######### \n\n\n", cfg.w2v_args)
            if hasattr(model, "modality_encoders"):
                if "modality_encoders.{}.encoder_mask".format(self.modality_type) not in state["model"]:
                    model.modality_encoders[self.modality_type].encoder_mask = None
                elif not cfg.zero_mask:
                    model.modality_encoders[self.modality_type].encoder_mask = None
                    del state["model"]["modality_encoders.{}.encoder_mask".format(self.modality_type)]

                for k in list(state["model"].keys()):
                    if k.startswith("modality_encoders.") and not k.startswith(
                            "modality_encoders.{}".format(self.modality_type)
                    ):
                        del state["model"][k]
            load_dict = state["model"]
            model.load_state_dict(state["model"], strict=True)
        logger.info("Loading the following weights:")
        logger.info(["Key: {}. Having Shape: {}".format(k, v.size()) for k, v in load_dict.items()])

    def forward(self, source, padding_mask=None, target=None, **kwargs):
        # we use BC Learning:
        # Tokozume, Y., Ushiku, Y., & Harada, T. (2017).
        # Learning from Between-class Examples for Deep Sound Recognition.
        # arXiv [cs.LG]. arXiv. https://arxiv.org/abs/1711.10282
        # And put the augmentation stage on the GPUs
        # print("\n source.size()", source.size())
        if self.cfg.source_mixup >= 0 and self.training and self.cfg.mixup_prob > 0:
            with torch.no_grad():
                mixed_source = source
                mix_mask = None
                if self.cfg.mixup_prob < 1:
                    mix_mask = (
                        torch.empty((source.size(0),), device=source.device)
                        .bernoulli_(self.cfg.mixup_prob)
                        .bool()
                    )
                    mixed_source = source[mix_mask]

                r = (
                    torch.FloatTensor(
                        1 if self.cfg.same_mixup else mixed_source.size(0)
                    )
                    .uniform_(max(1e-6, self.cfg.source_mixup), 1)
                    .to(dtype=source.dtype, device=source.device)
                )

                mixup_perm = torch.randperm(source.size(0))
                s2 = source[mixup_perm]

                if self.cfg.gain_mode == "none":
                    p = r.unsqueeze(-1)
                    if mix_mask is not None:
                        s2 = s2[mix_mask]
                else:
                    if self.cfg.gain_mode == "naive_rms":
                        G1 = source.pow(2).mean(dim=-1).sqrt()
                    else:
                        G1, _ = self.compute_gain_torch(
                            source, mode=self.cfg.gain_mode,
                            fs=self.cfg.sample_rate,
                            wl=self.cfg.mixing_window_length
                        ).max(-1)
                        G1 = G1.to(dtype=source.dtype)

                    G2 = G1[mixup_perm]

                    if mix_mask is not None:
                        G1 = G1[mix_mask]
                        G2 = G2[mix_mask]
                        s2 = s2[mix_mask]

                    p = 1 / (1 + 10 ** ((G1 - G2) / 20) * (1 - r) / r)
                    p = p.unsqueeze(-1)

                mixed = (p * mixed_source) + (1 - p) * s2

                if mix_mask is None:
                    source = mixed / torch.sqrt(p ** 2 + (1 - p) ** 2)
                else:
                    source[mix_mask] = mixed / torch.sqrt(p ** 2 + (1 - p) ** 2)

                if target is not None and self.cfg.target_mixup:
                    r = r.unsqueeze(-1)
                    if mix_mask is None:
                        target = target * r + (1 - r) * target[mixup_perm]
                    else:
                        target[mix_mask] = (
                                target[mix_mask] * r + (1 - r) * target[mixup_perm][mix_mask]
                        )

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        if self.is_d2v_multi:
            w2v_args["mode"] = self.modality_type

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            padding_mask = res["padding_mask"]

        layer_results = res["layer_results"]

        # print("\n res[x].size()", res["x"].size())
        avg_layer = self.cfg.average_top_k_layers

        # get everything
        if self.is_d2v_multi:
            target_layer_results = [l for l in layer_results[-avg_layer:]]
        else:
            target_layer_results = [l[0] for l in layer_results[-avg_layer:]]
        if not self.is_d2v_multi:  # transpose if w2v
            target_layer_results = [tl.transpose(0, 1) for tl in target_layer_results]  # TBC -> BTC
        # Average over transformer layers
        try:
            x = (sum(target_layer_results) / len(target_layer_results)).to(res["x"].dtype)
        except ZeroDivisionError as e:
            print("\n len(target_layer_results)", len(target_layer_results))
            print("\n target_layer_results\n\n", target_layer_results)
            print("\n len(layer_results)", len(layer_results))
            print("\n layer_results", layer_results)
            raise e
        # Dropout and classify
        x = self.final_dropout(x)
        x = self.proj(x)  # Output Shape is BTC

        # print("\n x.size()", x.size())
        # print("\n target.size()", target.size())

        return {
            "encoder_out": x,  # B x T x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": layer_results,
            "target": target
        }

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Linear(in_features, out_features, bias=True):
    m = tnn.Linear(in_features, out_features, bias)
    tnn.init.xavier_uniform_(m.weight)
    if bias:
        tnn.init.constant_(m.bias, 0.0)
    return m
