#!/usr/bin/env python3 -u
# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import partial
import numpy as np

from omegaconf import II
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from bacpipe.animal2vec_nn.nn import (
    Modality,
    MaskSeed,
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_annealed_rate,
    D2vDecoderConfig,
    AltBlock,
    Decoder1d,
    D2vAudioConfig,
    AudioEncoder,
    D2vImageConfig,
    ImageEncoder,
    sigmoid_focal_loss,
    FusedSegmentationMixin,
    confusion,
)

logger = logging.getLogger(__name__)


@dataclass
class D2vModalitiesConfig(FairseqDataclass):
    audio: D2vAudioConfig = D2vAudioConfig()
    image: D2vImageConfig = D2vImageConfig()


@dataclass
class Data2VecMultiConfig(FairseqDataclass):
    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    depth: int = 8
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False

    average_top_k_layers: int = field(
        default=16, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = False

    clone_batch: int = 1

    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )

    max_update: int = II("optimization.max_update")

    modalities: D2vModalitiesConfig = D2vModalitiesConfig()

    shared_decoder: Optional[D2vDecoderConfig] = None

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    supported_modality: Optional[Modality] = None
    mae_init: bool = False

    seed: int = II("common.seed")

    # skip_ema: bool = False
    skip_ema: bool = True

    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 1

    decoder_group: bool = False
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    unique_labels: str = II("task.unique_labels")
    with_labels: bool = II("task.with_labels")
    use_focal_loss: bool = II("criterion.use_focal_loss")
    sample_rate: int = II("task.sample_rate")
    metric_threshold: float = 0.25  # II("criterion.metric_threshold")
    iou_threshold: float = II("criterion.iou_threshold")
    sigma_s: float = II("criterion.sigma_s")
    maxfilt_s: float = II("criterion.maxfilt_s")
    max_duration_s: float = II("criterion.max_duration_s")
    lowP: float = II("criterion.lowP")
    method: str = II("criterion.method")
    segmentation_metrics: bool = II("criterion.segmentation_metrics")
    verbose_tensorboard_logging: bool = II("task.verbose_tensorboard_logging")
    conv_feature_layers: str = II("task.conv_feature_layers")
    mixup_prob: float = 0.5
    mixing_window_length: float = 0.1
    source_mixup: float = -1.  # Setting to negative values, effectively disables BC-Learning
    same_mixup: bool = True
    target_mixup: bool = True
    gain_mode: str = "A_weighting"
    target_mixup: bool = False


@register_model("data2vec_multi", dataclass=Data2VecMultiConfig)
class Data2VecMultiModel(BaseFairseqModel, FusedSegmentationMixin):
    def make_modality_encoder(
            self,
            cfg: D2vModalityConfig,
            embed_dim: int,
            make_block: Callable[[float], nn.ModuleList],
            norm_layer: Callable[[int], nn.LayerNorm],
            layer_norm_first: bool,
            alibi_biases,
            task,
    ) -> ModalitySpecificEncoder:
        if cfg.type == Modality.AUDIO:
            enc_cls = AudioEncoder
        elif cfg.type == Modality.IMAGE:
            enc_cls = ImageEncoder
            if hasattr(task, "text_task"):
                task = task.text_task
        else:
            raise Exception(f"unsupported modality {cfg.type}")

        return enc_cls(
            modality_cfg=cfg,
            embed_dim=embed_dim,
            make_block=make_block,
            norm_layer=norm_layer,
            layer_norm_first=layer_norm_first,
            alibi_biases=alibi_biases,
            task=task,
        )

    def __init__(self, cfg: Data2VecMultiConfig, modalities, skip_ema=False, task=None):
        super().__init__()
        self.cfg = cfg

        if self.cfg.with_labels:
            self.final_dropout = torch.nn.Dropout(cfg.final_dropout)
            self.linear_eval_projection = Linear(self.cfg.embed_dim,
                                                 len(eval(self.cfg.unique_labels)),
                                                 bias=True)
            self.use_focal_loss = cfg.use_focal_loss
            self.metric_threshold = cfg.metric_threshold
            self.verbose_tensorboard_logging = cfg.verbose_tensorboard_logging

        self.modalities = modalities
        self.task = task

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        for mod in self.modalities:
            mod_cfg = getattr(cfg.modalities, mod.name.lower())
            if mod.name.lower() == "audio":
                mod_cfg.conv_feature_layers = getattr(cfg, "conv_feature_layers")
            # logger.info("\n\n mod cfg")
            # logger.info(mod_cfg)
            enc = self.make_modality_encoder(
                mod_cfg,
                cfg.embed_dim,
                make_block,
                make_layer_norm,
                cfg.layer_norm_first,
                self.alibi_biases,
                task,
            )
            # logger.info("\n\n enc")
            # logger.info(enc)
            self.modality_encoders[mod.name] = enc
            # logger.info("\n\n mod.name")
            # logger.info(mod.name)
            #
            # logger.info("\n self.modality_encoders:")
            # logger.info(self.modality_encoders)
            #
            # logger.info("\n self.modality_encoders[mod.name].decoder:")
            # logger.info(self.modality_encoders[mod.name].decoder)

            # logger.info("\n\n modality_encoders[mod.name]")
            # logger.info(self.modality_encoders[mod.name])

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        if self.cfg.mae_init:
            self.apply(self._init_weights)
        else:
            from fairseq.modules.transformer_sentence_encoder import init_bert_params

            self.apply(init_bert_params)

        for mod_enc in self.modality_encoders.values():
            mod_enc.reset_parameters()

        if not skip_ema:
            self.ema = self.make_ema_teacher(cfg.ema_decay)
            self.shared_decoder = (
                Decoder1d(cfg.shared_decoder, cfg.embed_dim)
                if self.cfg.shared_decoder is not None
                else None
            )
            if self.shared_decoder is not None:
                self.shared_decoder.apply(self._init_weights)

            self.recon_proj = None
            if cfg.recon_loss > 0:
                self.recon_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        # logger.info("\n\n skip_ema")
        # logger.info(skip_ema)
        #
        # logger.info("\n\n After EMA creation \n\n")
        #
        # logger.info("\n self.modality_encoders:")
        # logger.info(self.modality_encoders)
        #
        # logger.info("\n self.modality_encoders[mod.name].decoder:")
        # logger.info(self.modality_encoders[mod.name].decoder)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn or "p_swish" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

        self.num_updates = 0

    def _init_weights(self, m):

        try:
            from apex.normalization import FusedLayerNorm

            fn = FusedLayerNorm
        except:
            fn = nn.LayerNorm

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def make_ema_teacher(self, ema_decay):
        ema_config = EMAModuleConfig(
            ema_decay=ema_decay,
            ema_fp32=True,
            # log_norms=self.cfg.log_norms,
            # add_missing_params=False,
        )

        model_copy = self.make_target_model()

        return EMAModule(
            model_copy,
            ema_config,
            # copy_model=False,
        )

    def make_target_model(self):
        # logger.info("making target model")

        model_copy = Data2VecMultiModel(
            self.cfg, self.modalities, skip_ema=True, task=self.task
        )

        if self.cfg.ema_encoder_only:
            model_copy = model_copy.blocks
            for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)

            for mod_enc in model_copy.modality_encoders.values():
                mod_enc.decoder = None
                if not mod_enc.modality_cfg.ema_local_encoder:
                    mod_enc.local_encoder = None
                    mod_enc.project_features = None

        model_copy.requires_grad_(False)
        return model_copy

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is not None and (
                (self.num_updates == 0 and num_updates > 1)
                or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            ema_weight_decay = None
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay, weight_decay=ema_weight_decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.blocks if self.cfg.ema_encoder_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_logits(self, net_output, reshape=True):
        y = net_output["linear_eval_projection"]
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
            if self.use_focal_loss:
                y = self.prepare_shapes(y, False)
            else:
                y = y.reshape(-1)
        return y

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

    @classmethod
    def build_model(cls, cfg: Data2VecMultiConfig, task=None):
        """Build a new model instance."""
        if task is None or not hasattr(task, "supported_modalities"):
            modalities = (
                [cfg.supported_modality]
                if cfg.supported_modality is not None
                else [
                    Modality.AUDIO,
                    Modality.IMAGE,
                ]
            )
        else:
            modalities = task.supported_modalities
        # return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)
        return cls(cfg, modalities, task=task, skip_ema=True)

    def forward(
            self,
            source,
            target=None,
            id=None,
            mode=None,
            padding_mask=None,
            mask=True,
            features_only=False,
            force_remove_masked=False,
            remove_extra_tokens=True,
            precomputed_mask=None,
            reduce=True,
            **kwargs
    ):
        # we use BC Learning:
        # Tokozume, Y., Ushiku, Y., & Harada, T. (2017).
        # Learning from Between-class Examples for Deep Sound Recognition.
        # arXiv [cs.LG]. arXiv. https://arxiv.org/abs/1711.10282
        # And put the augmentation stage on the GPUs
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

                if target is not None and self.cfg.target_mixup and self.cfg.with_labels:
                    r = r.unsqueeze(-1)
                    if mix_mask is None:
                        target = target * r + (1 - r) * target[mixup_perm]
                    else:
                        target[mix_mask] = (
                                target[mix_mask] * r + (1 - r) * target[mixup_perm][mix_mask]
                        )
        # print("\n mode pre:", mode, "\n")
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality

        if isinstance(mode, Modality):
            mode = mode.name
        # print("\n mode post:", mode, "\n")
        feature_extractor = self.modality_encoders[mode]

        # logger.info("\n\n mode")
        # logger.info(mode)
        #
        # logger.info("\n self.modality_encoders:")
        # logger.info(self.modality_encoders)
        #
        # logger.info("\n feature_extractor.decoder:")
        # logger.info(feature_extractor.decoder)

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)
        # print("\n Source size", source.size())
        # print("\n Target size", target.size() if target is not None else target)
        # if target is not None and self.cfg.with_labels:
        #     lin_eval_target = target.clone()
        #     target = None
        # else:
        #     lin_eval_target = None

        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )
        # print("\n extractor_out size", extractor_out["x"].size())

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                    not self.training
                    or self.cfg.layerdrop == 0
                    or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only or self.cfg.with_labels:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        if self.cfg.with_labels:
            # Average over all heads
            # no gradient information will flow back to the model when pretraining
            avg_layer = self.average_top_k_layers
            # get everything but detach from main graph
            target_layer_results = [l.clone().detach() for l in layer_results[-avg_layer:]]
            # target_layer_results = [tl for tl.transpose(0, 1) in target_layer_results]  # TBC -> BTC
            # Average over transformer layers
            x_lin_eval = (sum(target_layer_results) / len(target_layer_results)).to(x.dtype)
            # print("\n layer_results sizes", [x.size() for x in layer_results])
            if self.norm is not None:
                x_lin_eval = self.norm(x_lin_eval)

            # if one of the decoders is not None then we do pretrain
            # and the tokens are masked. Otherwise we do finetune and the
            # tokens are not masked. In finetune we do not need the decoder
            # print("\n x_lin_eval.size pre decoding", x_lin_eval.size())
            with torch.no_grad():
                if self.shared_decoder is not None:
                    x_lin_eval = self.forward_decoder(
                        x_lin_eval,
                        feature_extractor,
                        self.shared_decoder,
                        encoder_mask,
                    )
                if feature_extractor.decoder is not None:
                    x_lin_eval = self.forward_decoder(
                        x_lin_eval,
                        feature_extractor,
                        feature_extractor.decoder,
                        encoder_mask,
                    )
            # print("\n x_lin_eval.size post decoding", x_lin_eval.size())

            x_lin_eval = self.final_dropout(x_lin_eval)
            x_lin_eval = self.linear_eval_projection(x_lin_eval.to(dtype=x.dtype))  # Output Shape is BxTxClasses
            # print("\n x_lin_eval.size post projection", x_lin_eval.size())

        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens:]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                                          :, feature_extractor.modality_cfg.num_extra_tokens:
                                          ]

            return {
                "x": x,
                "linear_eval_projection": x_lin_eval if self.cfg.with_labels else None,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

        xs = []

        # print("\n x.size post decoding", x.size())
        if self.shared_decoder is not None:
            dx = self.forward_decoder(  # BTC
                x,
                feature_extractor,
                self.shared_decoder,
                encoder_mask,
            )
            xs.append(dx)
        if feature_extractor.decoder is not None:
            dx = self.forward_decoder(
                x,
                feature_extractor,
                feature_extractor.decoder,
                encoder_mask,
            )
            xs.append(dx)
            # orig_x = x

        assert len(xs) > 0
        # print("\n len(xs) post decoding", len(xs))
        # print("\n xs[0].size post decoding", xs[0].size())

        p = next(self.ema.model.parameters())
        device = x.device
        dtype = x.dtype
        ema_device = p.device
        ema_dtype = p.dtype

        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        if ema_device != device or ema_dtype != dtype:
            logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
            self.ema.model = self.ema.model.to(dtype=dtype, device=device)
            ema_dtype = dtype

            def to_device(d):
                for k, p in d.items():
                    if isinstance(d[k], dict):
                        to_device(d[k])
                    else:
                        d[k] = p.to(device=device)

            to_device(self.ema.fp32_params)
        tm = self.ema.model

        with torch.no_grad():
            tm.eval()

            # print("\n self.cfg.ema_encoder_only: ", self.cfg.ema_encoder_only)
            if self.cfg.ema_encoder_only:
                assert target is None
                ema_input = extractor_out["local_features"]
                ema_input = feature_extractor.contextualized_features(
                    ema_input.to(dtype=ema_dtype),
                    padding_mask,
                    mask=False,
                    remove_masked=False,
                )
                ema_blocks = tm
            else:
                ema_blocks = tm.blocks
                # print("\n feature_extractor.modality_cfg.ema_local_encoder: ",
                #       feature_extractor.modality_cfg.ema_local_encoder)
                if feature_extractor.modality_cfg.ema_local_encoder:
                    inp = (
                        target.to(dtype=ema_dtype)
                        if target is not None
                        else source.to(dtype=ema_dtype)
                    )
                    ema_input = tm.modality_encoders[mode](
                        inp,
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )
                else:
                    # assert target is None
                    ema_input = extractor_out["local_features"]
                    ema_feature_enc = tm.modality_encoders[mode]
                    ema_input = ema_feature_enc.contextualized_features(
                        ema_input.to(dtype=ema_dtype),
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )

            ema_padding_mask = ema_input["padding_mask"]
            ema_alibi_bias = ema_input.get("alibi_bias", None)
            ema_alibi_scale = ema_input.get("alibi_scale", None)
            ema_input = ema_input["x"]

            y = []
            ema_x = []
            extra_tokens = feature_extractor.modality_cfg.num_extra_tokens
            for i, blk in enumerate(ema_blocks):
                ab = ema_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        ema_alibi_scale[i]
                        if ema_alibi_scale.size(0) > 1
                        else ema_alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                ema_input, lr = blk(
                    ema_input,
                    padding_mask=ema_padding_mask,
                    alibi_bias=ab,
                )
                y.append(lr[:, extra_tokens:])
                ema_x.append(ema_input[:, extra_tokens:])

        y = self.make_targets(y, self.average_top_k_layers)
        orig_targets = y
        # print("\n y.size pre repeat interleave", y.size())

        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)
        # print("\n y.size post repeat interleave", y.size())

        masked = encoder_mask.mask.unsqueeze(-1)
        masked_b = encoder_mask.mask.bool()
        y = y[masked_b]
        # print("\n y.size post encoder_mask masking", y.size())

        if xs[0].size(1) == masked_b.size(1):
            xs = [x[masked_b] for x in xs]
        else:
            xs = [x.reshape(-1, x.size(-1)) for x in xs]
        # print("\n xs[0].size post masking", xs[0].size())
        sample_size = masked.sum().long()

        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        sample_size = result["sample_size"]

        if self.cfg.cls_loss > 0:
            assert extra_tokens > 0
            cls_target = orig_targets.mean(dim=1)
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
            cls_pred = x[:, extra_tokens - 1]
            result["losses"]["cls"] = self.d2v_loss(cls_pred, cls_target) * (
                    self.cfg.cls_loss * sample_size
            )

        if self.cfg.recon_loss > 0:
            with torch.no_grad():
                recon_target = feature_extractor.patchify(source)
                mean = recon_target.mean(dim=-1, keepdim=True)
                var = recon_target.var(dim=-1, keepdim=True)
                recon_target = (recon_target - mean) / (var + 1.0e-6) ** 0.5

                if self.cfg.clone_batch > 1:
                    recon_target = recon_target.repeat_interleave(self.cfg.clone_batch, 0)

                if masked_b is not None:
                    recon_target = recon_target[masked_b]

            recon = xs[0]
            if self.recon_proj is not None:
                recon = self.recon_proj(recon)

            result["losses"]["recon"] = (
                    self.d2v_loss(recon, recon_target.float()) * self.cfg.recon_loss
            )

        if self.cfg.d2v_loss > 0:
            for i, x in enumerate(xs):
                # print("\n x.size pre d2v loss", x.size())
                reg_loss = self.d2v_loss(x, y)
                n = f"{mode}_regression_{i}" if len(xs) > 1 else f"{mode}_regression"
                result["losses"][n] = reg_loss * self.cfg.d2v_loss

        if self.cfg.with_labels:
            assert target is not None
            # print("\n target.size pre interleave", target.size())
            if self.cfg.clone_batch > 1:
                target = target.repeat_interleave(self.cfg.clone_batch, 0)

            # print("\n target.size pre masking", target.size())
            if target.size(1) == masked_b.size(1):
                target = target[masked_b]

            # print("\n x_lin_eval.size pre masking", x_lin_eval.size())
            if x_lin_eval.size(1) == masked_b.size(1):
                x_lin_eval = x_lin_eval[masked_b]  # BTC masked

            # print("\n target.size pre reshaping", target.size())
            # print("\n x_lin_eval.size pre reshaping", x_lin_eval.size())
            # target = target.reshape(-1, target.size(-1))
            # x_lin_eval = x_lin_eval.reshape(-1, x_lin_eval.size(-1))

            reduction = "none" if not reduce else "sum"
            if self.use_focal_loss:
                linear_eval_loss = sigmoid_focal_loss(x_lin_eval, target,
                                                      reduction=reduction)
            else:
                linear_eval_loss = torch.nn.functional.cross_entropy(x_lin_eval,
                                                                     target.reshape(-1),
                                                                     reduction=reduction)

            n_correct, total = self.compute_accuracy(x_lin_eval, target)
            tp, fp, tn, fn = self.compute_prec_rec_f1(x_lin_eval, target)
            result["pretrain/n_correct"] = n_correct
            result["pretrain/total"] = total
            result["pretrain/tp"] = tp
            result["pretrain/fp"] = fp
            result["pretrain/tn"] = tn
            result["pretrain/fn"] = fn
            result["losses"]["linear_eval_loss"] = linear_eval_loss

            if self.verbose_tensorboard_logging and not self.training:
                result["_predictions"] = torch.sigmoid(x_lin_eval)
                # print("\n writing ", result["_predictions"].size(), "to results out dict \n")
                result["_targets"] = target
                if self.cfg.segmentation_metrics:
                    result["_source_size"] = source.size(-1)  # needed for the fused metric routine

        suffix = "" if len(self.modalities) == 1 else f"_{mode}"
        with torch.no_grad():
            if encoder_mask is not None:
                result["masked_pct"] = 1 - (
                        encoder_mask.ids_keep.size(1) / encoder_mask.ids_restore.size(1)
                )
            for i, x in enumerate(xs):
                n = f"pred_var{suffix}_{i}" if len(xs) > 1 else f"pred_var{suffix}"
                result[n] = self.compute_var(x.float())
            if self.ema is not None:
                for k, v in self.ema.logs.items():
                    result[k] = v

            y = y.float()
            result[f"target_var{suffix}"] = self.compute_var(y)

            if self.num_updates > 5000:
                if result[f"target_var{suffix}"] < self.cfg.min_target_var:
                    logger.error(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )
                    raise Exception(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )

                for k in result.keys():
                    if k.startswith("pred_var") and result[k] < self.cfg.min_pred_var:
                        logger.error(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )
                        raise Exception(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )

            result["ema_decay"] = self.ema.get_decay() * 1000
        return result

    def forward_decoder(
            self,
            x,
            feature_extractor,
            decoder,
            mask_info,
    ):
        x = feature_extractor.decoder_input(x, mask_info)
        x = decoder(*x)

        return x

    def d2v_loss(self, x, y):
        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, x.size(-1))

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none")
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        reg_loss = loss * scale

        return reg_loss

    def make_targets(self, y, num_layers):

        with torch.no_grad():
            target_layer_results = y[-num_layers:]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
                ]
                permuted = True
            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]
            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        return y

    def compute_accuracy(self, logits, target):
        if target.dim() == logits.dim():
            preds = torch.sigmoid(logits.view(-1, logits.size(-1)))
            preds = torch.where(preds < self.metric_threshold, 0, 1)
            n_correct = torch.sum(preds.eq(target))
            total = torch.tensor(target.shape).prod()
        else:
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1,
                                                     dtype=torch.float32)
            n_correct = torch.sum(
                lprobs.argmax(1).eq(target)
            )
            total = torch.sum(target)

        return n_correct.squeeze(), total.squeeze()

    def compute_prec_rec_f1(self, logits, target):
        if target.dim() == logits.dim():
            preds = torch.sigmoid(logits.view(-1, logits.size(-1)))
            preds = torch.where(preds < self.metric_threshold, 0, 1)
        else:
            preds = torch.softmax(logits, -1)
            preds = torch.nn.functional.one_hot(preds, target.size(-1))
        tp, fp, tn, fn = confusion(preds, target)

        return tp, fp, tn, fn

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
            self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res

    def remove_pretraining_modules(self, modality=None, keep_decoder=False):
        self.ema = None
        self.cfg.clone_batch = 1
        self.recon_proj = None

        if not keep_decoder:
            self.shared_decoder = None

        modality = modality.lower() if modality is not None else None
        for k in list(self.modality_encoders.keys()):
            if modality is not None and k.lower() != modality:
                del self.modality_encoders[k]
            else:
                self.modality_encoders[k].remove_pretraining_modules(
                    keep_decoder=keep_decoder
                )
                if not keep_decoder:
                    self.modality_encoders[k].decoder = None


def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.0)
    return m
