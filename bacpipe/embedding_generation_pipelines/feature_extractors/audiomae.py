# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


import torch
import torch.nn as nn
from tqdm import tqdm

from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple


import bacpipe.model_specific_utils.audiomae.models_vit as models_vit
from bacpipe.model_specific_utils.audiomae.dataset import AudiosetDataset
from ..utils import ModelBaseClass

BATCH_SIZE = 8  # important to lower this if run on laptop cpu

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(10 * SAMPLE_RATE)


class PatchEmbed_new(nn.Module):
    """Flexible Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )  # with overlapped patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        self.nb_classes = 527
        self.model = "vit_base_patch16"
        self.model_path = self.model_base_path / "audiomae/finetuned.pth"
        self.global_pool = True

        self.drop_path = 0.1
        self.dataset = "audioset"

        self.mask_2d = True
        self.norm_stats = [-4.2677393, 4.5689974]
        target_length = 1024
        self.img_size = (target_length, 128)  # 1024, 128
        self.in_chans = 1
        self.emb_dim = 768
        self.audio_conf_val = {
            "num_mel_bins": 128,
            "target_length": target_length,
            "freqm": 0,
            "timem": 0,
            "mixup": 0,
            "dataset": self.dataset,
            "mode": "val",
            "mean": self.norm_stats[0],
            "std": self.norm_stats[1],
            "noise": False,
        }
        self.model = models_vit.__dict__[self.model](
            num_classes=self.nb_classes,
            drop_path_rate=self.drop_path,
            global_pool=self.global_pool,
            mask_2d=self.mask_2d,
            use_custom_patch=False,
        )
        self.model.patch_embed = PatchEmbed_new(
            img_size=self.img_size,
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=self.emb_dim,
            stride=16,
        )  # no overlap. stride=img_size=16
        num_patches = self.model.patch_embed.num_patches
        # num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        self.model.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.emb_dim), requires_grad=False
        )  # fixed sin-cos embedding

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        checkpoint_model = checkpoint["model"]
        # load pre-trained model
        self.model.load_state_dict(checkpoint_model)
        # manually initialize fc layer
        trunc_normal_(self.model.head.weight, std=2e-5)

        self.audio_obj = AudiosetDataset(sr=SAMPLE_RATE, audio_conf=self.audio_conf_val)

    def preprocess(self, audio):
        processed_frame = []
        for frame in audio:
            processed_frame.append(self.audio_obj.process(frame.view(1, -1)))
        processed_frame = torch.stack(processed_frame)
        return processed_frame.unsqueeze(dim=1)

    @torch.inference_mode()
    def __call__(self, input):
        return self.model(input)
