import torch
from torch import nn
import yaml
from pydantic import BaseModel

from bacpipe.model_specific_utils.naturelm_audio.BEATs import BEATs, BEATsConfig
from ..utils import ModelBaseClass

SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = int(10 * SAMPLE_RATE)
with open("bacpipe/settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

BEATS_PRETRAINED_PATH_FT = (
    "bacpipe/model_checkpoints/naturelm_audio/BEATs_iter1_finetuned_on_AS2M_cpt1.pt"
)
BEATS_PRETRAINED_PATH_SSL = (
    "bacpipe/model_checkpoints/naturelm_audio/BEATs_iter3_plus_AS2M.pt"
)
BEATS_PRETRAINED_PATH_NATURELM = (
    "bacpipe/model_checkpoints/naturelm_audio/naturebeats.pt"
)


def universal_torch_load(f, **kwargs):
    with open(f, "rb") as opened_file:
        model = torch.load(f, **kwargs)
    return model


# class Model(ModelBase):
class InferenceModel:
    """Wrapper that adapts the raw *BEATs* backbone for our training loop.

    This module follows the same conventions as the other model wrappers
    (e.g. ``efficientnet.py``) so that it can be selected via
    ``representation_learning.models.get_model.get_model``.

    The underlying BEATs implementation operates directly on raw‐waveform
    inputs.  We therefore do *not* apply the optional :class:`AudioProcessor`
    from :pymeth:`ModelBase.process_audio` unless an ``audio_config`` is
    explicitly supplied.

    Notes
    -----
    1.  BEATs extracts a sequence of frame-level embeddings with dimension
        ``cfg.encoder_embed_dim`` (default: ``768``).  We convert this
        variable-length sequence into a fixed-dimensional vector via masked
        mean-pooling before feeding it to a linear classifier.
    2.  When ``return_features_only=True`` the classifier layer is skipped and
        the pooled embedding is returned directly, which is handy for
        representation extraction / linear probing.
    """

    def __init__(
        self,
        *,
        num_classes: int = None,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config=None,
        return_features_only: bool = False,
        use_naturelm: bool = False,
        fine_tuned: bool = False,
    ) -> None:

        # ------------------------------------------------------------------
        # 1.  Build the BEATs backbone
        # ------------------------------------------------------------------

        if fine_tuned:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_FT
        else:
            beats_checkpoint_path = BEATS_PRETRAINED_PATH_SSL

        beats_ckpt = universal_torch_load(
            beats_checkpoint_path, map_location="cpu", weights_only=True
        )
        self.use_naturelm = use_naturelm
        self.fine_tuned = fine_tuned
        beats_cfg = BEATsConfig(beats_ckpt["cfg"])
        print(beats_cfg)
        if use_naturelm:  # BEATs-NatureLM has no config, load from regular ckpt first.
            beats_ckpt_naturelm = universal_torch_load(
                BEATS_PRETRAINED_PATH_NATURELM, map_location="cpu", weights_only=True
            )
        else:
            beats_ckpt_naturelm = beats_ckpt["model"]
        # beats_ckpt_naturelm = beats_ckpt
        self.backbone = BEATs(beats_cfg)
        self.backbone.to(settings["device"])
        self.backbone.load_state_dict(beats_ckpt_naturelm, strict=False)

        # ------------------------------------------------------------------
        # 2.  Optional classifier for supervised training
        # ------------------------------------------------------------------
        self._return_features_only = return_features_only
        if not return_features_only:
            self.classifier = nn.Linear(768, num_classes)
        else:
            # self.register_module("classifier", None)  # type: ignore[arg-type]
            self.classifier = None  # type: ignore[arg-type]

    # ----------------------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, padding_mask=None, dont_pool=False
    ) -> torch.Tensor:  # noqa: D401 – keep signature consistent
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw audio waveform with shape ``(batch, time)``.
        padding_mask : torch.Tensor, optional
            Boolean mask where *True* denotes padding elements.  Shape must be
            ``(batch, time)`` and match *x*.

        Returns
        -------
        torch.Tensor
            • When *return_features_only* is **False**: logits of shape
              ``(batch, num_classes)``
            • Otherwise: pooled embeddings of shape
              ``(batch, encoder_embed_dim)``
        """
        features, frame_padding = self.backbone(x, padding_mask)

        # features: (B, T', D)
        # frame_padding: (B, T') or None

        # ------------------------------------------------------------------
        # 3.  Masked mean-pooling over the temporal dimension
        # ------------------------------------------------------------------
        if frame_padding is not None and frame_padding.any():
            masked_features = features.clone()
            masked_features[frame_padding] = 0.0  # Zero-out padded frames
            valid_counts = (~frame_padding).sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_features.sum(dim=1) / valid_counts
        else:
            pooled = features.mean(dim=1)

        if dont_pool:
            return features
        elif self._return_features_only:
            return pooled
        else:
            return self.classifier(pooled)

    def extract_embeddings(
        self,
        x: torch.Tensor,
        padding_mask=None,
    ) -> torch.Tensor:
        self._return_features_only = True
        if isinstance(x, dict):
            return self.forward(x["raw_wav"], x["padding_mask"])
        else:
            return self.forward(x)

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        # audio = super().process_audio(x)
        if self.use_naturelm:
            audio = torch.clamp(x, -1.0, 1.0)
        return audio


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES)
        self.model = InferenceModel(
            fine_tuned=False, use_naturelm=True, return_features_only=True
        )

    def preprocess(self, audio):
        audio = audio.cpu()
        return self.model.process_audio(audio)

    def __call__(self, input):
        features = self.model.extract_embeddings(input)
        return features
