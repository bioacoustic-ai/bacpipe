# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This initializes the nn module
"""
from bacpipe.model_utils.animal2vec_nn.nn.sinc import SincConv
from bacpipe.model_utils.animal2vec_nn.nn.modalities.modality import Modality
from bacpipe.model_utils.animal2vec_nn.nn.utils import (
    get_conv_size,
    plot_confusion_matrices,
    rename_attribute,
    log_metrics,
    to_2tuple,
    get_2d_sincos_pos_embed,
    fuse_to_segmented_predictions,
    FusedSegmentationMixin,
    confusion,
    sigmoid_focal_loss,
    ConcatTensorMeter,
    ConvFeatureExtractionModel,
    pad_left_right,
    get_padding_value,
    chunk_and_normalize
)

from bacpipe.model_utils.animal2vec_nn.nn.audio_tasks import (
    AudioTaskCCAS
)
from bacpipe.model_utils.animal2vec_nn.nn.wav2vec2 import (
    Wav2VecCcasFinetune,
)
from bacpipe.model_utils.animal2vec_nn.nn.audio_train_routine import (
    animal2vec_audio_main
)
from bacpipe.model_utils.animal2vec_nn.nn.criterions import (
    FinetuneCrossEntropyCriterion,
    ExpandedModelCriterion
)
from bacpipe.model_utils.animal2vec_nn.nn.modalities.base import (
    MaskSeed,
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_annealed_rate,
)
from bacpipe.model_utils.animal2vec_nn.nn.modalities.modules import (
    D2vDecoderConfig,
    AltBlock,
    Decoder1d,
    FixedPositionalEncoder
)

from bacpipe.model_utils.animal2vec_nn.nn.modalities.audio import (
    D2vAudioConfig,
    AudioEncoder,
)

from bacpipe.model_utils.animal2vec_nn.nn.modalities.images import (
    D2vImageConfig,
    ImageEncoder,
)

from bacpipe.model_utils.animal2vec_nn.nn.data2vec2 import (
    Data2VecMultiModel
)
