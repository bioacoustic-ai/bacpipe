import torch
from types import SimpleNamespace
import yaml
import torch
import timm
import torch.nn as nn
import torchaudio as ta
from ..utils import ModelBaseClass

SAMPLE_RATE = 44100
LENGTH_IN_SAMPLES = int(5.5 * SAMPLE_RATE)


class SpectrogramCNN(nn.Module):
    def __init__(self, cfg, init_backbone=True):
        """
        Pytorch network class containing the transformation from waveform to
        mel spectrogram, as well as the forward pass through a CNN backbone.

        Data augmentation like mixup or masked frequency or time can also be
        applied here.

        Parameters
        ----------
        cfg: SimpleNameSpace containing all configurations
        init_backbone: bool (Default=True). Whether to download and initialize the backbone.
                       Not always necessary when debugging.
        """
        super(SpectrogramCNN, self).__init__()
        self.cfg = cfg
        # for k, v in self.cfg.items():
        #     setattr(self.cfg, k, v)
        self.n_classes = cfg.n_classes

        # Initializes the transformation from waveform to mel spectrogram
        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            n_mels=cfg.n_mels,
            power=cfg.power,
            normalized=cfg.mel_normalized,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.wav2timefreq = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)

        if init_backbone:
            # Initialize pre-trained CNN
            # Input and output layers are automatically adjusted
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=cfg.pretrained,
                num_classes=cfg.n_classes,
                in_chans=cfg.in_chans,
            )


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        with open(f"{self.model_base_path}/insect66/config_insecteffnet.yaml", "rt") as infp:
            cfg = SimpleNamespace(**yaml.safe_load(infp))

        checkpoint = torch.jit.load(f"{self.model_base_path}/insect66/model_traced.pt")
        state_dict = checkpoint.state_dict()
        for k in ["wav2img.0.spectrogram.window", "wav2img.0.mel_scale.fb"]:
            state_dict[k.replace("wav2img", "wav2timefreq")] = state_dict.pop(k)

        self.model = SpectrogramCNN(cfg)
        self.model.load_state_dict(state_dict)

    def preprocess(self, audio):
        audio = audio[:, None, :]

        # (bs, channel, mel, time)
        return self.model.wav2timefreq(audio)

    @torch.inference_mode()
    def __call__(self, input):
        self.model.block_features = self.model.backbone.blocks(
            self.model.backbone.bn1(self.model.backbone.conv_stem(input))
        )

        self.model.embeddings = self.model.backbone.global_pool(
            self.model.backbone.bn2(
                self.model.backbone.conv_head(self.model.block_features)
            )
        )

        return self.model.embeddings
