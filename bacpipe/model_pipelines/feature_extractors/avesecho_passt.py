from hear21passt.base import get_basic_model, get_model_passt
from ..model_utils import ModelBaseClass
import torch.nn as nn
import torchaudio
import torch
import yaml
import pandas as pd

SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = int(3 * SAMPLE_RATE)  # burooj used 3 seconds
# LENGTH_IN_SAMPLES = int(10 * SAMPLE_RATE)

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization


class AugmentMelSTFT(nn.Module):
    def __init__(
        self,
        n_mels=128,
        sr=32000,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        freqm=48,
        timem=192,
        htk=False,
        fmin=0.0,
        fmax=None,
        norm=1,
        fmin_aug_range=1,
        fmax_aug_range=1000,
    ):
        """
        Initialize the augmentable mel STFT module.

        Parameters
        ----------
        n_mels : int
            number of mel bins
        sr : int
            sample rate
        win_length : int
            window length of the STFT
        hopsize : int
            hop size of the STFT
        n_fft : int
            FFT size
        freqm : int
            frequency mask width for augmentation (unused for eval)
        timem : int
            time mask width for augmentation (unused for eval)
        htk : bool
            whether to use the HTK mel scale
        fmin : float
            minimum frequency of the mel filterbank
        fmax : float or None
            maximum frequency of the mel filterbank
        norm : int
            mel filterbank normalization type
        fmin_aug_range : int
            fmin augmentation range
        fmax_aug_range : int
            fmax augmentation range
        """
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer(
            "window",
            torch.hann_window(win_length, periodic=False),
            persistent=False,
        )
        assert (
            fmin_aug_range >= 1
        ), f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert (
            fmin_aug_range >= 1
        ), f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer(
            "preemphasis_coefficient",
            torch.as_tensor([[[-0.97, 1]]]),
            persistent=False,
        )

    def forward(self, x):
        """
        Convert the input waveform to a normalized mel spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            input waveform

        Returns
        -------
        torch.Tensor
            normalized mel spectrogram
        """
        x = nn.functional.conv1d(
            x.unsqueeze(1), self.preemphasis_coefficient
        ).squeeze(1)
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=False,
        )
        x = (x**2).sum(dim=-1)  # power mag
        fmin = self.fmin  # + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = (
            self.fmax
            + self.fmax_aug_range // 2
            - 500
            # - torch.randint(self.fmax_aug_range, (1,)).item()
        )
        # don't augment eval data

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sr,
            fmin,
            fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(
                mel_basis, (0, 1), mode="constant", value=0
            ),
            device=x.device,
        )
        with torch.amp.autocast("cuda", enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec

    def extra_repr(self):
        """
        Return the extra string representation of the module.
        """
        return "winsize={}, hopsize={}".format(self.win_length, self.hopsize)


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        """
        Initialize the AvesEcho PaSST model.
        """
        super().__init__(
            sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
        )
        self.model = get_basic_model(
            mode="logits", arch="passt_s_kd_p16_128_ap486"
        )
        self.model.net = get_model_passt(
            arch="passt_s_kd_p16_128_ap486", n_classes=585
        )
        self.model.net.to(self.device)
        ckpnt = torch.load(
            self.model_base_path / "avesecho_passt/best_model_passt.pt",
            weights_only=True,
            map_location=torch.device(self.device),
        )
        self.model.load_state_dict(ckpnt)
        self.model = self.model.to(self.device)
        self.preprocessor = AugmentMelSTFT(
            n_mels=128, sr=SAMPLE_RATE, win_length=800, hopsize=320, n_fft=1024
        )
        self.preprocessor = self.preprocessor.to(self.device)
        class_file_path = (
            self.model_utils_base_path / "avesecho_passt/list_AvesEcho.csv"
        )
        df = pd.read_csv(class_file_path, header=None)
        self.classes = df.iloc[:, 1].values.tolist()

    def preprocess(self, audio):
        """
        Convert the audio samples to a mel spectrogram.

        Parameters
        ----------
        audio : torch.Tensor
            audio samples to be preprocessed

        Returns
        -------
        torch.Tensor
            mel spectrogram
        """
        audio = audio.to(self.device)
        return self.preprocessor(audio)

    @torch.inference_mode()
    def __call__(self, input):
        """
        Run the model on the input spectrogram.

        Parameters
        ----------
        input : torch.Tensor
            input mel spectrogram

        Returns
        -------
        torch.Tensor
            model features
        """
        self.logits, features = self.model.net(input.unsqueeze(1))
        return features

    def classifier_predictions(self, embeddings):
        """
        Return the sigmoid class probabilities from the last call.

        Parameters
        ----------
        embeddings : torch.Tensor
            embeddings from the last model call (unused)

        Returns
        -------
        torch.Tensor
            sigmoid class probabilities
        """
        probs = torch.sigmoid(self.logits)
        return probs
