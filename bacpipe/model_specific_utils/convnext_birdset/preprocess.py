### Taken from https://huggingface.co/DBD-research-group/ConvNeXT-Base-BirdSet-XCL

import torch
import torchaudio
from torchvision import transforms
import torchaudio



class PowerToDB(torch.nn.Module):
    """
    A power spectrogram to decibel conversion layer. See birdset.datamodule.components.augmentations
    """

    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0, device='cpu'):
        super(PowerToDB, self).__init__()
        # Initialize parameters
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.device = device

    def forward(self, S):
        # Convert S to a PyTorch tensor if it is not already
        S = torch.as_tensor(S, dtype=torch.float32)

        if self.amin <= 0:
            raise ValueError("amin must be strictly positive")

        if torch.is_complex(S):
            magnitude = S.abs()
        else:
            magnitude = S

        # Check if ref is a callable function or a scalar
        if callable(self.ref):
            ref_value = self.ref(magnitude)
        else:
            ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

        # Compute the log spectrogram
        log_spec = 10.0 * torch.log10(
            torch.maximum(magnitude, torch.tensor(self.amin, device=self.device))
        )
        log_spec -= 10.0 * torch.log10(
            torch.maximum(ref_value, torch.tensor(self.amin, device=self.device))
        )

        # Apply top_db threshold if necessary
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

        return log_spec



class ConvNextPreProcess:
    def __init__(self, sample_rate, device):
        
        # Initialize the transformations
        device='cpu'

        self.spectrogram_converter = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=320, power=2.0
        )
        self.mel_converter = torchaudio.transforms.MelScale(
            n_mels=128, n_stft=513, sample_rate=sample_rate
        )
        self.normalizer = transforms.Normalize((-4.268,), (4.569,))
        self.powerToDB = PowerToDB(top_db=80, device=device)


    def preprocess(self, audio):
        """
        Preprocess the audio to the format that the model expects
        - Resample to 32kHz
        - Convert to melscale spectrogram n_fft: 1024, hop_length: 320, power: 2. melscale: n_mels: 128, n_stft: 513
        - Normalize the melscale spectrogram with mean: -4.268, std: 4.569 (from AudioSet)

        """
        # convert waveform to spectrogram
        spectrogram = self.spectrogram_converter(audio.cpu())
        spectrogram = spectrogram.to(torch.float32)
        melspec = self.mel_converter(spectrogram)
        dbscale = self.powerToDB(melspec)
        normalized_dbscale = self.normalizer(dbscale)
        # add dimension 3 from left
        normalized_dbscale = normalized_dbscale.unsqueeze(-3)
        return normalized_dbscale

