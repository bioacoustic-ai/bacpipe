import torch, numpy as np

def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq,
                          norm=True, crop=False):
    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 7000.0)
    max_mel = 1127 * np.log1p(max_freq / 7000.0)
    peaks_mel = torch.linspace(min_mel, max_mel, num_bands + 2)
    peaks_hz = 7000 * (torch.expm1(peaks_mel / 1127))
    peaks_bin = peaks_hz * frame_len / sample_rate

    # create filterbank
    input_bins = (frame_len // 2) + 1
    if crop:
        input_bins = min(input_bins,
                         int(np.ceil(max_freq * frame_len /
                                     float(sample_rate))))
    x = torch.arange(input_bins, dtype=peaks_bin.dtype)[:, np.newaxis]
    l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
    tri_left = (x - l) / (c - l)
    tri_right = (x - r) / (c - r)
    tri = torch.min(tri_left, tri_right)
    filterbank = torch.clamp(tri, min=0)
    if norm:
        filterbank /= filterbank.sum(0)
    return filterbank



class MelFilter(torch.nn.Module):
    def __init__(self, sample_rate, winsize, num_bands, min_freq, max_freq):
        super(MelFilter, self).__init__()
        melbank = create_mel_filterbank(sample_rate, winsize, num_bands,
                                        min_freq, max_freq, crop=True)
        self.register_buffer('bank', melbank)

    def forward(self, x):
        x = x.transpose(-1, -2)  # put fft bands last
        x = x[..., :self.bank.shape[0]]  # remove unneeded fft bands
        x = x.matmul(self.bank)  # turn fft bands into mel bands
        x = x.transpose(-1, -2)  # put time last
        return x

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        result = super(MelFilter, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # remove all buffers; we use them as cached constants
        for k in self._buffers:
            del result[prefix + k]
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # ignore stored buffers for backwards compatibility
        for k in self._buffers:
            state_dict.pop(prefix + k, None)
        # temporarily hide the buffers; we do not want to restore them
        buffers = self._buffers
        self._buffers = {}
        result = super(MelFilter, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result

class STFT(torch.nn.Module):
    def __init__(self, winsize, hopsize, complex=False):
        super(STFT, self).__init__()
        self.winsize = winsize
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(winsize, periodic=False))
        self.complex = complex

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        result = super(STFT, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # remove all buffers; we use them as cached constants
        for k in self._buffers:
            del result[prefix + k]
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # ignore stored buffers for backwards compatibility
        for k in self._buffers:
            state_dict.pop(prefix + k, None)
        # temporarily hide the buffers; we do not want to restore them
        buffers = self._buffers
        self._buffers = {}
        result = super(STFT, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result

    def forward(self, x):
        x = x.unsqueeze(1)
        # we want each channel to be treated separately, so we mash
        # up the channels and batch size and split them up afterwards
        batchsize, channels = x.shape[:2]
        x = x.reshape((-1,) + x.shape[2:])
        # we apply the STFT
        x = torch.stft(x, self.winsize, self.hopsize, window=self.window,
                       center=False, return_complex=False)
        # we compute magnitudes, if requested
        if not self.complex:
            x = x.norm(p=2, dim=-1)
        # restore original batchsize and channels in case we mashed them
        x = x.reshape((batchsize, channels, -1) + x.shape[2:]) #if channels > 1 else x.reshape((batchsize, -1) + x.shape[2:])
        return x

class MedFilt(torch.nn.Module):
    """
    Withdraw the median of each frequency band
    """
    def __init__(self):
        super(MedFilt, self).__init__()
    def forward(self, x):
        return x - torch.quantile(x, 0.2, dim=-1, keepdim=True)[0]

class Log1p(torch.nn.Module):
    """
    Applies log(1 + 10**a * x), with scale fixed or trainable.
    """
    def __init__(self, a=0, trainable=False):
        super(Log1p, self).__init__()
        if trainable:
            a = torch.nn.Parameter(torch.tensor(a, dtype=torch.get_default_dtype()))
        self.a = a
        self.trainable = trainable

    def forward(self, x):
        if self.trainable or self.a != 0:
            x = torch.log1p(10 ** self.a * x)
        return x

    def extra_repr(self):
        return 'trainable={}'.format(repr(self.trainable))

class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

class Croper2D(torch.nn.Module):
    def __init__(self, *shape):
        super(Croper2D, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x[:,:,:self.shape[0],(x.shape[-1] - self.shape[1])//2:-(x.shape[-1] - self.shape[1])//2]

frontend_medfilt = lambda sr, nfft, sampleDur, n_mel : torch.nn.Sequential(
  STFT(nfft, int((sampleDur*sr - nfft)/128)),
  MelFilter(sr, nfft, n_mel, sr//nfft, sr//2),
  Log1p(7, trainable=False),
  torch.nn.InstanceNorm2d(1),
  MedFilt(),
  Croper2D(n_mel, 128)
)

sparrow_encoder = lambda nfeat, shape : torch.nn.Sequential(
  torch.nn.Conv2d(1, 32, 3, stride=2, bias=False, padding=(1)),
  torch.nn.BatchNorm2d(32),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1),
  torch.nn.BatchNorm2d(64),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(64, 128, 3, stride=2, bias=False, padding=1),
  torch.nn.BatchNorm2d(128),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(128, 256, 3, stride=2, bias=False, padding=1),
  torch.nn.BatchNorm2d(256),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(256, nfeat, 3, stride=2, padding=1),
  Reshape(nfeat * shape[0] * shape[1])
)

sparrow_decoder = lambda nfeat, shape : torch.nn.Sequential(
  Reshape(nfeat//(shape[0]*shape[1]), *shape),
  torch.nn.ReLU(True),

  torch.nn.Upsample(scale_factor=2),
  torch.nn.Conv2d(nfeat//(shape[0]*shape[1]), 256, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(256),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(256, 256, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(256),
  torch.nn.ReLU(True),

  torch.nn.Upsample(scale_factor=2),
  torch.nn.Conv2d(256, 128, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(128),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(128, 128, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(128),
  torch.nn.ReLU(True),

  torch.nn.Upsample(scale_factor=2),
  torch.nn.Conv2d(128, 64, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(64),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(64, 64, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(64),
  torch.nn.ReLU(True),

  torch.nn.Upsample(scale_factor=2),
  torch.nn.Conv2d(64, 32, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(32),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(32),
  torch.nn.ReLU(True),

  torch.nn.Upsample(scale_factor=2),
  torch.nn.Conv2d(32, 1, (3, 3), bias=False, padding=1),
  torch.nn.BatchNorm2d(1),
  torch.nn.ReLU(True),
  torch.nn.Conv2d(1, 1, (3, 3), bias=False, padding=1),
)