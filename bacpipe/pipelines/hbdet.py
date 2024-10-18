# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
import librosa as lb
from tensorflow_addons import metrics
import collections
from .utils import ModelBaseClass

SAMPLE_RATE = 2000
LENGTH_IN_SAMPLES = 7755

STFT_FRAME_LENGTH = 1024
FFT_HOP = 53


Config = collections.namedtuple(
    "Config",
    [
        "stft_frame_length",
        "stft_frame_step",
        "freq_bins",
        "sample_rate",
        "lower_f",
        "upper_f",
    ],
)

Config.__new__.__defaults__ = (
    STFT_FRAME_LENGTH,
    FFT_HOP,
    64,
    SAMPLE_RATE,
    0.0,
    SAMPLE_RATE / 2,
)


class MelSpectrogram(tf.keras.layers.Layer):
    """Keras layer that converts a waveform to an amplitude mel spectrogram."""

    def __init__(self, config=None, name="mel_spectrogram"):
        super(MelSpectrogram, self).__init__(name=name)
        if config is None:
            config = Config()
        self.config = config

    def get_config(self):
        config = super().get_config()
        config.update({key: val for key, val in config.items()})
        return config

    def build(self, input_shape):
        self._stft = tf.keras.layers.Lambda(
            lambda t: tf.signal.stft(
                tf.squeeze(t, 2),
                frame_length=self.config.stft_frame_length,
                frame_step=self.config.stft_frame_step,
            ),
            name="stft",
        )
        num_spectrogram_bins = self._stft.compute_output_shape(input_shape)[-1]
        self._bin = tf.keras.layers.Lambda(
            lambda t: tf.square(
                tf.tensordot(
                    tf.abs(t),
                    tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins=self.config.freq_bins,
                        num_spectrogram_bins=num_spectrogram_bins,
                        sample_rate=self.config.sample_rate,
                        lower_edge_hertz=self.config.lower_f,
                        upper_edge_hertz=self.config.upper_f,
                        name="matrix",
                    ),
                    1,
                )
            ),
            name="mel_bins",
        )

    def call(self, inputs):
        return self._bin(self._stft(inputs))

class Model(ModelBaseClass):
    def __init__(self):
        orig_model = tf.keras.models.load_model('ievad/models/hbdet',
                custom_objects={"FBetaScote": metrics.FBetaScore},
        )
        model_list = orig_model.layers[:-2]
        model_list.insert(0, tf.keras.layers.Input([LENGTH_IN_SAMPLES]))
        model_list.insert(
            1, tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1))
        )
        model_list.insert(2, MelSpectrogram())
        self.model = tf.keras.Sequential(
            layers=[layer for layer in model_list]
        )
    
    def preprocess(self, audio):
        re_audio = lb.resample(audio, orig_sr = self.config['sr'], 
                               target_sr = SAMPLE_RATE)
        num = np.ceil(len(re_audio) / LENGTH_IN_SAMPLES)
        # zero pad in case the end is reached
        re_audio = [*re_audio, 
                    *np.zeros([int(num * LENGTH_IN_SAMPLES - len(re_audio))])]
        wins = np.array(re_audio).reshape([int(num), LENGTH_IN_SAMPLES])

        return tf.convert_to_tensor(wins)
    
    def __call__(self, input):
        return self.model.predict(input)