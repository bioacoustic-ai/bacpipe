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

import tensorflow as tf

import collections
from ..utils import ModelBaseClass

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


class FBetaScore(tf.keras.metrics.Metric):
    def __init__(
        self,
        num_classes=1,
        average=None,
        beta=0.5,
        threshold=0.5,
        name="fbeta",
        dtype=tf.float32,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold

        # Must match variable names used in TFA version
        self.true_positives = self.add_weight(
            name="true_positives", shape=(num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", shape=(num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, self.dtype)
        y_true = tf.cast(y_true, self.dtype)

        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        beta_sq = self.beta**2
        precision = self.true_positives / (
            self.true_positives + self.false_positives + 1e-7
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + 1e-7
        )

        return (
            (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-7)
        )

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }


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
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        orig_model = tf.keras.models.load_model(
            self.model_base_path / "hbdet",
            custom_objects={"Addons>FBetaScore": FBetaScore},
        )
        model_list = orig_model.layers[:-2]
        model_list.insert(0, tf.keras.layers.Input([LENGTH_IN_SAMPLES]))
        model_list.insert(1, tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1)))
        model_list.insert(2, MelSpectrogram())
        self.model = tf.keras.Sequential(layers=[layer for layer in model_list])

    def preprocess(self, audio):
        return tf.convert_to_tensor(audio.cpu())

    def __call__(self, input):
        return self.model.predict(input)
