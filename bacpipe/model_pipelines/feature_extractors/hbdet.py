import tensorflow as tf

from ..model_utils import ModelBaseClass

SAMPLE_RATE = 2000
LENGTH_IN_SAMPLES = 7755


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        """
        Initialize the HBDet model.
        """
        super().__init__(
            sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
        )
        loaded_model = tf.saved_model.load(self.model_base_path / "hbdet")
        self.model = loaded_model.signatures["serving_default"]

    def preprocess(self, audio):
        """
        Convert the audio tensor to a TensorFlow tensor.

        Parameters
        ----------
        audio : torch.Tensor
            audio samples to be preprocessed

        Returns
        -------
        tf.Tensor
            audio tensor
        """
        return tf.convert_to_tensor(audio.cpu())

    def __call__(self, input):
        """
        Run the model on the input audio.

        Parameters
        ----------
        input : tf.Tensor
            preprocessed audio

        Returns
        -------
        tf.Tensor
            pooled model output
        """
        return self.model(input)["pool"]
