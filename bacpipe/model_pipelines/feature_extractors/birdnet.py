import tensorflow as tf
import keras
import pandas as pd
import numpy as np

SAMPLE_RATE = 48000
LENGTH_IN_SAMPLES = 144000

from ..model_utils import ModelBaseClass


class Model(ModelBaseClass):

    def __init__(self, **kwargs):
        """
        Initialize the BirdNET model.
        """
        super().__init__(
            sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs
        )
        if tf.__version__ == "2.15.1":
            self.model = tf.keras.models.load_model(
                self.model_base_path / "birdnet/birdnet_tf215", compile=False
            )
        else:
            self.model = tf.keras.models.load_model(
                self.model_base_path / "birdnet/birdnetv2.4.keras",
                compile=False,
            )

        loaded_preprocessor = tf.saved_model.load(
            self.model_base_path / "birdnet/BirdNET_Preprocessor",
        )
        self.preprocessor = lambda x: (
            loaded_preprocessor.signatures["serving_default"](x)["concatenate"]
        )

        all_classes = pd.read_csv(
            self.model_utils_base_path
            / "birdnet/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt",
            header=None,
        )
        self.classes = [s.split("_")[-1] for s in all_classes.values.squeeze()]

        self.embeds = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output,
            name="embeddings_model",
        )

        x = keras.Input(shape=self.model.layers[-3].output.shape[1:])
        y = self.model.layers[-2](x)
        y = self.model.layers[-1](y)
        self.classifier = tf.keras.Model(x, y, name="classifier_model")

    def preprocess(self, audio):
        """
        Preprocess the audio samples in batches of 511 frames.

        Parameters
        ----------
        audio : torch.Tensor
            audio samples to be preprocessed

        Returns
        -------
        tf.Tensor
            preprocessed audio
        """
        audio = audio.cpu()
        for idx in range(0, audio.shape[0], 511):
            if idx == 0:
                processed = self.preprocessor(
                    tf.convert_to_tensor(audio[:511], dtype=tf.float32)
                ).numpy()
            else:
                processed = np.vstack(
                    [
                        processed,
                        self.preprocessor(
                            tf.convert_to_tensor(
                                audio[idx : idx + 511], dtype=tf.float32
                            )
                        ).numpy(),
                    ]
                )
        return tf.convert_to_tensor(processed, dtype=tf.float32)

    def __call__(self, input):
        """
        Run the embeddings model on the input.

        Parameters
        ----------
        input : tf.Tensor
            preprocessed audio

        Returns
        -------
        tf.Tensor
            embeddings
        """
        return self.embeds(input, training=False)

    def classifier_predictions(self, embeddings):
        """
        Run the classifier head on the embeddings.

        Parameters
        ----------
        embeddings : tf.Tensor
            embeddings from the backbone

        Returns
        -------
        np.array
            sigmoid class logits
        """
        logits = self.classifier(embeddings).numpy()
        return tf.nn.sigmoid(logits).numpy()


class Rebuilder:
    def __init__(self, model):
        """
        Initialize the model rebuilder.

        Parameters
        ----------
        model : keras.Model
            model whose layers should be rebuilt from layer 4 onward
        """
        self.input_layer = tf.keras.Input(
            shape=model.layers[4].input.shape[1:],
            name="inputs",
            dtype=tf.float32,
        )
        # we're starting at layer 4 because the MelSpecLayerSimple classes wont be deserializable by keras 3.11
        # but we don't need them because they are just part of the preprocessing anyway
        # So essentially we're starting after the concatenation of the two spectrograms
        self.layer_map = {layer.name: layer for layer in model.layers[4:]}
        self.output_cache = {}  # cache outputs to stop infinite recursion
        self.layer_confs = []

    def rebuild_layer(self, layer):
        """
        Rebuild a layer and its dependencies into the new model graph.

        Parameters
        ----------
        layer : keras.layers.Layer
            layer to rebuild

        Returns
        -------
        keras.Tensor
            output tensor of the rebuilt layer
        """
        if layer.name in self.output_cache:
            return self.output_cache[layer.name]

        layer_config = {
            "name": layer.name,
            "class_name": layer.__class__.__name__,
            "config": layer.get_config(),
            "inbound_nodes": [],
        }

        if "axis" in layer_config["config"]:
            if not isinstance(layer_config["config"]["axis"], int):
                layer_config["config"]["axis"] = layer_config["config"][
                    "axis"
                ][0]

        # Handle multiple inputs
        inbound = []
        if isinstance(layer.input, list):
            inputs = []
            for inp in layer.input:
                inbound.append(inp._keras_history.layer.name)
                inputs.append(self.rebuild_layer(inp._keras_history.layer))
        else:
            inp = layer.input
            inbound.append(inp._keras_history.layer.name)
            if inp.name.startswith("concat") or inp.name.startswith("INPUT"):
                layer_config["inbound_nodes"] = inbound

                out = layer(self.input_layer)
                self.output_cache[layer.name] = out
                self.layer_confs.append(layer_config)
                return out
            inputs = [self.rebuild_layer(inp._keras_history.layer)]
        layer_config["inbound_nodes"] = inbound

        out = layer(inputs if len(inputs) > 1 else inputs[0])
        self.output_cache[layer.name] = out
        self.layer_confs.append(layer_config)
        return out

    def build_model(self, model):
        """
        Build the rebuilt model from the input layer to the final output.

        Parameters
        ----------
        model : keras.Model
            original model to rebuild

        Returns
        -------
        tuple
            (layer configuration dict, rebuilt keras.Model)
        """
        # Recurse from final output layer
        output = self.rebuild_layer(model.layers[-1])
        return (
            {
                "input_shape": self.input_layer.shape,
                "layer_confs": self.layer_confs,
            },
            keras.Model(self.input_layer, output, name="rebuilt_model"),
        )
