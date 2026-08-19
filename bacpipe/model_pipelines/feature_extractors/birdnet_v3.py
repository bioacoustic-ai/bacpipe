import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("bacpipe")

from ..model_utils import ModelBaseClass

SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = 96000

class Model(ModelBaseClass):
    def __init__(
        self,
        sr=SAMPLE_RATE,
        segment_length=LENGTH_IN_SAMPLES,
        **kwargs,
    ):
        """
        Initialize the BirdNET v3 model wrapper.

        Parameters
        ----------
        sr : int
            sample rate in Hz
        segment_length : int
            number of samples per input segment
        **kwargs
            additional keyword arguments passed to the base class
        """
        super().__init__(sr=sr, segment_length=segment_length, **kwargs)
        
        label_path = self.model_utils_base_path / 'birdnet_v3/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv'
        checkpoint_path=(
            self.model_base_path / 'birdnet_v3' / 'model.onnx'
            )
        self.model = birdnet_v3_ONNX(checkpoint_path, device=self.device)
        self.classes = pd.read_csv(label_path, sep=';')['com_name'].values
        
    def preprocess(self, audio):
        """
        Pass-through preprocessing.

        Parameters
        ----------
        audio : torch.Tensor
            input audio tensor

        Returns
        -------
        torch.Tensor
            the unchanged input audio
        """
        return audio

    def __call__(self, input):
        """
        Run the BirdNET v3 model on the input audio.

        Parameters
        ----------
        input : torch.Tensor
            input audio tensor

        Returns
        -------
        torch.Tensor
            embeddings produced by the model
        """
        input = input.cpu()
        self.predictions, self.embeddings = self.model(np.array(input))

        return self.embeddings

    def classifier_predictions(self, embeddings):
        """
        Return the class predictions from the last model call.

        Parameters
        ----------
        embeddings : torch.Tensor
            embeddings from the last model call (unused)

        Returns
        -------
        torch.Tensor
            class predictions produced by the model
        """
        return self.predictions



class birdnet_v3_ONNX(nn.Module):
    """ONNX Model Wrapper with multi-platform GPU acceleration.
    Adapted from https://huggingface.co/justinchuby/Perch-onnx.
    
    Supports: Linux (CUDA/CPU), macOS (CoreML/CPU), Windows (CUDA/DirectML/CPU).
    Input: Audio tensor of shape (batch_size, 160000) at 32kHz sample rate.
    """

    def __init__(self, checkpoint_path, device: str = "auto"):
        """
        Initialize the BirdNET v3 ONNX classifier.

        Parameters
        ----------
        checkpoint_path : pathlib.Path
            path to the ONNX checkpoint
        device : str
            requested device ("auto", "cpu", or "cuda")
        """
        super().__init__()
        
        try:
            providers = self._get_execution_providers(device)
            self.session = ort.InferenceSession(checkpoint_path, providers=providers)
            # Report actual provider used
            actual_provider = self.session.get_providers()[0] if self.session.get_providers() else "unknown"
            print(f"ONNX provider: {actual_provider}")
        except Exception as e:
            print(f"Error loading ONNX model: {e}", file=sys.stderr)
            sys.exit(1)

    def _get_execution_providers(self, device: str) -> list[str]:
        """
        Resolve the ONNX execution providers for the requested device.

        Parameters
        ----------
        device : str
            requested device ("auto", "cpu", or "cuda")

        Returns
        -------
        list of str
            execution providers to use for the ONNX session
        """
        providers = []

        # Select execution provider based on device
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        providers.append("CPUExecutionProvider")
    

    def forward(
        self,# session: "ort.InferenceSession",
        chunks: np.ndarray,
        batch_size: int = 16,
        return_embeddings: bool = True,
    ):
        """
        Run inference with ONNX model.

        Parameters
        ----------
        chunks : np.ndarray
            [N, T] float32 mono audio
        batch_size : int, optional
            batch size, by default 16
        return_embeddings : bool, optional
            if True, also return stacked embeddings [N, D], by default True

        Returns
        -------
        np.ndarray
            predictions: [N, C] float32
        np.ndarray or None
            embeddings: [N, D] float32 or None
        """
        if chunks.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32), None
        
        # Get input/output info
        input_name = self.session.get_inputs()[0].name
        input_type = self.session.get_inputs()[0].type
        output_names = [o.name for o in self.session.get_outputs()]
        
        # Determine input dtype (handle FP16 models)
        if "float16" in input_type:
            input_dtype = np.float16
        else:
            input_dtype = np.float32
        
        preds_out = []
        embs_out = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size].astype(input_dtype)
            outputs = self.session.run(output_names, {input_name: batch})
            
            # Model outputs: predictions, embeddings (two outputs) or just predictions
            if len(outputs) == 2:
                pred, emb = outputs
                if return_embeddings:
                    embs_out.append(emb.astype(np.float32))
            else:
                pred = outputs[0]
            
            preds_out.append(pred.astype(np.float32))
        
        predictions = torch.tensor(np.concatenate(preds_out, axis=0))
        embeddings = torch.tensor(np.concatenate(embs_out, axis=0))
        return predictions, embeddings