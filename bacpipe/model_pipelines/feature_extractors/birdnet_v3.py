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
        super().__init__(sr=sr, segment_length=segment_length, **kwargs)
        
        label_path = self.model_utils_base_path / 'birdnet_v3/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv'
        checkpoint_path=(
            self.model_base_path / 'birdnet_v3' / 'model.onnx'
            )
        self.model = birdnet_v3_ONNX(checkpoint_path, device=self.device)
        self.classes = pd.read_csv(label_path, sep=';')['com_name'].values
        
    def preprocess(self, audio):
        return audio

    def __call__(self, input):
        input = input.cpu()
        self.predictions, self.embeddings = self.model(np.array(input))

        return self.embeddings

    def classifier_predictions(self, embeddings):
        return self.predictions



class birdnet_v3_ONNX(nn.Module):
    """ONNX Model Wrapper with multi-platform GPU acceleration.
    Adapted from https://huggingface.co/justinchuby/Perch-onnx.
    
    Supports: Linux (CUDA/CPU), macOS (CoreML/CPU), Windows (CUDA/DirectML/CPU).
    Input: Audio tensor of shape (batch_size, 160000) at 32kHz sample rate.
    """

    def __init__(self, checkpoint_path, device: str = "auto"):
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
        
        Args:
            session: ONNX Runtime inference session.
            chunks: [N, T] float32 mono audio.
            batch_size: batch size.
            return_embeddings: if True, also return stacked embeddings [N, D].
        
        Returns:
            predictions: [N, C] float32
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