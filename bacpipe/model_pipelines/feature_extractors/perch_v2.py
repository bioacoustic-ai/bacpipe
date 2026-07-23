import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from huggingface_hub import hf_hub_download

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("bacpipe")

# tf.keras.backend.clear_session()

from ..model_utils import ModelBaseClass

SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = 160000


class Model(ModelBaseClass):
    def __init__(
        self,
        sr=SAMPLE_RATE,
        segment_length=LENGTH_IN_SAMPLES,
        **kwargs,
    ):
        super().__init__(sr=sr, segment_length=segment_length, **kwargs)

        self.model = PerchV2ONNX()
        self.classes = self.model.classes
        
    def preprocess(self, audio):
        return audio

    def __call__(self, input):
        self.results = self.model(input)

        return self.results['embeddings']

    def classifier_predictions(self, embeddings):
        inferece_results = self.results['logits']
        return inferece_results



class PerchV2ONNX(nn.Module):
    """Perch v2 ONNX Model Wrapper with multi-platform GPU acceleration.
    
    Supports: Linux (CUDA/CPU), macOS (CoreML/CPU), Windows (CUDA/DirectML/CPU).
    Input: Audio tensor of shape (batch_size, 160000) at 32kHz sample rate.
    """

    def __init__(self, device: str = "auto", load_labels: bool = True):
        super().__init__()
        
        # 1. Download model weights
        model_path = hf_hub_download(
            repo_id="justinchuby/Perch-onnx",
            filename="perch_v2_no_dft.onnx",
        )
        
        # 2. Download taxonomy labels file (14,795 species)
        self.classes = []
        if load_labels:
            try:
                label_path = hf_hub_download(
                    repo_id="tphakala/Perch-v2",
                    filename="labels.txt",
                )
                with open(label_path, "r", encoding="utf-8") as f:
                    self.classes = [line.strip() for line in f.readlines()][1:]
            except Exception as e:
                print(f"Notice: Could not load labels.txt ({e}). Species names will not be mapped.")

        # 3. Configure execution providers
        providers = self._get_execution_providers(device)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )
        
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        
        print(f"PerchV2 initialized using providers: {self.active_providers}")

    def _get_execution_providers(self, requested_device: str) -> list[str]:
        available = ort.get_available_providers()
        providers = []

        if requested_device.lower() in ("gpu", "cuda", "auto"):
            if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
            elif "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            elif "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")

        providers.append("CPUExecutionProvider")
        return providers

    def forward(self, x: torch.Tensor, return_probabilities: bool = True) -> dict[str, torch.Tensor]:
        """Runs inference on input audio tensor.
        
        Returns dict containing:
          - 'embedding': (batch, 1536)
          - 'spatial_embedding': (batch, 16, 4, 1536)
          - 'spectrogram': (batch, 500, 128)
          - 'logits': (batch, 14795)
          - 'probabilities': (batch, 14795) [optional]
        """
        target_device = x.device
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        x_np = x.detach().cpu().numpy().astype(np.float32)

        # ONNX outputs: [0: embedding, 1: spatial_embedding, 2: spectrogram, 3: label]
        outputs = self.session.run(None, {self.input_name: x_np})

        logits = torch.from_numpy(outputs[3]).to(target_device)

        results = {
            "embeddings": torch.from_numpy(outputs[0]).to(target_device),
            "spatial_embedding": torch.from_numpy(outputs[1]).to(target_device),
            "spectrogram": torch.from_numpy(outputs[2]).to(target_device),
            "logits": logits,
        }

        if return_probabilities:
            results["probabilities"] = torch.sigmoid(logits)

        return results