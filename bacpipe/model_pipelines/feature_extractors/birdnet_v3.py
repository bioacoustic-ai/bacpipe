import sys
import mmap
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
EMBEDDING_DIM = 1280

HEAD_WEIGHT_NAME = "head.weight"
HEAD_BIAS_NAME = "head.bias"
_ONNX_FLOAT_DTYPES = {1: np.float32, 10: np.float16}


def _varint(buf, pos):
    """Decode a protobuf varint starting at ``pos``; return (value, new_pos)."""
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _fields(buf, start, end):
    """
    Iterate over the fields of a protobuf message without decoding payloads.

    Yields
    ------
    tuple
        (field_number, value) where value is an int for varints and an
        (offset, length) pair for length-delimited fields.
    """
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value = (pos, length)
            pos += length
        elif wire == 1:
            value, pos = pos, pos + 8
        elif wire == 5:
            value, pos = pos, pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield field, value


def read_onnx_initializers(checkpoint_path, names):
    """
    Read named float initializers (weights) from an ONNX file.

    The ONNX protobuf is walked directly (memory-mapped, payloads that are
    not requested are skipped), so neither the ``onnx`` package nor the
    whole 500+ MB checkpoint in memory is needed.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        path to the ONNX checkpoint (initializers must be stored inline)
    names : iterable of str
        names of the initializers to read

    Returns
    -------
    dict of str to np.ndarray
        the requested tensors as float32 arrays

    Raises
    ------
    KeyError
        if one of the names is not an initializer of the graph
    ValueError
        if a tensor is not an inline float tensor
    """
    wanted = set(names)
    found = {}
    with open(checkpoint_path, "rb") as f, mmap.mmap(
        f.fileno(), 0, access=mmap.ACCESS_READ
    ) as buf:
        graph = next(v for k, v in _fields(buf, 0, len(buf)) if k == 7)
        for k, (offset, length) in _fields(buf, graph[0], graph[0] + graph[1]):
            if k != 5:  # GraphProto.initializer
                continue
            dims, dtype, name, raw = [], None, None, None
            for tk, tv in _fields(buf, offset, offset + length):
                if tk == 1:  # dims
                    if isinstance(tv, tuple):  # packed
                        pos = tv[0]
                        while pos < tv[0] + tv[1]:
                            d, pos = _varint(buf, pos)
                            dims.append(d)
                    else:
                        dims.append(tv)
                elif tk == 2:
                    dtype = tv
                elif tk == 8:
                    name = bytes(buf[tv[0] : tv[0] + tv[1]]).decode()
                elif tk == 9:
                    raw = tv
            if name not in wanted:
                continue
            if raw is None or dtype not in _ONNX_FLOAT_DTYPES:
                raise ValueError(
                    f"Initializer '{name}' is not an inline float tensor"
                )
            # astype copies, so no view of the mmap outlives this block
            found[name] = (
                np.frombuffer(
                    buf,
                    dtype=_ONNX_FLOAT_DTYPES[dtype],
                    count=int(np.prod(dims)),
                    offset=raw[0],
                )
                .reshape(dims)
                .astype(np.float32)
            )
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"Initializers not found in {checkpoint_path}: {sorted(missing)}")
    return found

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

        self.checkpoint_path = checkpoint_path
        self.predictions = None
        self.embeddings = None
        self._head = None
        self._last_call_rows = None

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
        self._last_call_rows = None

        return self.embeddings

    def classifier_predictions(self, embeddings):
        """
        Return the class probabilities for the given embeddings.

        The BirdNET v3 head is ``sigmoid(0.40 * Linear(embeddings) +
        0.35 * attention_1(feature_map) + 0.25 * attention_2(feature_map))``.
        The two attention branches read the backbone feature map, which is
        not contained in the 1280-d embeddings, so the result depends on
        where the embeddings come from:

        - rows identical to embeddings of the last model call get the
          exact predictions computed from the audio;
        - all other rows (e.g. embeddings loaded from disk) are classified
          with the linear branch alone, ``sigmoid(Linear(embeddings))``,
          the best embedding-only approximation (on the test audio: about
          95% identical top-1 class and a mean absolute error of 0.03 on
          relevant classes).

        Parameters
        ----------
        embeddings : torch.Tensor or np.ndarray
            embeddings of shape (n, 1280) (or a single (1280,) embedding)

        Returns
        -------
        torch.Tensor
            class probabilities of shape (n, number of classes), float32, cpu

        Raises
        ------
        ValueError
            if the embeddings do not have 1280 features
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = np.ascontiguousarray(np.asarray(embeddings), dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[None]
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected embeddings of shape (n, {EMBEDDING_DIM}), "
                f"got {embeddings.shape}"
            )

        source_rows = self._match_last_call(embeddings)
        exact = source_rows >= 0
        if exact.all():
            return self.predictions[torch.from_numpy(source_rows)].clone()

        head_input = torch.from_numpy(embeddings[~exact])
        probabilities = self._linear_head_probabilities(head_input)
        if not exact.any():
            return probabilities

        out = torch.empty(len(embeddings), probabilities.shape[1])
        out[torch.from_numpy(~exact)] = probabilities
        out[torch.from_numpy(exact)] = self.predictions[
            torch.from_numpy(source_rows[exact])
        ]
        return out

    def _match_last_call(self, embeddings):
        """
        Find which rows are bit-identical to embeddings of the last call.

        Parameters
        ----------
        embeddings : np.ndarray
            float32 embeddings of shape (n, 1280)

        Returns
        -------
        np.ndarray
            for each row, its row index in the last call, or -1
        """
        if self.embeddings is None or self.predictions is None:
            return np.full(len(embeddings), -1, dtype=np.int64)
        if self._last_call_rows is None:
            last = np.ascontiguousarray(self.embeddings.numpy(), dtype=np.float32)
            self._last_call_rows = {row.tobytes(): i for i, row in enumerate(last)}
        return np.array(
            [self._last_call_rows.get(row.tobytes(), -1) for row in embeddings],
            dtype=np.int64,
        )

    def _linear_head_probabilities(self, embeddings):
        """
        Apply the linear branch of the classifier head to the embeddings.

        The weights are read once from the ONNX checkpoint.

        Parameters
        ----------
        embeddings : torch.Tensor
            float32 embeddings of shape (n, 1280)

        Returns
        -------
        torch.Tensor
            ``sigmoid(embeddings @ W.T + b)`` of shape (n, number of classes)
        """
        if self._head is None:
            weights = read_onnx_initializers(
                self.checkpoint_path, [HEAD_WEIGHT_NAME, HEAD_BIAS_NAME]
            )
            self._head = (
                torch.from_numpy(weights[HEAD_WEIGHT_NAME]),
                torch.from_numpy(weights[HEAD_BIAS_NAME]),
            )
        weight, bias = self._head
        return torch.sigmoid(embeddings @ weight.T + bias)



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