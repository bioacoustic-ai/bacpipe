"""
Unit tests for the dimensionality reduction models in
``bacpipe.model_pipelines.dimensionality_reduction``.
"""

import numpy as np
import pytest

from bacpipe.model_pipelines.dimensionality_reduction.pca import (
    Model as PCAModel,
)
from bacpipe.model_pipelines.dimensionality_reduction.sparse_pca import (
    Model as SparsePCAModel,
)
from bacpipe.model_pipelines.dimensionality_reduction.t_sne import (
    Model as TSNEModel,
)
from bacpipe.model_pipelines.dimensionality_reduction.umap import (
    Model as UMAPModel,
)

KWARGS = dict(
    device="cpu",
    run_pretrained_classifier=False,
    dim_reduction_model=True,
    model_name="pca",
)


def make_embeddings(n_samples=30, n_dims=10):
    rng = np.random.RandomState(42)
    return rng.rand(n_samples, n_dims)


class TestPCAModel:
    def test_reduces_dimensions(self):
        model = PCAModel(**KWARGS)
        embeds = make_embeddings()
        reduced = model(embeds)
        assert reduced.shape == (embeds.shape[0], 2)

    def test_preprocess_is_identity(self):
        model = PCAModel(**KWARGS)
        embeds = make_embeddings()
        assert model.preprocess(embeds) is embeds


class TestSparsePCAModel:
    def test_reduces_dimensions(self):
        model = SparsePCAModel(**KWARGS)
        embeds = make_embeddings()
        reduced = model(embeds)
        assert reduced.shape == (embeds.shape[0], 2)


class TestTSNEModel:
    def test_reduces_dimensions(self):
        model = TSNEModel(**KWARGS)
        # more samples than the default perplexity (30)
        embeds = make_embeddings(n_samples=40)
        reduced = model(embeds)
        assert reduced.shape == (embeds.shape[0], 2)


class TestUMAPModel:
    def test_reduces_dimensions(self):
        model = UMAPModel(**KWARGS)
        embeds = make_embeddings()
        reduced = model(embeds)
        assert reduced.shape == (embeds.shape[0], 2)
