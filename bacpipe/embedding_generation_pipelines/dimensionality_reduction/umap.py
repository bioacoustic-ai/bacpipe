from ..utils import ModelBaseClass
import umap


# UMAP settings


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        self.umap_config = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
        }

        super().__init__(sr=None, segment_length=None, **kwargs)
        self.model = umap.UMAP(**self.umap_config).fit_transform

    def preprocess(self, embeddings):
        return embeddings

    def __call__(self, input):
        return self.model(input)
