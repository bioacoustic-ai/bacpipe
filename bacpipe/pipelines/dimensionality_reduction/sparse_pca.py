from ..utils import ModelBaseClass
from sklearn.decomposition import SparsePCA


# UMAP settings
tsne_config = {"n_components": 2}


class Model(ModelBaseClass):
    def __init__(self):
        super().__init__(sr=None, segment_length=None)
        self.model = SparsePCA(**tsne_config).fit_transform

    def preprocess(self, embeddings):
        return embeddings

    def __call__(self, input):
        return self.model(input)
