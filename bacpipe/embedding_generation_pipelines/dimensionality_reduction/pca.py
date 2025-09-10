from ..utils import ModelBaseClass
from sklearn.decomposition import PCA


# UMAP settings
tsne_config = {"n_components": 2}


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=None, segment_length=None, **kwargs)
        self.model = PCA(**tsne_config).fit_transform

    def preprocess(self, embeddings):
        return embeddings

    def __call__(self, input):
        return self.model(input)
