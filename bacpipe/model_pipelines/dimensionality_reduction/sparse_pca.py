from ..model_utils import ModelBaseClass
from sklearn.decomposition import SparsePCA

# UMAP settings
tsne_config = {"n_components": 2}


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        """
        Initialize the SparsePCA dimensionality reduction model.
        """
        super().__init__(sr=None, segment_length=None, **kwargs)
        self.model = SparsePCA(**tsne_config).fit_transform

    def preprocess(self, embeddings):
        """
        Preprocess the embeddings for the dimensionality reduction model.

        Parameters
        ----------
        embeddings : np.array
            embeddings to be reduced

        Returns
        -------
        np.array
            embeddings without any modification
        """
        return embeddings

    def __call__(self, input):
        """
        Reduce the dimensionality of the input embeddings.

        Parameters
        ----------
        input : np.array
            embeddings to be reduced

        Returns
        -------
        np.array
            reduced embeddings
        """
        return self.model(input)
