from ..model_utils import ModelBaseClass
import umap

from bacpipe import settings
# UMAP settings


class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        """
        Initialize the UMAP dimensionality reduction model.
        """
        self.umap_config = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": settings.visualization_dimensions,
            "metric": "euclidean",
            "random_state": 42,
        }

        super().__init__(sr=None, segment_length=None, **kwargs)
        self.model = umap.UMAP(**self.umap_config)

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
        Fit the UMAP model to the input embeddings and transform them.

        Parameters
        ----------
        input : np.array
            embeddings to be reduced

        Returns
        -------
        np.array
            reduced embeddings
        """
        return self.model.fit_transform(input)
