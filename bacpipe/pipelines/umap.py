from .utils import ModelBaseClass
import umap

class Model(ModelBaseClass):
    def __init__(self):
        super().__init__()
        self.model = umap.UMAP(n_neighbors=self.config['n_neighbors'],
                            n_components=self.config['n_components'],
                            min_dist=self.config['min_dist'],
                            metric=self.config['metric'],
                            random_state=self.config['random_state']
                            ).fit_transform
        
        
    def preprocess(self, audio):
        return audio
    
    def __call__(self, input):
        return self.model(input)
        