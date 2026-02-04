from .aves_especies import Model


class Model(Model):
    def __init__(self, **kwargs):
        super().__init__(birdaves=True, **kwargs)
