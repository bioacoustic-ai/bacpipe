import yaml

class ModelBaseClass:
    def __init__(self, **kwargs):
        import yaml
        with open('ievad/config.yaml', 'rb') as f:
            self.config = yaml.safe_load(f)