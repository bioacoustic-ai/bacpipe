from .perch_v2 import Model

SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = 160000


class Model(Model):
    def __init__(self, **kwargs):
        self.class_label_key = "label"
        super().__init__(
            sr=SAMPLE_RATE, 
            segment_length=LENGTH_IN_SAMPLES, 
            model_choice="perch_8",
            **kwargs
        )
