from .perch_bird import Model

SAMPLE_RATE = 32000
LENGTH_IN_SAMPLES = 160000


class Model(Model):
    def __init__(self, **kwargs):
        super().__init__(
            sr=SAMPLE_RATE, 
            segment_length=LENGTH_IN_SAMPLES, 
            model_choice="surfperch",
            **kwargs
        )
        self.class_labels = "reef_label"
