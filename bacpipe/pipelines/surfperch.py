from .perch import Model

SAMPLE_RATE = 32000
LENGH_IN_SAMPLES = 160000


class Model(Model):
    def __init__(self):
        super().__init__(
            sr=SAMPLE_RATE, segment_length=LENGH_IN_SAMPLES, model_choice="surfperch"
        )
