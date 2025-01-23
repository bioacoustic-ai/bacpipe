from .animal2vec_mk import Model as A2VMK
from ..utils import MODEL_BASE_PATH

SAMPLE_RATE = 24_000
LENGTH_IN_SAMPLES = int(5 * SAMPLE_RATE)
PATH_TO_PT_FILE = (
    MODEL_BASE_PATH
    + "/animal2vec_xc/checkpoint_last_xeno_canto_base_pretrain_5s-2-1_5_sinc_90ms_mixup_pswish_pretrain.pt"
)


class Model(A2VMK):
    def __init__(self):
        super().__init__(xeno_canto=True)
