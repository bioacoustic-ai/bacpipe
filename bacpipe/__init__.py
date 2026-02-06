import logging
import yaml
from types import SimpleNamespace
import importlib.resources as pkg_resources



# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logger = logging.getLogger("bacpipe")
if not logger.handlers:
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(c_handler)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------
# Load config & settings
# --------------------------------------------------------------------
with pkg_resources.open_text(__package__, "config.yaml") as f:
    _config_dict = yaml.load(f, Loader=yaml.CLoader)

with pkg_resources.open_text(__package__, "settings.yaml") as f:
    _settings_dict = yaml.load(f, Loader=yaml.CLoader)

# Expose as mutable namespaces
config = SimpleNamespace(**_config_dict)
settings = SimpleNamespace(**_settings_dict)



# --------------------------------------------------------------------
### EXPOSE API ENDPOINTS ### 
# --------------------------------------------------------------------

from bacpipe.core.experiment_manager import Loader
from bacpipe.model_pipelines.runner import Embedder

from bacpipe.core.workflows import (
    play,
    ensure_models_exist,
    get_model_names,
    evaluation_with_settings_already_exists,
    model_specific_embedding_creation,
    model_specific_evaluation,
    cross_model_evaluation,
    visualize_using_dashboard,
)

from bacpipe.embedding_evaluation.label_embeddings import (
    DefaultLabels, 
    get_default_labels,
    get_ground_truth, 
    get_dt_filename
    )

from bacpipe.core.constants import (
    supported_models, 
    models_needing_checkpoint,
    TF_MODELS,
    EMBEDDING_DIMENSIONS,
    NEEDS_CHECKPOINT
    )

