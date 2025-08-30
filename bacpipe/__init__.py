import logging

# Unzip models_example.zip to initiate models dir structure
import zipfile
from pathlib import Path


# Determine package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
PACKAGE_MAIN = Path(__file__).parent

# Unzip models_example.zip if needed
models_dir = PACKAGE_MAIN / "model_checkpoints"
zip_file = PACKAGE_MAIN / "model_checkpoints.zip"

if not models_dir.exists() and zip_file.exists():
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(PACKAGE_MAIN)

# Initialize Logger
logger = logging.getLogger("bacpipe")
c_handler = logging.StreamHandler()
c_format = logging.Formatter("%(name)s::%(levelname)s:%(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.WARNING)
