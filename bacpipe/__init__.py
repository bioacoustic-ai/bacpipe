import logging

# Unzip models_example.zip to initiate models dir structure
import zipfile
from pathlib import Path

if not Path("bacpipe/model_checkpoints").exists():
    with zipfile.ZipFile("bacpipe/model_checkpoints.zip", "r") as zip_ref:
        zip_ref.extractall("bacpipe")


# Initialize Logger
logger = logging.getLogger("bacpipe")
c_handler = logging.StreamHandler()
c_format = logging.Formatter("%(name)s::%(levelname)s:%(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.DEBUG)
