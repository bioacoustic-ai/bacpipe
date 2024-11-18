from setuptools import setup, find_packages

setup(
    name="bacpipe",
    version="0.1.0",
    description="BaCPipe: Bioacoustic Collection Pipeline",
    packages=find_packages(),
)

# Unzip models_example.zip to initiate models dir structure
import zipfile

with zipfile.ZipFile("bacpipe/model_checkpoints.zip", "r") as zip_ref:
    zip_ref.extractall("bacpipe")
