# Welcome to **bacpipe** (**B**io**A**coustic **C**ollection **Pipe**line)

[![Documentation Status](https://readthedocs.org/projects/bacpipe/badge/?version=latest)](https://bacpipe.readthedocs.io/en/latest/?badge=latest)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bacpipe?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bacpipe)
[![PyPI version](https://badge.fury.io/py/bacpipe.svg?icon=si%3Apython&icon_color=%23f66151)](https://badge.fury.io/py/bacpipe)
[![arXiv](https://img.shields.io/badge/arXiv-2604.11560-b31b1b.svg)](https://arxiv.org/abs/2604.11560)


[![Tests](https://img.shields.io/github/actions/workflow/status/bioacoustic-ai/bacpipe/github_actions_311_linux.yaml?label=3.11&&logo=linux)](https://github.com/bioacoustic-ai/bacpipe/actions/workflows/github_actions_311_linux.yaml)
[![Tests](https://img.shields.io/github/actions/workflow/status/bioacoustic-ai/bacpipe/github_actions_312_linux.yaml?label=3.12&&logo=linux)](https://github.com/bioacoustic-ai/bacpipe/actions/workflows/github_actions_312_linux.yaml)
[![Tests](https://img.shields.io/github/actions/workflow/status/bioacoustic-ai/bacpipe/github_actions_311_apple.yaml?label=3.11&&logo=apple)](https://github.com/bioacoustic-ai/bacpipe/actions/workflows/github_actions_311_apple.yaml)
[![Tests](https://img.shields.io/github/actions/workflow/status/bioacoustic-ai/bacpipe/github_actions_312_apple.yaml?label=3.12&&logo=apple)](https://github.com/bioacoustic-ai/bacpipe/actions/workflows/github_actions_312_apple.yaml)
[![Tests](https://custom-icon-badges.demolab.com/github/actions/workflow/status/bioacoustic-ai/bacpipe/github_actions_312_windows.yaml?label=3.12&logo=windows11&logoColor=white)](https://github.com/bioacoustic-ai/bacpipe/actions/workflows/github_actions_312_windows.yaml)
[![Tests](https://custom-icon-badges.demolab.com/github/actions/workflow/status/bioacoustic-ai/bacpipe/github_actions_311_windows.yaml?label=3.11&logo=windows11&logoColor=white)](https://github.com/bioacoustic-ai/bacpipe/actions/workflows/github_actions_311_windows.yaml)

![](src/bacpipe_logo.png)
image by Nicole Allison

**bacpipe** makes using deep learning models for bioacoustics easy!
Using **bacpipe** you can generate embeddings and classifier predictions and evaluate them using probing, clustering and benchmarking. All you need is your audio data!

There's an applications paper available on [arxiv](https://arxiv.org/abs/2604.11560) describing the pipeline and its capabilities which is currently under review.

The best part is, **bacpipe** comes with an interactive GUI for you to explore your data organized by the state-of-the-art deep learning models for bioacoustics. 

**bacpipe** is also available on pip: `pip install bacpipe`

```python
import bacpipe

# This will execute the whole pipeline
# if nothing is specified it will generate embeddings on
# a set of audio test data using the models birdnet and perch

bacpipe.play()
```
A more detailed description of the API can be found under [API](#api). In `bacpipe/examples` you can find 3 jupyter notebooks demonstrating different use cases of the API. Full documentation can be found at [https://bacpipe.readthedocs.io](https://bacpipe.readthedocs.io). The github repository can be found at [https://github.com/bioacoustic-ai/bacpipe](https://github.com/bioacoustic-ai/bacpipe).

There is a [video tutorial](https://www.youtube.com/watch?v=kw713jF5ts8) available on youtube to install and run bacpipe.


__Try it out__ and (__please__) feel free to give feedback, make suggestions and ask questions either in the [Discussion](https://github.com/bioacoustic-ai/bacpipe/discussions), [Raise an Issue](https://github.com/bioacoustic-ai/bacpipe/issues) or submit a [Pull Request](https://github.com/bioacoustic-ai/bacpipe/pulls). 

This project is still in its early stages and so bugs can still occur. However, all models have been tested successfully on different operating systems. The traffic suggests that it is useful for researchers in our field. It would be great to see it grow into a community project.


## 📚 Table of Contents

- [How it works](#how-it-works)
- [Dashboard visualization](#dashboard-visualization)
    - [Using your annotations](#using-your-annotations)
- [Available models](#available-models)
- [Installation](#installation)
    - [Install prerequisites](#install-prerequisites)
    - [Clone the git repository](#clone-the-git-repository)
    - [Create a virtual environment for this project](#create-a-virtual-environment-for-this-project)
    - [Bacpipe comes with requirements files to install without cuda or tensorflow](#bacpipe-comes-with-requirements-files-to-install-without-cuda-or-tensorflow)
    - [GPU support](#gpu-support)
    - [Model checkpoints are downloaded automatically](#model-checkpoints-are-downloaded-automatically)
    - [Test the installation was successful](#test-the-installation-was-successful)
- [Using bacpipe](#using-bacpipe)
    - [Configurations and settings](#configurations-and-settings)
    - [Running the pipeline](#running-the-pipeline)
    - [Model selection](#model-selection)
    - [Dimensionality reduction](#dimensionality-reduction)
    - [Dashboard](#dashboard)
    - [Evaluation](#evaluation)
    - [Only generate embeddings for annotated segments](#only-generate-embeddings-for-annotated-segments)
- [API](#api)
    - [Use bacpipe immediately on the integrated test data](#use-bacpipe-immediately-on-the-integrated-test-data)
    - [Modify configurations and settings as attributs](#modify-configurations-and-settings-as-attributs)
    - [Modify audio source path, models, device](#modify-audio-source-path-models-device)
    - [Use bacpipe in an existing pipeline](#use-bacpipe-in-an-existing-pipeline)
    - [Produce embeddings for multiple models in your own pipeline](#produce-embeddings-for-multiple-models-in-your-own-pipeline)
    - [Models with classifiers](#models-with-classifiers)
- [Bacpipe's generated data](#bacpipes-generated-data)
    - [Generated Files](#generated-files)
    - [More detailed model information](#more-detailed-model-information)
    - [Brief description of models](#brief-description-of-models)
- [Dimensionality reduction models](#dimensionality-reduction-models)
- [Add a new model](#add-a-new-model)
- [Contribute](#contribute)
- [Known issues](#known-issues)
- [Citation](#citation)
- [Newsletter and Q&A sessions](#newsletter-and-qa-sessions)


---
<!-- <details> -->
<!-- <summary><b style="font-size: 2em;">How it works</b></summary> -->
# How it works



### This repository aims to streamline the generation and evaluation of embeddings using a large variety of bioacoustic models.

The below image shows a comparison of umap embeddings based on 15 different bioacoustic models. The models are being evaluated on a bird and frog dataset (more details in [this conference paper](https://arxiv.org/abs/2504.06710)). 

![](src/normal_overview.png)

**bacpipe** requires a dataset of audio files, runs them through a series of models, and generates embeddings. These embeddings can then be used visualized and evaluated for various tasks such as clustering or classification.

By default the embeddings will be generated for the models specified in the [config.yaml](bacpipe/config.yaml) file. 

Currently these bioacoustic models are supported (more details below):
```yaml

available_models : [
    "audiomae",
    "audioprotopnet",
    "avesecho_passt",
    "aves_especies",
    "bat",
    "batdetect2_clip_avg",
    "batdetect2_dets_avg",
    "beats",
    "birdaves_especies",
    "biolingual",
    "birdnet",
    "birdmae",
    "convnext_birdset",
    "hbdet",
    "insect66",
    "insect459",
    "mix2",
    "naturebeats",
    "perch_bird",
    "perch_v2",
    "protoclr",
    "rcl_fs_bsed",
    "surfperch",
    "google_whale",
    "vggish"
  ]
```
Once the embeddings are generated, 2d reduced embeddings will be created using the dimensionality reduction model specified in the [config.yaml](bacpipe/config.yaml) file. 
And these dimensionality reduction models are supported:

```yaml
available_reduction_models: [
  "pca",
  "sparse_pca",
  "t_sne",
  "umap"
]
```

Furthermore, the embeddings can be evaluated using different downstream tasks. These currently include clustering and probing. The evaluation results are saved in the `bacpipe_results` directory using a standardized folder structure. 

```yaml

available_evaluation_tasks: [
  "probing",
  "clustering"
]
```

# Dashboard visualization

### bacpipe includes a dashboard visualization by default allowing you to easily explore the generated embeddings

Once embeddings are generated, they can be easily visualized using a dashboard (built using `panel`) by simply setting the `dashboard` setting in the [config.yaml](bacpipe/config.yaml) file to `True`.

Below you can see a gif showing the basic usage of the dashboard. Embeddings can also be visualized in 3 dimensions using the `visualization_dimensions` parameter in the [settings](bacpipe/settings.yaml) file.

![](src/bacpipe_demo.gif)

The dashboard has 5 main sections:
1. Single model
2. Two models
3. All models
4. Single model predictions
5. Two model predictions

In the single model section, you can select a model and visualize the embeddings generated by that model. The embeddings can be colored by :
- metadata extracted from the files (date and time information, and file and parent directory) 
- the labels specified in the `annotations.csv` file (if available)
- the cluster labels generated by a clustering algorithm (by default kmeans)

In the dashboard sidebar you can select the model, by which to label the embeddings, whether to remove noise, and the type of classification task to show the results for. 

The noise removal is done by removing the embeddings that do not correspond to annotated sections of the audio files (does not apply if you do not have annotations). This is useful if you want to focus on the annotated sections of the audio files and disregard the rest of the data. 

The visualizations can be saved as png files by clicking the save button in the bottom right corner of the plot.



## Using your annotations

If you have annotations for your dataset, **bacpipe** will read them and use them in visualizations and evaluations. 

To use your annotations for evaluation, they must be in a specific format and file name: The file must be called `annotations.csv` and located in the root directory or your audio data (specified in the `audio_dir` variable in the [config.yaml](bacpipe/config.yaml) file). The file should contain the following columns:
```csv
audiofilename,start,end,label:species
```
Where `audiofilename` is the name of the audio file, `start` and `end` are the start and end times of the annotation in seconds, and `label:species` is the label of the annotation. Your file can have multiple label columns. In that case `species` can be replaced with other label-types such as `calltype` or `individual`, just the `label:` part is necessary to specify it as a label column.

For reference see the [example annotations file](bacpipe/tests/test_data/annotations.csv).

If this file exists, **bacpipe** will create ground_truth.csv files for each `label:` column it finds. These ground truth files will have timestamps that match the embedding timestamps of the respective model. You will be able to find them under the folder `bacpipe_results/<dataset_name>/evaluations/<model_name>/labels`. 

Each label will be selectable to color your embeddings in the dashboard.


# Available models

The models all have their model specific code to ensure inference runs smoothly. 

Models currently include:

|   Name|   ref paper|   ref code|   sampling rate|   input length| embedding dimension |
|---|---|---|---|---|---|
|   AudioMAE    |   [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)   |   [code](https://github.com/facebookresearch/AudioMAE)    |   16 kHz|   10 s| 768 |
|   AudioProtoPNet   |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954125000901)   |   [code](https://github.com/DBD-research-group/AudioProtoPNet)    |   32 kHz|   5 s| 1024 |
|   AvesEcho_PASST   |   [paper](https://arxiv.org/abs/2409.15383)   |   [code](https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1)    |   32 kHz|   3 s| 768 |
|   AVES_ESpecies        |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s| 768 |
| Bat | [paper](https://arxiv.org/abs/2309.11218) | [code](https://github.com/FrankFundel/BAT-cli) | 220.5 kHz | ~1 s | 64 |
| BatDetect2_Clip_Avg | [paper](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1) | [code](https://github.com/macaodha/batdetect2) | 256 kHz | 1 s | 32 |
| BatDetect2_Dets_avg | [paper](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1) | [code](https://github.com/macaodha/batdetect2) | 256 kHz | 1 s | 32 |
| BEATs | [paper](https://arxiv.org/abs/2212.09058) | [code](https://github.com/microsoft/unilm/tree/master/beats) | 16 kHz | 10 s | 768 |
|   BioLingual  |   [paper](https://arxiv.org/abs/2308.04978)   |   [code](https://github.com/david-rx/biolingual)    |   48 kHz|   10 s| 512 |
|   BirdAVES_ESpecies    |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s| 1024 |
|   BirdMAE    |   [paper](https://arxiv.org/abs/2504.12880)   |   [code](https://github.com/DBD-research-group/Bird-MAE)    |   32 kHz|   10 s| 1280 |
|   BirdNET     |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954121000273)   |   [code](https://github.com/kahst/BirdNET-Analyzer)    |   48 kHz|   3 s| 1024 |
|   ConveNeXT_BirdSet   |   [paper](https://arxiv.org/abs/2504.12880)   |   [code](https://github.com/DBD-research-group/BirdSet)    |   32 kHz|   5 s| 1024 |
|   Google_Whale       |   paper   |   [code](https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2)    |   24 kHz|   5 s| 1280 |
|   hbdet |   [paper](https://pubs.aip.org/asa/jasa/article/155/3/2050/3271347)   |   [code](https://github.com/vskode/acodet)    |   2 kHz|   3.9124 s| 2048|
|   Insect66NET |   [paper](https://doi.org/10.1371/journal.pcbi.1011541)   |   [code](https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet)    |   44.1 kHz|   5.5 s| 1280 |
|   Insect459NET |   [paper](https://arxiv.org/pdf/2503.15074)   |   pending    |   44.1 kHz|   5.5 s| 1280 |
|   Mix2        |   [paper](https://arxiv.org/abs/2403.09598)   |   [code](https://github.com/ilyassmoummad/Mix2/tree/main)    |   16 kHz|   3 s| 960 |
|   NatureBEATs        |   [paper](https://arxiv.org/abs/2411.07186)   |   [code](https://github.com/earthspecies/NatureLM-audio)    |   16 kHz|   5 s| 768 |
|   Perch_Bird       |   [paper](https://www.nature.com/articles/s41598-023-49989-z.epdf)   |   [code](https://github.com/google-research/perch)    |   32 kHz|   5 s| 1280 |
|   Perch_V2       |   [paper](https://arxiv.org/abs/2508.04665)   |   [code](https://github.com/google-research/perch_hoplite)    |   32 kHz|   5 s| 1536 |
|   ProtoCLR     |   [paper](https://arxiv.org/pdf/2409.08589)   |   [code](https://github.com/ilyassmoummad/ProtoCLR)    |   16 kHz|   6 s| 384 |
|   RCL_FS_BSED     |   [paper](https://arxiv.org/abs/2309.08971)   |   [code](https://github.com/ilyassmoummad/RCL_FS_BSED)    |   22.05 kHz|   0.2 s| 2048 |
|   SurfPerch       |   [paper](https://arxiv.org/abs/2404.16436)   |   [code](https://www.kaggle.com/models/google/surfperch)    |   32 kHz|   5 s| 1280 |
|   VGGish      |   [paper](https://ieeexplore.ieee.org/document/7952132)   |   [code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)    |   16 kHz|   0.96 s| 128 |



---
# Installation

**bacpipe** runs on all major operating systems and is supported for Python version 3.11 and 3.12. For linux and mac all models are supported for both 3.11 and 3.12. For windows tensorflow models (birdnet, perch_bird, surfperch, google_whale and vggish) are only supported for Python 3.11.

## Install prerequisites

In the following instructions, you will find the steps to install Python 3.11 and git on your local computer. If you already have these installed, you can skip this step.

### Install Python 3.11 and git on your local computer

For **Windows**:
- Download python 3.11: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
- Download Git: https://github.com/git-for-windows/git/releases/download/v2.51.0.windows.1/Git-2.51.0-64-bit.exe

For **Linux**
- `sudo add-apt-repository ppa:deadsnakes/ppa`
- `sudo apt install python3.11`
- `sudo apt install git`

For **Mac**
- (install homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" `
- install git: `brew install git`




### Install `uv` (recommended) or `poetry`

For speed and stability it is recommended to use `uv`. To install `uv` use the following command (you can also do it all without `uv` then, just leave the `uv` part away, as all the commands are also `pip` commands):

`pip install uv` 


For **Windows**:
- move to a folder of your choice (choose wisely - something like `Documents` is always a good starting point), then right click and `Open Git Bash here`
- install **uv**: `"$HOME/AppData/Local/Programs/Python/Python311/python" -m pip install uv`

For **Linux** / **MAC**:
- open a terminal (console) in the folder of your choice
- install **uv**: `/usr/bin/python/Python311/python -m pip install uv`
- (if Mac users get an error locate python with `which python3` and use that path instead followed by `-m pip install uv`)

If you prefer to use `poetry`, you can install it using: 

`pipx install poetry` 

## Clone the git repository


For all systems navitage to a folder of your choice and run:

- `git clone https://github.com/bioacoustic-ai/bacpipe`


## Create a virtual environment for this project

Virtual environments are very important. They ensure that specific libraries that are needed for one project don't get in the way of libraries you need for another project. Luckily packages like `uv` take care of everything and only require a single command.

Create a virtual environment and install the packages with one command:
- `uv sync`

If it's causing trouble, try using `python -m uv sync` instead. Alternatively, you can create the environment manually, using `uv venv --python 3.11` and then `python -m uv pip install -r pyproject.toml`. All commands should install the identical dependencies, but splitting up the commands allows you to have more control over the environment, the python version and name and location.

### Install virtual environment and packages yourself
If you prefer having control over the environment name, use these commands:

Create your virtual environment (all systems):
- `uv venv --python 3.11`

(if that doesn't work for windows, try `$HOME/AppData/Local/Programs/Python/Python311/python.exe -m uv venv`)

After that we will activate the environment and then install the packages.

For **Windows**, activate your environment:
- `source .venv/Scripts/activate`

For **Linux**/**Mac**, activate your environment:
- `source .venv/bin/activate`

(alternatively for `poetry` use `poetry env use 3.11`)

Install the project dependencies using `uv` (all systems):
- `uv pip install -r pyproject.toml`

If you do not have admin rights and encounter a `permission denied` error when using `pip install`, use `python -m pip install ...` instead.


For `poetry`:

- `poetry lock`

- `poetry install`


---
 
## Bacpipe comes with requirements files to install without cuda or tensorflow

If you would like to limit the size of your virtual environment, **bacpipe** comes with requirements files to install without cuda or tensorflow. This is useful if you know that you are not going to use a GPU or you know you don't want to use tensorflow models (like BirdNet).

Use the files `requirements_no_cuda.txt` and `requirements_no_tf.txt` to install the dependencies without cuda or tensorflow.

Install them using `uv` after creating and activating your environment (see previous section):
For requirements without cuda:
- `uv pip install -r requirements_no_cuda.txt`

or for requirements without tensorflow (necessary on **Windows** when using Python 3.12):
- `uv pip install -r requirements_no_tf.txt`

## GPU support

### For PyTorch models

**Bacpipe** supports inference on GPUs. For Mac and linux users the necessary packages are installed alongside `pytorch` and should work out of the box. To use the GPU, set the `device` variable in the [settings.yaml](bacpipe/settings.yaml) file to `cuda` (for linux) and `mps` (for Mac).

### For TensorFlow models

TensorFlow models can also be used on the GPU but only for Linux machines. 

Because of the requirements of `torch==2.6` the cuda versions are installed corresponding to what pytorch supports. However, it is possible to also support tensorflow with cuda. To achieve this you have to first install the normal requirements as explained above. Once you have installed the requirements. Install the following dependencies using 
`uv pip install -r requirements_tf_gpu.txt`.

```python
nvidia-cublas-cu12==12.5.2.13
nvidia-cuda-cccl-cu12==12.5.39.post1
nvidia-cuda-cupti-cu12==12.5.39
nvidia-cuda-cuxxfilt-cu12==12.5.39
nvidia-cuda-nvcc-cu12==12.5.40
nvidia-cuda-nvrtc-cu12==12.5.40
nvidia-cuda-opencl-cu12==12.5.39
nvidia-cuda-profiler-api-cu12==12.5.39
nvidia-cuda-runtime-cu12==12.5.39
nvidia-cuda-sanitizer-api-cu12==12.5.39
nvidia-cudnn-cu12==9.3.0.75
nvidia-cufft-cu12==11.2.3.18
nvidia-curand-cu12==10.3.6.39
nvidia-cusolver-cu12==11.6.2.40
nvidia-cusparse-cu12==12.4.1.24
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-npp-cu12==12.3.0.116
nvidia-nvfatbin-cu12==12.5.39
nvidia-nvjitlink-cu12==12.5.40
nvidia-nvjpeg-cu12==12.3.2.38
nvidia-nvml-dev-cu12==12.5.39
nvidia-nvtx-cu12==12.5.39
```

This should allow you to still support all the pytorch models and also use the tensorflow models with cuda.

## Model checkpoints are downloaded automatically. 

Model checkpoints will be downloaded automatically. Once you run `bacpipe.play()`, it will automatically download models that were included but are not yet available locally. Models are downloaded from [this huggingface repo](https://huggingface.co/datasets/vskode/bacpipe_models/tree/main).

## Test the installation was successful

By doing so you will also ensure that the directory structure for the model checkpoints will be created. 

`pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py`

The tests could take a while, so to run a small test, you can also pass the model you would like to test:

`pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py --models=birdnet,perch_bird`

(keep in mind you have to have the checkpoints locally for the models that require it)

In case of a permission denied error, run
`python -m pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py`

If everything passes then you've successfully installed bacpipe and can now proceed to use it.




# Using bacpipe

## Configurations and settings

To see the capabilities of bacpipe, go ahead and run the `run_pipeline.py` script. This will run the pipeline with the default settings and configurations on a small set of test data. 

### To use bacpipe on your own data, you will need to modify the configuration files.

The only two files that need to be modified are the [config.yaml](bacpipe/config.yaml) and [settings.yaml](bacpipe/settings.yaml) files. The [config.yaml](bacpipe/config.yaml) is used for the standard configurations: 
- path to audio files
- models to run
- dimensionality reduction model
- evaluation tasks
- whether to run the dashboard or not

The [settings.yaml](bacpipe/settings.yaml) file is used for more advanced configurations and does not need to be modified unless you have specific preferences. It includes settings such as to run on CPU, then use `cpu` or a GPU, then use `cuda` (linux)/`mps` (mac) (by default `cpu`). Other advances settings are the paths where results are saved, configurations for the evaluation tasks and more. 

Modify the [config.yaml](bacpipe/config.yaml) file in the root directory to specify the path to your `dataset`. Define what models to run by specifying the strings in the `models` list (copy and paste as needed, I usually just comment the model's I don't want to run). 


## Running the pipeline

Once the configuration is complete, execute the run_pipeline.py file (make sure the environment is activated)
`python run_pipeline.py`

While the scripts are executed, directories will be created corresponding to the `main_results_dir` setting. Embeddings will be saved in `main_results_dir/YOUR_DATASET/embeddings` (see [here](results/test_data/embeddings/README.md) for more info) and if selected, reduced dimensionality embeddings will be saved in `main_results_dir/evaluation/dim_reduced_embeddings` (see [here](results/test_data/dim_reduced_embeddings/README.md) for more info).

### Reusing already created embeddings

If you have already computed embeddings with a model, **bacpipe** will compare between the audio files and the generated embedding files what files still need to be processed. It will then continue where it left off. If all files have been processed already, loading the embeddings and displaying the dashboard should only take a few seconds. 

Even if `overwrite` is set to `True`, **bacpipe** will not overwrite the embeddings if they already exist. `overwrite` only specifies if the evaluations (generating ground truth, clustering and probing) should be overwritten.

Embeddings can be regenerated by setting the parameters `check_if_already_processed` (for embeddings) and `check_if_already_dim_reduced` (for dimensionality reduced embeddings) to False. 

## Model selection

Select the models you want to run in the [config.yaml](bacpipe/config.yaml) file. The models are specified in this ReadMe and in the file [constants](bacpipe/core/constants.py). You can select the models you want to run by adding them to the `models` list in the [config.yaml](bacpipe/config.yaml) file.

## Dimensionality reduction

Different dimensionality reduction models can be selected in the [config.yaml](bacpipe/config.yaml) file. The available models are specified in the section [Dimensionality reduction models](#dimensionality-reduction-models). Insert the name of the selected model in the `dim_reduction_model` variable in the [config.yaml](bacpipe/config.yaml) file. The default is `umap`, but you can also select `pca`, `sparse_pca` or `t_sne`.


## Dashboard

The dashboard is a panel application that allows you to visualize the generated embeddings. To enable the dashboard, set the `dashboard` variable in the [config.yaml](bacpipe/config.yaml) file to `True`. The dashboard will automatically open in your browser (at `http://localhost:8050`) after running the `run_dashboard.py` script.

## Evaluation

As explained in the previous section ([Using your annotations](#using-your-annotations)), if you have an `annotations.csv` file in the root directory of your audio data, **bacpipe** will use it to generate ground truth files for each label column.


The evaluation script will automatically use the annotations to evaluate the clustering and probing performance of the embeddings. The labels will also be used to color the points in the dashboard visualization showing the embeddings.


The labels can then be used to perform clustering and probing evaluation. This can be done only in regard to one label, so specify the main label column in the `label_column` variable in [settings.yaml](bacpipe/settings.yaml). This defaults to `species`. Only labels that exceed the `min_label_occurrences` value will be used. This is to make sure you have enough data to train linear classifiers and do meaningful evaluations. If you have enough labeled data, feel free to increase this. 

See the file [annotations.csv](bacpipe/tests/test_data/annotations.csv) for an example of how the annotations file should look like.

Once the annotations file is created, add either `probing` or `clustering` (or both) to the `evaluation_task` variable in the [config.yaml](bacpipe/config.yaml) file (use double quotes: "probing" or "clustering"). You can run the evaluation script using normal `python run_pipeline.py` command. The evaluation script will automatically use the annotations to compute the clustering and probing performance of the embeddings. The results will be saved in the `bacpipe/results/YOUR_DATASET/evaluation` directory.

If you selected probing, a linear probe will be trained and saved in the probing subdirectory of the evaluation folder. This .pt file can be used to generate class predictions with a model that wasn't originally trained on these classes. A tutorial will be available shortly explaining this in more detail. The .pt file can be used in the repository [acodet](https://github.com/vskode/acodet) to generate class predictions with the combination of a feature extractor and the trained linear classifier.



## Only generate embeddings for annotated segments


Using the settings attribute `only_embed_annotations`, you can also decide to only generate embedding corresponding to your annotations. In that case **bacpipe** will take each annotation and create embeddings for each selected model from only those segments. If segments are shorter than the model input length, the segments will be padded. If they are longer, they will produce several embeddings.


## Models with classifiers

Models that already contain classification heads, are the following:
- AudioProtoPNet
- Bat
- BatDetect2_Clip_Avg
- BirdNET
- ConvNeXT_birdset
- google_whale
- Perch_v2
- Perch_bird
- SurfPerch

With all of these models, you only need to set `run_pretrained_classifier` to True and then the model will save the classification outputs in the `predictions/original_classifier_outputs` folder. Only predictions exceeding the `classifier_threshold` value will be saved. A csv file in the shape of the annotations.csv file is also saved corresponding to the class predictions. The dashboard will also contain an extra `label_by` option `default_classifier`.

# API

**bacpipe** can be used as a package and installed from pip

`pip install bacpipe`

There are `jupyter` notebooks available in the `bacpipe/examples` directory to show you how to use the API. Below you can find a quick overview of some of the API capabilities. 

All previously explained features can also be accessed using the API.

## Use bacpipe immediately on the integrated test data

```python
import bacpipe

# This will execute the whole pipeline
# if nothing is specified it will generate embeddings on
# a set of audio test data using the models birdnet and perch

bacpipe.play()
```

## Modify configurations and settings as attributs
To modify configurations and settings, you can simply access them as attributes. To see available settings and configs run the following commands

```python
bacpipe.config
bacpipe.settings
# you can also check the bacpipe/config.yaml and bacpipe/settings.yaml
# files here in the repository to see all the available settings and
# read their respective description
```

## Modify audio source path, models, device
If you're on a Windows machine, make sure to add a `r` before the path like `r'path\to\audio'` otherwise the path will likely cause problems due to the backslashes. 
```python
# to modify the audio data path for example, do
bacpipe.config.audio_dir = '/path/to/your/audio/dir'

# to modify the models you want to run, do
bacpipe.config.models = ['birdnet', 'birdmae', 'naturebeats']
# if you do not have the checkpoint yet, it will be automatically 
# downloaded and stored locally

bacpipe.settings.device = 'cuda' 
# bacpipe uses multithreading which speeds up model inference if 
# run on a machine supporting cuda

# then run with your settings.
# By default the save logs is True
bacpipe.play(save_logs=True)
# That way bacpipe will generate log files of the outputs and also save your
# config and settings files, which can be helpful in retrospect to remember
# all the settings you chose for a run. 

```

## Use bacpipe in an existing pipeline
If you just want to run models and get embeddings returned without saving them and don't want the dashboard and all of that, define an embedder object and pass it the model name and the settings you modified.

```python
em = bacpipe.Embedder('perch_bird') 

audio_file = '/path/to/all/the/audio/file'
embeddings = em.get_embeddings_from_model(audio_file)

# if the model has a built in classifier, like birdnet, the classification
# score are also saved. You can check if there is a classifier included by 
# checking 
em.model.bool_classifier

# After the generating of embeddings above, you will then be able to access 
# the class predictions using
em.model.classifier_outputs
```

## Produce embeddings for multiple models in your own pipeline
If you want to produce embeddings for multiple models, bacpipe will always store them to keep your memory from overfilling. Still you can use the package to easily access the embeddings and all the metadata

```python
loader = bacpipe.run_pipeline_for_models(
    models=['birdnet', 'perch_bird'], 
    audio_dir='/path/to/your/audio/dir', 
    device='cuda'
)
# this call will initiate the embedding generation process, it will check if embeddings
# already exist for the combination of each model and the dataset and if so it will
# be ready to load them. The loader keys will be the model name and the values will
# be the loader objects for each model. Each object contains all the information
# on the generated embeddings. To name access them:
embeds = loader['birdnet'].embeddings() 
# this will give you a dictionary with the keys corresponding to embedding files
# and the values corresponding to the embeddings as numpy arrays

predictions, label2index = loader['birdnet'].predictions() 
# this will give you a dictionary with the keys corresponding to audio files
# and the values corresponding to the predictions as numpy arrays
# the label2index dict associates the columns of the numpy array with the class label

loader['birdnet'].metadata_dict
# This will give you a dictionary overview of:
# - where the audio data came from,
# - where the embeddings were saved
# - all the audio files, 
# - the embedding size of the model, 
# - the audio file lengths,
# - the number of embeddings for each audio files
# - the sample rate
# - the number of samples per window
# - and the total length of the processed dataset in seconds
# Thic dictionary is also saved as a yaml file in the directory of the embeddings
```

# Bacpipe's generated data


## Generated Files

When processing `bacpipe` generates a number of files. It will firstly create the `main_results_dir` specified in [settings.yaml](bacpipe/settings.yaml) and within that it will create a folder named like the dataset selected in `audio_dir` in [config.yaml](bacpipe/config.yaml). Within this directory, **bacpipe** will create 4 directories: `embeddings`, `dim_reduced_embeddings`, `evaluations`, `logs`. 

The `embeddings` will contain the model-specific folders with the timestamps when they were processed. The `deim_reduced_embeddings` contains the model-specific and dimensionality reduction-specific folder again with a timestamp when they were processed. The `evaluations` contains folders for each model that has been processed. `logs` contains the logs of the runs and the config and settings files used for the run.

### Embedding folders

By default, the naming conventions for the embedding directories is:

`_year-month-day_hour-minute___modelname_datasetname`

The `metadata.yaml` file that will be created in each of the embedding folders has the following structure:

Inside this directory you will find:
- `metadata.yml`, which contains metadata in the shape of a dictionary with the following keys:
    - audio_dir: _data_directory_
    - embed_dir: _path_to_embedding_files_
    - embedding_size: _dimension_of_embeddings_
    - files: _dictionary_with_lists_containing_per_file_information_
        - audio_files: _name_of_audio_files_
        - file_lengths (s): _file_length_in_seconds_
    - model_name: _name_of_model_
    - sample_rate (Hz): _sample_rate_
    - segment_length (samples): _length_of_input_segment_in_samples_
    - total_dataset_length (s): _total_length_of_dataset_in_seconds_
- either the embeddings files (ending on `.npy`) or directories corresponding to subdirectories in the dataset folder


It is important that the name of this directory remains unchanged, so that it can be found automatically. 


### Dimensionality reduced embedding folders

By default the naming conventions for the embedding directories is:

`_year-month-day_hour-minute___DimReductionModelName-DatasetName-ModelName`

Inside this directory you will find:
- `metadata.yml`, which contains metadata in the shape of a dictionary with the following keys (all of the metadata from the embeddings is also repeated):
    - files: _dictionary_with_lists_containing_per_file_information_
        - embedding_dimensions: _tuple_of_number_of_embeddings_by_embedding_size_
        - embedding_files: _name_of_embedding_files_
        - nr_embeds_per_file: _number_of_embeddings_per_file_
    - nr_embeds_total: _total_number_of_embeddings_
- one `.json` file containing the reduced embeddings of the entire dataset
- a plot, visiualizing the reduced embeddings in 2d

It is important that the name of this directory remains unchanged, so that it can be found automatically. 

### Evaluation folders

Within the `evaluations` folder, you will find the following folders: `predictions`, `probing`, `clustering`, `labels` and `plots`. These folders will be filled with results if the corresponding evaluation tasks are selected. `labels` will contain the auto-generated labels from the metadata and if available ground_truth. `predictions` will contain the predictions from the pretrained classifier if available. `probing` will contain the results of the probing evaluation and `clustering` will contain the results of the clustering evaluation. `plots` will contain plots of the evaluation results.

### Pretrained classifier annotations

If the model has a pretrained classifier and `run_pretrained_classifier` is `True`, `bacpipe` will the folders `original_classifier_output` and `raven_tables` and two csv files: `<model_name>_all_predictions.csv` and `<model_name>_classifier_annotations.csv`. 

The folder `original_classifier_output` will contain `.json` files for each audio file with the species and in which time bin they occurred with what certainty. `raven_tables` will contain raven annotation tables for each audio file. 

`<model_name>_all_predictions.csv` contains predictions from the classifier. Per timestamp a maximum number of predictiosn is defined by the parameter `max_labels_per_timestamp` in the [settings.yaml](bacpipe/settings.yaml) file exceeding the threshold (also specified in the settings file).

`<model_name>_classifier_annotations.csv` only lists the highest ranking prediction for a timestamp along with the confidence. The file contains annotations in the same style as the [annotations.csv](bacpipe/tests/test_data/annotations.csv) file.


### Example result files structure

Generated from running with the test_data (timestamps will change) with only directories shown.

This is the source audio data structure:
```.
└── audio
    ├── FewShot
    └── UrbanSoundscape
```

This is the resulting folder structure:

```
.
├── dim_reduced_embeddings
│   ├── 2025-09-09_03-08___umap-test_data-birdnet
│   └── 2025-09-09_03-09___umap-test_data-perch_bird
├── embeddings
│   ├── 2025-09-09_03-08___birdnet-test_data
│   │   └── audio
│   │       ├── FewShot
│   │       └── UrbanSoundscape
│   └── 2025-09-09_03-08___perch_bird-test_data
│       └── audio
│           ├── FewShot
│           └── UrbanSoundscape
└── evaluations
    ├── birdnet
    │   ├── predictions
    │   │   └── birdnet_all_predictions.csv
    │   │   └── birdnet_classifier_annotations.csv
    │   │   └── raven_tables
    │   │   └── original_classifier_outputs
    │   │       └── audio
    │   │           ├── FewShot
    │   │           └── UrbanSoundscape
    │   ├── clustering
    │   ├── labels
    │   ├── probing
    │   └── plots
    ├── overview
    └── perch_bird
        ├── predictions
        │   └── perch_bird_all_predictions.csv
        │   └── perch_bird_classifier_annotations.csv
        │   └── raven_tables
        │   └── original_classifier_outputs
        │       └── audio
        │           ├── FewShot
        │           └── UrbanSoundscape
        ├── clustering
        ├── labels
        ├── probing
        └── plots
```

## More detailed model information


|   Name|   paper|   code|   training|   CNN/Trafo| architecture | checkpoint link |
|---|---|---|---|---|---|---|
|   [AudioMAE](#audiomae)    |   [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)   |   [code](https://github.com/facebookresearch/AudioMAE)    | ssl + ft|   trafo| ViT | [weights](https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view)|
|   [AudioProtoPNet](#audioprotopnet)   |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954125000901)   |   [code](https://github.com/DBD-research-group/AudioProtoPNet)    |  sup l |   CNN | ConvNext | included|
|   [AvesEcho_PaSST](#avesecho_passt)   |   [paper](https://arxiv.org/abs/2409.15383)   |   [code](https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1)    |   sup l |   trafo | PaSST | [weights](https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1/-/blob/main/checkpoints/best_model_passt.pt?ref_type=heads) |
|   [AVES_ESpecies](#aves_especies)        |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   ssl|   trafo | HuBERT | [weights](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.pt)|
| [Bat](#bat) | [paper](https://arxiv.org/abs/2309.11218) | [code](https://github.com/FrankFundel/BAT-cli) | sup l | trafo | CNN + trafo | [weights](https://github.com/FrankFundel/BAT-cli/tree/main/models) |
| [BatDetect2_Clip_Avg](#batdetect2_clip_avg) | [paper](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1) | [code](https://github.com/macaodha/batdetect2) | sup l | trafo | U-Net | included |
| [BatDetect2_Dets_avg](#batdetect2_dets_avg) | [paper](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1) | [code](https://github.com/macaodha/batdetect2) | sup l | trafo | U-Net | included |
| [BEATs](#beats) | [paper](https://arxiv.org/abs/2212.09058) | [code](https://github.com/microsoft/unilm/tree/master/beats) | ssl | trafo | ViT | [weights](https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC) |
|   [BioLingual](#biolingual)  |   [paper](https://arxiv.org/abs/2308.04978)   |   [code](https://github.com/david-rx/biolingual)    |   ssl|   trafo| CLAP | included |
|   [BirdAVES_ESpecies](#birdaves_especies)    |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   ssl|   trafo | HuBERT | [weights](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-large.torchaudio.pt)|
|   [BirdMAE](#birdmae)    |   [paper](https://arxiv.org/abs/2504.12880)   |   [code](https://github.com/DBD-research-group/Bird-MAE)    |   ssl | trafo | ViT | included |
|   [BirdNET](#birdnet)     |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954121000273)   |   [code](https://github.com/kahst/BirdNET-Analyzer)    |   sup l|   CNN | EffNetB0 | [weights](https://github.com/kahst/BirdNET-Analyzer/tree/main/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model)|
|   [ConvNeXT_BirdSet](#convnext_birdset)   |   [paper](https://arxiv.org/abs/2504.12880)   |   [code](https://github.com/DBD-research-group/BirdSet)    |  sup l |   CNN | ConvNext | included|
|   [Google_Whale](#google_whale)       |   paper   |   [code](https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2)    |   sup l|   CNN| EffNetb0 | included|
|   [hbdet](#hbdet) |   [paper](https://pubs.aip.org/asa/jasa/article/155/3/2050/3271347)   |   [code](https://github.com/vskode/acodet)    |   sup l |   CNN | ResNet50| [weights](https://github.com/vskode/acodet/blob/main/acodet/src/models/Humpback_20221130.zip)|
|   [Insect66NET](#insect66net) |   paper   |   [code](https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet)    |   sup l|   CNN | EffNetv2s | [weights](https://gitlab.com/arise-biodiversity/DSI/algorithms/cricket-cicada-detector-capgemini/-/blob/main/src/model_traced.pt?ref_type=heads)|
|   [Insect459NET](#insect459net) |   paper   |   pending    |   sup l|   CNN | EffNetv2s | pending |
|   [Mix2](#mix2)        |   [paper](https://arxiv.org/abs/2403.09598)   |   [code](https://github.com/ilyassmoummad/Mix2/tree/main)    |   sup l|   CNN| MobNetv3 | release pending|
|   [NatureBEATs](#naturebeats)        |   [paper](https://arxiv.org/abs/2411.07186)   |   [code](https://github.com/earthspecies/NatureLM-audio)    | ssl | trafo | BEATs | [weights](https://drive.google.com/file/d/12BrWRbxJsuwZHOkzX8HEpGgSMy5VnwCp/view?usp=sharing) |
|   [Perch_Bird](#perch_bird)       |   [paper](https://www.nature.com/articles/s41598-023-49989-z.epdf)   |   [code](https://github.com/google-research/perch)    |   sup l|   CNN| EffNetb0 | included |
|   [Perch_V2](#perch_v2)       |   [paper](https://arxiv.org/abs/2508.04665)   |   [code](https://github.com/google-research/perch_hoplite)    |   sup l |   CNN | EfficientNetB3 | included |
|   [ProtoCLR](#protoclr)     |   [paper](https://arxiv.org/pdf/2409.08589)   |   [code](https://github.com/ilyassmoummad/ProtoCLR)    |   sup cl|   trafo| CvT-13 | [weights](https://huggingface.co/ilyassmoummad/ProtoCLR)|
|   [RCL_FS_BSED](#rcl_fs_bsed)     |   [paper](https://arxiv.org/abs/2309.08971)   |   [code](https://github.com/ilyassmoummad/RCL_FS_BSED)    |   sup cl|   CNN| ResNet9 | [weights](https://zenodo.org/records/11353694)|
|   [SurfPerch](#surfperch)       |   [paper](https://arxiv.org/abs/2404.16436)   |   [code](https://www.kaggle.com/models/google/surfperch)    |   sup l|   CNN| EffNetb0 | included |
|   [VGGish](#vggish)      |   [paper](https://ieeexplore.ieee.org/document/7952132)   |   [code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)    |   sup l|   CNN| VGG | [weights](https://storage.googleapis.com/audioset/vggish_model.ckpt)|

## Brief description of models
All information is extracted from the respective repositories and manuscripts. Please refer to them for more details

### AudioMAE
- spectrogram input
- self-supervised pretrained model, fine-tuned
- vision transformer
- trained on general audio

AudioMAE from the facebook research group is a vision transformer pretrained on AudioSet-2M data and fine-tuned on AudioSet-20K.

### AudioProtoPNet
- spectrogram input
- supervised learning, trained using asymmetric loss
- ConvNext architecture as feature extractor
- trained on the xeno-canto large section of BirdSet

This CNN is trained in two phases. The main contribution of this model is its interpretability. It learned prototypes during its second training phase which can be used during inference time to visualize sections of the spectrogram that were most important for classification. It also reaches competitive performance on bird classification tasks. The (included) prototype-based classifier can distinguish 9736 classes. 

### AvesEcho_PaSST
- transformer
- supervised pretrained model, fine-tuned
- pretrained on general audio and bird song data

AvesEcho_PaSST is a vision transformer trained on AudioSet and (deep) fine-tuned on xeno-canto. The model is based on the [PaSST](https://github.com/kkoutini/PaSST) framework. 

### AVES_ESpecies
- transformer
- self-supervised pretrained model
- trained on general audio

AVES_ESpecies is short for Animal Vocalization Encoder based on Self-Supervision by the Earth Species Project. The model is based on the HuBERT-base architecture. The model is pretrained on unannotated audio datasets AudioSet-20K, FSD50K and the animal sounds from AudioSet and VGGSound.

### Bat
- trafo
- supervised learning
- bat data (Skiba et al. 2003))

Bat is a hybrid CNN+transformer model used for bat classification. 

### BatDetect2_Clip_Avg
- trafo
- supervised learning
- trained on bat data

BatDetect2_Clip_Avg is a transformer model trained on bat data. The model was trained to detect bat calls in audio clips and to classify them into different species. The model uses a clip-level average pooling strategy to aggregate the predictions from multiple segments of the audio clip.

### BatDetect2_Dets_avg
- trafo
- supervised learning
- trained on bat data

BatDetect2_Dets_avg is a transformer model trained on bat data. This variant of the model returns the spatially averaged embeddings generated from the detection part of the model.


### BEATs
- trafo
- self-supervised learning
- trained on AudioSet

BEATs is microsofts SotA audio model based on audio pre-training with acoustic tokenizers. The model reaches competitive results with many bioacosutic models in benchmarks for linear and attentive probing, and is therefore also included in bacpipe as a general audio baseline model.

### BioLingual
- transformer
- spectrogram input
- contrastive-learning
- self-supervised pretrained model
- trained on animal sound data (primarily bird song)

BioLingual is a language-audio model trained on captioning bioacoustic datasets inlcuding xeno-canto and iNaturalist. The model architecture is based on the [CLAP](https://arxiv.org/pdf/2211.06687) model architecture. 

### BirdAVES_ESpecies
- transformer
- self-supervised pretrained model
- trained on general audio and bird song data

BirdAVES_ESpecies is short for Bird Animal Vocalization Encoder based on Self-Supervision by the Earth Species Project. The model is based on the HuBERT-large architecture. The model is pretrained on unannotated audio datasets AudioSet-20K, FSD50K and the animal sounds from AudioSet and VGGSound as well as bird vocalizations from xeno-canto. 


### BirdMAE
- trafo (ViT)
- self-supervised model
- trained on XCM

BirdMAE is a masked autoencoder inspired by meta's AudioMAE, however the model was heavily adapted for the bioacoustic domain. The model was trained on the xeno-canto M dataset (1.7 million samples) from BirdSet and evaluated on various soundscape datasets, where it outperformed all competing models (including SotA bioacoustic models).

### BirdNET
- CNN
- supervised training model
- trained on bird song data

BirdNET (v2.4) is based on a EfficientNET(b0) architecture. The model is trained on a large amount of bird vocalizations from the xeno-canto database alongside other bird song databses. 

### ConvNeXT_BirdSet
- CNN
- supervised learning
- trained on BirdSet

The ConvNeXT_birdset model is a ConvNeXT CNN trained on the BirdSet dataset (which consists of large portions of the xeno-canto database and uses a varierty of multilabel soundscape recordings for evaluation.)



### Google_Whale
- CNN
- supervised training model
- trained on 7 whale species

Google_Whale (multispecies_whale) is a EFficientNet B0 model trained on whale vocalizations and other marine sounds.


### hbdet
- CNN
- supervised training model
- trained on humpback whale song

hbdet is a binary classifier based on a ResNet-50 model trained on humpback whale data from different parts in the North Atlantic. 

### Insect66NET
- CNN
- supervised training model
- trained on insect sounds

InsectNET66 is a [EfficientNet v2 s](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html) model trained on the [Insect66 dataset](https://zenodo.org/records/8252141) including sounds of grasshoppers, crickets, cicadas developed by the winning team of the Capgemini Global Data Science Challenge 2023.

### Insect459NET
- CNN
- supervised training model
- trained on insect sounds

InsectNET459 is a [EfficientNet v2 s](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html) model trained on the Insect459 dataset (publication pending).


### Mix2
- CNN
- supervised training model
- trained on frog sounds

Mix2 is a [MobileNet v3](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py) model trained on the [AnuranSet](https://github.com/soundclim/anuraset) which includes sounds of 42 different species of frogs from different regions in Brazil. The model was trained using a mixture of Mixup augmentations to handle the class imbalance of the data.

### NatureBEATs
- trafo
- self-supervised training model
- trained on diverse set of bioacoustics, general sound, music, human speech

NatureLM-Audio is a very ambitious foundational model specifically for bioacoustics. It uses Microsoft's BEATs backbone as an audio encoder along with Meta's Llama-3.1-8B large language model capabilities. In the implementation used here in bacpipe, only the support for BEATs audio-encoder with NatureLM-Audio's weights, referred to here as NatureBEATs, is provided. 

### RCL_FS_BSED
- CNN
- supervised contrastive learning
- trained on dcase 2023 task 5 dataset [link](https://zenodo.org/records/6482837)

RCL_FS_BSED stands for Regularized Contrastive Learning for Few-shot Bioacoustic Sound Event Detection and features a model based on a ResNet model. The model was originally created for the DCASE bioacoustic few shot challenge (task 5) and later improved.

### ProtoCLR
- transformer
- supervised contrastive learning
- trained on bird song data

ProtoCLR stands for Prototypical Contrastive Learning for robust representation learning. The architecture is a CvT-13 (Convolutional vision transformer) with 20M parameters. ProtoCLR has been validated on transfer learning tasks for bird sound classification, showing strong domain-invariance in few-shot scenarios. The model was trained on the xeno-canto dataset.


### Perch_Bird
- CNN
- supervised training model
- trained on bird song data

Perch_Bird is a EFficientNet B1 model trained on the entire Xeno-canto database.

### Perch_V2
- CNN
- supervised learning
- trained on birds, amphibians, insects and mammals (xeno-canto, iNaturalis, Tierstimmenarchiv)

Perch V2 or Perch 2.0 is the updated version of the Perch model from bioacousticians at Google. The model is a EfficientNetB3, trained on a very large database of various species. The classifier is able to distinguish 14795 different species. 
**Bacpipe** uses the onnx implementation which allows the model to run through pytorch so it can be used with all operating systems and also supports cuda and mps accelleration.

### SurfPerch
- CNN
- supervised training model
- trained on bird song, fine-tuned on tropical reef data

Perch is a EFficientNet B1 model trained on the entire Xeno-canto database and fine tuned on coral reef and unrelated sounds.


### VGGISH
- CNN
- supervised training model
- trained on general audio

VGGish is a model based on the [VGG](https://arxiv.org/pdf/1409.1556) architecture. The model is trained on audio from youtube videos (YouTube-8M)

# Dimensionality reduction models

To evaluate the generated embeddings a number of dimensionality reduction models are included in this repository:


|   name| reference|code reference   | linear |
|---|---|---|---|
|   UMAP        |   [paper](https://arxiv.org/abs/1802.03426)   |   [code](https://github.com/lmcinnes/umap)    |   No |
|   t-SNE        |   [paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl)   |   [code](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)    |   No |
|   PCA        |   [paper](http://www.cs.columbia.edu/~blei/fogm/2020F/readings/LeeSeung1999.pdf)   |   [code](https://scikit-learn.org/dev/modules/generated/sklearn.decomposition.PCA.html)    |   Yes |
|   Sparse_PCA        |   [paper](http://www.cs.columbia.edu/~blei/fogm/2020F/readings/LeeSeung1999.pdf)   |   [code](https://scikit-learn.org/dev/modules/generated/sklearn.decomposition.SparsePCA.html)    |   Yes |




# Add a new model

To add a new model, simply add a pipeline with the name of your model. Make sure your model follows the following criteria:

- define the model specific __sampling rate__
- define the model specific input __segment length__
- define a class called "__Model__" which inherits the __ModelBaseClass__ from __bacpipe.utils__
- define the __init__, preproc, and __call__ methods so that the model can be called
- if necessary save the checkpoint in the __bacpipe.model_checkpoints__ dir with the name corresponding to the name of the model
- if you need to import code where your specific model class is defined, create a directory in __bacpipe.model_specific_utils__ corresponding to your model name "newmodel" and add all the necessary code in there

Here is an example:

```python 
import torch
from bacpipe.model_pipelines.model_specific_utils.newmodel.module import MyClass

SAMPLE_RATE = 12345
LENGTH_IN_SAMPLES = int(10 * SAMPLE_RATE)

from ..utils import ModelBaseClass

class Model(ModelBaseClass):
    def __init__(self, **kwargs):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)
        self.model = MyClass()
        state_dict = torch.load(
            self.model_base_path + "/newmodel/checkpoint_path.pth",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)

    def preprocess(self, audio): # audio is a torch.tensor object
        # insert your preprocessing steps
        return processed_audio

    def __call__(self, x):
        # by default the model will be called with .eval() mode
        return self.model(x)

```

Most of the models are based on pytorch. For tensorflow models, see __birdnet__, __hbdet__ or __vggish__.

</details>

---

# Contribute

This repository is intended to be a collaborative project for people working in the field of bioacoustics. If you think there is some improvement that could be useful, please raise an [issue](https://github.com/bioacoustic-ai/bacpipe/issues), submit a [PR](https://github.com/bioacoustic-ai/bacpipe/pulls) or get in touch. 

There are two main intentions for this repository that should always be considered when contributing: 
      
      1. Only add new requirements if truly necessary

Given the large number of different models, there are already a lot of requirements. To ensure that the repository is stable, and installation errors are kept minimal, please only add code with new requirements if truly necessary.
      
      2. The main purpose of bacpipe is quickly generating embeddings from models
      
There should always be a baseline minimal use case, where embeddings are created from different feature extractors and everything else is an add-on.

# Known issues


`bacpipe` is being updated regularly. To make sure you're always up to date with the latest fixes run `git pull` regularly.
If you have local changes (like changes in the `config` or `settings` files) you don't want to lose run the following:
```bash
git stash
git pull
git stash apply
```
That way it will stash your changes, update `bacpipe` for you and then apply your changes again, so you can continue where you left off. 


Given that this repository compiles a large number of very different deep learning models with different requirements, some issues have been noted. 

Please raise issues if there are questions or bugs. 

Previous versions of **bacpipe** included models like animal2vec, but the requirements conflicts led me to remove them. In the future I hope there will be an updated version of those models and then they will be included again.

# Citation

A lot of work has gone into creating these bioacoustic models, both by data collectors and by machine learning practitioners, please cite the authors of the respective models (all models are referenced in the table above).


This work now has a paper associated with it. The manuscript is currently under review. A preprint is available on arXiv. If you use **bacpipe** for your research, please include the following reference:

```bibtex
@misc{kather2026bacpipe,
      title={bacpipe: a Python package to make bioacoustic deep learning models accessible}, 
      author={Vincent S. Kather and Sylvain Haupert and Burooj Ghani and Dan Stowell},
      year={2026},
      eprint={2604.11560},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.11560}, 
}
```


# Newsletter and Q&A sessions

Reading from the traffic on the repository, there seems to be an interest in bacpipe. I have set up a newsletter under this link: https://buttondown.com/vskode. Once more than 30 people have signed up for the newsletter, I will schedule a Q&A session and post the link in the newsletter. Hopefully I can then help answer questions and address issues that people are running into. 