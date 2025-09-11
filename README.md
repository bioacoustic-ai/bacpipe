# Welcome to **bacpipe** (**B**io**A**coustic **C**ollection **Pipe**line)

![](src/bacpipe_logo.png)

**bacpipe** makes using deep learning models for bioacoustics easy!
Using **bacpipe** you can generate embeddings, classification predictions and clusters. All you need is your audio data and to customize the settings.

And the best part is, it comes with a GUI for you to explore the results.

**bacpipe** also ties in nicely with [acodet](https://github.com/vskode/acodet), allowing you to generate heatmaps of species activity from your datasets based on predictions of deep learning models. But keep in mind predictions are only as good as the model's you are using - they can seem confident but still be very wrong - **bacpipe**'s aim is to make model evaluations easier and let us improve them.

**bacpipe** is also available on pip: `pip install bacpipe`

```python
import bacpipe

# This will execute the whole pipeline
# if nothing is specified it will generate embeddings on
# a set of audio test data using the models birdnet and perch

bacpipe.play()
```
A more detailed description of the API can be found under [API](#api)


## 📚 Table of Contents

- [How it works](#how-it-works)
- [API](#api)
- [Dashboard visualization](#dashboard-visualization)
- [Installation](#installation)
- [Usage](#usage)
  - [Configurations and settings](#configurations-and-settings)
  - [Running the pipeline](#running-the-pipeline)
  - [Model selection](#model-selection)
  - [Dimensionality reduction](#dimensionality-reduction)
  - [Dashboard](#dashboard)
  - [Evaluation](#evaluation)
- [Available models](#available-models)
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
    "audiomae"
    "audioprotopnet"
    "avesecho_passt"
    "aves_especies"
    "beats"
    "birdaves_especies"
    "biolingual"
    "birdnet"
    "birdmae"
    "hbdet"
    "insect66"
    "insect459"
    "mix2"
    "naturebeats"
    "perch_bird"
    "protoclr"
    "rcl_fs_bsed"
    "surfperch"
    "google_whale"
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

Furthermore, the embeddings can be evaluated using different metrics. The evaluation is done using the `evaluate.py` script, which takes the generated embeddings and computes various metrics such as clustering performance and classification performance. The evaluation results are saved in the `bacpipe/results` directory.

```yaml

available_evaluation_tasks: [
  "classification",
  "clustering"
]
```


The repository also includes a panel dashboard for visualizing the generated embeddings. To enable the dashboard, simply set the `dashboard` variable to `True` in the [settings.yaml](bacpipe/settings.yaml) file. The dashboard will automatically open in your browser (at `http://localhost:8050`) after running the `run_dashboard.py` script.

The pipeline is designed to be modular, so you can easily add or remove models as needed. The models are organized into pipelines, which are defined in the `bacpipe/embedding_generation_pipelines/feature_extractors` directory. If you want to add a different dimensionality reduction model, you do so by adding new pipeline to the `bacpipe/embedding_generation_pipelines/dimensionality_reduction` directory.

## Using annotations for evaluation

If you have annotations for your dataset, you can use them to evaluate the generated embeddings. The labels will be used to compute the clustering and classification performance of the embeddings. 

To use the annotations for evaluation, create a file called `annotations.csv` in the directory specified in the `audio_dir` variable in the [config.yaml](bacpipe/config.yaml) file. The file should contain the following columns:
```csv
audiofilename,start,end,label
```
Where `audiofilename` is the name of the audio file, `start` and `end` are the start and end times of the annotation in seconds, and `label` is the label of the annotation.

For reference see the [example annotations file](bacpipe/tests/test_data/annotations.csv).

If this file exists, the evaluation script will automatically use the annotations to compute the clustering and classification performance of the embeddings. The labels will also be used to color the points in the dashboard visualization showing the embeddings.

# API

**bacpipe** can be used as a package and installed from pip

`pip install bacpipe`

```python
import bacpipe

# This will execute the whole pipeline
# if nothing is specified it will generate embeddings on
# a set of audio test data using the models birdnet and perch

bacpipe.play()

# to modify configurations and settings, you can simply access them
# as attributes

# to see available settings and configs run the above commands
bacpipe.config
bacpipe.settings

# to modify the audio data path for example, do
bacpipe.config.audio_dir = '/path/to/your/audio/dir'

# to modify the models you want to run, do
bacpipe.config.models = ['birdnet', 'birdmae', 'naturebeats']
# keep in mind some models require checkpoints, to find out which ones, run
bacpipe.models_needing_checkpoint
# links to checkpoints are to be found in this readme file, 
# location of the checkpoints is specified under 
bacpipe.settings.model_base_path = '/path/to/model_checkpoints'
# On first execution the birdnet checkpoint is downloaded and the
# model_checkpoints folder is created from the current working directory

# If you just want to run models and get embeddings and don't want 
# the dashboard and all of that, define an embedder object and pass it
# the model name, and the settings you modified
# To ensure birdnet is downloaded, run the following (this is done by default
# if you run bacpipe.play(), but without that, you need to call this explicitly)
bacpipe.ensure_std_models(bacpipe.settings.model_base_path)

em = bacpipe.Embedder('birdnet', **vars(bacpipe.settings)) 
# the vars part is important!

audio_file = '/path/to/all/the/audio/file'
embeddings = em.get_embeddings_from_model(audio_file)

# if the model has a built in classifier, like birdnet
# you can make sure the class predictions are also saved 
# by setting 
bacpipe.settings.run_pretrained_classifier = True
# the generating of embeddings above will then let you access 
# the class predictions using
em.model.classifier_outputs

# If you want to produce embeddings for various models, bacpipe will always store
# them to keep your memory from overfilling. Still you can use the package to easily
# access the embeddings and all the metadata
loader = bacpipe.model_specific_embedding_creation(
  **vars(bacpipe.config), **vars(bacpipe.settings)
  )

# this call will initiate the embedding generation process, it will check if embeddings
# already exist for the combination of each model and the dataset and if so it will
# be ready to load them. The loader keys will be the model name and the values will
# be the loader objects for each model. Each object contains all the information
# on the generated embeddings. To name access them:
loader['birdnet'].embedding_dict() 
# this will give you a dictionary with the keys corresponding to embedding files
# and the values corresponding to the embeddings as numpy arrays

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


# Dashboard visualization

### bacpipe includes a dashboard visualization by default allowing you to easily explore the generated embeddings

Once embeddings are generated, they can be easily visualized using a dashboard (built using `panel`) by simply setting the `dashboard` setting in the [config.yaml](bacpipe/config.yaml) file to `True`.

Below you can see a gif showing the basic usage of the dashboard.

![](src/bacpipe_demo.gif)

The dashboard has 3 main sections:
1. Single model
2. Two models
3. All models

In the single model section, you can select a model and visualize the embeddings generated by that model. The embeddings can be colored by :
- metadata extracted from the files (date and time information, and file and parent directory) 
- the labels specified in the `annotations.csv` file
- the cluster labels generated by the clustering algorithm (kmeans)

In the dashboard sidebar you can select the model, by which to label the embeddings, whether to remove noise, and the type of classification task to show the results for. 

The noise removal is done by removing the embeddings that do not correspond to annotated sections of the audio files. This is useful if you want to focus on the annotated sections of the audio files and disregard the rest of the data. 

The visualizations can be saved as png files by clicking the save button in the bottom right corner of the plot.

__Try it out__ and (__please__) feel free to give feedback and ask questions (or suggestions for improvements) - or in case something does not work raise issues.




---
# Installation

### Install `uv` (recommended) or `poetry`

It is recommended to use python 3.11 for this repository, as some of the models require it. 

For speed and stability it is recommended to use `uv`. To install `uv` use the following command (you can also do it all without `uv` then, just leave the `uv` part away, as all the commands are also `pip` commands):

`pip install uv` 

(for windows use `/c/Users/$USERNAME/AppData/Local/Programs/Python/Python311/python.exe -m pip install uv`)

If you prefer to use `poetry`, you can install it using: 

`pipx install poetry` 

### Create a virtual environment

`python3.11 -m uv venv .env_bacpipe`

(for windows use `/c/Users/$USERNAME/AppData/Local/Programs/Python/Python311/python.exe -m uv venv .env_bacpipe`)

(alternatively for `poetry` use `poetry env use 3.11`)

activate the environment

`source .env_bacpipe/bin/activate` (for windows use `source .env_bacpipe\Scripts\activate`)

### Clone the repository
`git clone https://github.com/bioacoustic-ai/bacpipe.git`

cd into the bacpipe directory (`cd bacpipe`)

### Install the dependencies once the prerequisites are satisfied.

`uv pip install -r pyproject.toml`

- this will automatically install requirements based on your os, so windows should also work fine. However, gpu support is not available on windows

For `poetry`:

`poetry lock`
`poetry install`

Alternatively: 

`uv sync`

If for some reasons you would prefer requirements, use the these for windows:

`uv pip install -r requirements_windows.txt` 

If you do not have admin rights and encounter a `permission denied` error when using `pip install`, use `python -m pip install ...` instead.

### OPTIONAL: Add other model checkpoints that are not included by default.

Download the ones that are available from [here](https://github.com/bioacoustic-ai/bacpipe/tree/main/bacpipe/pipelines) and create directories corresponding to the pipeline-names and place the checkpoints within them. 

## Test the installation was successful

By doing so you will also ensure that the directory structure for the model checkpoints will be created. 

`pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py`

The tests could take a while, so to run a small test, you can also pass the model you would like to test:

`pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py --models=birdnet,perch`

(keep in mind you have to have the checkpoints locally for the models that require it)

In case of a permission denied error, run
`python -m pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py`

If everything passes then you've successfully installed bacpipe and can now proceed to use it.

# Usage

## Configurations and settings

To see the capabilities of bacpipe, go ahead and run the `run_pipeline.py` script. This will run the pipeline with the default settings and configurations on a small set of test data. 

### To use bacpipe on your own data, you will need to modify the configuration files.

The only two files that need to be modified are the [config.yaml](bacpipe/config.yaml) and [settings.yaml](bacpipe/settings.yaml) files. The [config.yaml](bacpipe/config.yaml) is used for the standard configurations: 
- path to audio files
- models to run
- dimensionality reduction model
- evaluation tasks
- whether to run the dashboard or not

The [settings.yaml](bacpipe/settings.yaml) file is used for more advanced configurations and does not need to be modified unless you have specific preferences. It includes settings such as to run on a `cpu` or a `cuda` (gpu) (by default `cpu`), the paths where results are saved, configurations for the evaluation tasks and more. 

Modify the [config.yaml](bacpipe/config.yaml) file in the root directory to specify the path to your `dataset`. Define what models to run by specifying the strings in the `models` list (copy and paste as needed, I usually just comment the model's I don't want to run). 

If you have already computed embeddings on the dataset specified in audio_data, and you want to do the dimensionality reduction and evaluation for the models you have already run, you can set the `already_computed` variable to `True`. This will only select the models that have already been computed.

In either case if you have already computed embeddings with a model, **bacpipe** will skip the model and use the **already computed** embeddings (if they are still located in the same directory). Even if `overwrite` is set to `True`, **bacpipe** will not overwrite the embeddings if they already exist. It will recompute clusterings and label generation.

## Running the pipeline

Once the configuration is complete, execute the run_pipeline.py file (make sure the environment is activated)
`python run_pipeline.py`

While the scripts are executed, directories will be created corresponding to the `main_results_dir` setting. Embeddings will be saved in `main_results_dir/YOUR_DATASET/embeddings` (see [here](results/test_data/embeddings/README.md) for more info) and if selected, reduced dimensionality embeddings will be saved in `main_results_dir/evaluation/dim_reduced_embeddings` (see [here](results/test_data/dim_reduced_embeddings/README.md) for more info).


## Model selection

Select the models you want to run in the [config.yaml](bacpipe/config.yaml) file. The models are specified in this ReadMe and in the [test_file](bacpipe/tests/test_embedding_creation.py). You can select the models you want to run by adding them to the `models` list in the [config.yaml](bacpipe/config.yaml) file.

## Dimensionality reduction

Different dimensionality reduction models can be selected in the [config.yaml](bacpipe/config.yaml) file. The available models are specified in the section [Dimensionality reduction models](#dimensionality-reduction-models). Insert the name of the selected model in the `dim_reduction_model` variable in the [config.yaml](bacpipe/config.yaml) file. The default is `umap`, but you can also select `pca`, `sparse_pca` or `t_sne`.


## Dashboard

The dashboard is a panel application that allows you to visualize the generated embeddings. To enable the dashboard, set the `dashboard` variable in the [config.yaml](bacpipe/config.yaml) file to `True`. The dashboard will automatically open in your browser (at `http://localhost:8050`) after running the `run_dashboard.py` script.

## Evaluation

You can use `bacpipe` to evaluate the generated embeddings using different metrics. To evaluate the embeddings, you need annotations for your dataset. The annotations should be in a file called `annotations.csv` in the directory specified in the `audio_dir` variable in the [config.yaml](bacpipe/config.yaml) file or the results directory of your dataset `main_results_dir/YOUR_DATASET`. The file should contain the following columns:
```csv
audiofilename,start,end,label:species
```

Where `audiofilename` is the name of the audio file, `start` and `end` are the start and end times of the annotation in seconds, and `label` is the label of the annotation.

`species` is a placeholder here and can be replaced with any label description. So if you have labelled call types, change it to `label:call_type`. But it's important that there are no spaces and that it contains `label:`. By doing this you will be able to visualize your data based on all of these label columns.

The labels can then be used to perform clustering and classification evaluation. This can be done only in regard to one label, so specify the main label column in the `label_column` variable in [settings.yaml](bacpipe/settings.yaml). This defaults to `species`. Only labels that exceed the `min_label_occurances` value will be used. This is to make sure you have enough data to train linear classifiers and do meaningful evaluations. If you have enough labeled data, feel free to increase this. 

See the file [annotations.csv](bacpipe/tests/test_data/annotations.csv) for an example of how the annotations file should look like.

Once the annotations file is created, add either `classification` or `clustering` (or both) to the `evaluation_task` variable in the [config.yaml](bacpipe/config.yaml) file (use double quotes: "classification" or "clustering"). You can run the evaluation script using normal `python run_pipeline.py` command. The evaluation script will automatically use the annotations to compute the clustering and classification performance of the embeddings. The results will be saved in the `bacpipe/results/YOUR_DATASET/evaluation` directory.

If you selected classification, a linear classifier will be trained and saved in the classification subdirectory of the evaluation folder. This .pt file can be used to generate class predictions with a model that wasn't originally trained on these classes. A tutorial will be available shortly explaining this in more detail. The .pt file can be used in the repository [acodet](https://github.com/vskode/acodet) to generate class predictions with the combination of a feature extractor and the trained linear classifier.

## Models with classifiers

Models that already contain classification heads, are the following:
- AudioProtoPNet
- BirdNET
- Perch_bird
- SurfPerch
- google_whale

With all of these models, you only need to set `run_pretrained_classifier` to True and then the model will save the classification outputs in the `classification/original_classifier_outputs` folder. Only predictions exceeding the `classifier_threshold` value will be saved. A csv file in the shape of the annotations.csv file is also saved corresponding to the class predictions. The dashboard will also contain an extra `label_by` option `default_classifier`.

# Available models

The models all have their model specific code to ensure inference runs smoothly. More info on the models and their pipelines can be found [here](bacpipe/pipelines/README.md).

Models currently include:

|   Name|   ref paper|   ref code|   sampling rate|   input length| embedding dimension |
|---|---|---|---|---|---|
|   AudioMAE    |   [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)   |   [code](https://github.com/facebookresearch/AudioMAE)    |   16 kHz|   10 s| 768 |
|   AudioProtoPNet   |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954125000901)   |   [code](https://github.com/DBD-research-group/AudioProtoPNet)    |   32 kHz|   5 s| 1024 |
|   AvesEcho_PASST   |   [paper](https://arxiv.org/abs/2409.15383)   |   [code](https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1)    |   32 kHz|   3 s| 768 |
|   AVES_ESpecies        |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s| 768 |
| BEATs | [paper](https://arxiv.org/abs/2212.09058) | [code](https://github.com/microsoft/unilm/tree/master/beats) | 16 kHz | 10 s | 768 |
|   BioLingual  |   [paper](https://arxiv.org/abs/2308.04978)   |   [code](https://github.com/david-rx/biolingual)    |   48 kHz|   10 s| 512 |
|   BirdAVES_ESpecies    |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s| 1024 |
|   BirdMAE    |   [paper](https://arxiv.org/abs/2504.12880)   |   [code](https://github.com/DBD-research-group/Bird-MAE)    |   32 kHz|   10 s| 1280 |
|   BirdNET     |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954121000273)   |   [code](https://github.com/kahst/BirdNET-Analyzer)    |   48 kHz|   3 s| 1024 |
|   Google_Whale       |   paper   |   [code](https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2)    |   24 kHz|   5 s| 1280 |
|   HumpbackNET |   [paper](https://pubs.aip.org/asa/jasa/article/155/3/2050/3271347)   |   [code](https://github.com/vskode/acodet)    |   2 kHz|   3.9124 s| 2048|
|   Insect66NET |   [paper](https://doi.org/10.1371/journal.pcbi.1011541)   |   [code](https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet)    |   44.1 kHz|   5.5 s| 1280 |
|   Insect459NET |   [paper](https://arxiv.org/pdf/2503.15074)   |   pending    |   44.1 kHz|   5.5 s| 1280 |
|   Mix2        |   [paper](https://arxiv.org/abs/2403.09598)   |   [code](https://github.com/ilyassmoummad/Mix2/tree/main)    |   16 kHz|   3 s| 960 |
|   NatureBEATs        |   [paper](https://arxiv.org/abs/2411.07186)   |   [code](https://github.com/earthspecies/NatureLM-audio)    |   16 kHz|   10 s| 768 |
|   Perch_Bird       |   [paper](https://www.nature.com/articles/s41598-023-49989-z.epdf)   |   [code](https://github.com/google-research/perch)    |   32 kHz|   5 s| 1280 |
|   ProtoCLR     |   [paper](https://arxiv.org/pdf/2409.08589)   |   [code](https://github.com/ilyassmoummad/ProtoCLR)    |   16 kHz|   6 s| 384 |
|   RCL_FS_BSED     |   [paper](https://arxiv.org/abs/2309.08971)   |   [code](https://github.com/ilyassmoummad/RCL_FS_BSED)    |   22.05 kHz|   0.2 s| 2048 |
|   SurfPerch       |   [paper](https://arxiv.org/abs/2404.16436)   |   [code](https://www.kaggle.com/models/google/surfperch)    |   32 kHz|   5 s| 1280 |
|   VGGish      |   [paper](https://ieeexplore.ieee.org/document/7952132)   |   [code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)    |   16 kHz|   0.96 s| 128 |

<details>
<summary>Click to see more details on the models</summary>


|   Name|   paper|   code|   training|   CNN/Trafo| architecture | checkpoint link |
|---|---|---|---|---|---|---|
|   [AudioMAE](#audiomae)    |   [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)   |   [code](https://github.com/facebookresearch/AudioMAE)    | ssl + ft|   trafo| ViT | [weights](https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view)|
|   [AudioProtoPNet](#audioprotopnet)   |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954125000901)   |   [code](https://github.com/DBD-research-group/AudioProtoPNet)    |  sup l |   CNN | ConvNext | included|
|   [AvesEcho_PaSST](#avesecho_passt)   |   [paper](https://arxiv.org/abs/2409.15383)   |   [code](https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1)    |   sup l |   trafo | PaSST | [weights](https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1/-/blob/main/checkpoints/best_model_passt.pt?ref_type=heads) |
|   [AVES_ESpecies](#aves_especies)        |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   ssl|   trafo | HuBERT | [weights](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.pt)|
| [BEATs](#beats) | [paper](https://arxiv.org/abs/2212.09058) | [code](https://github.com/microsoft/unilm/tree/master/beats) | ssl | trafo | ViT | [weights](https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC) |
|   [BioLingual](#biolingual)  |   [paper](https://arxiv.org/abs/2308.04978)   |   [code](https://github.com/david-rx/biolingual)    |   ssl|   trafo| CLAP | included |
|   [BirdAVES_ESpecies](#birdaves_especies)    |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   ssl|   trafo | HuBERT | [weights](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-large.torchaudio.pt)|
|   [BirdMAE](#birdmae)    |   [paper](https://arxiv.org/abs/2504.12880)   |   [code](https://github.com/DBD-research-group/Bird-MAE)    |   ssl | trafo | ViT | included |
|   [BirdNET](#birdnet)     |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954121000273)   |   [code](https://github.com/kahst/BirdNET-Analyzer)    |   sup l|   CNN | EffNetB0 | [weights](https://github.com/kahst/BirdNET-Analyzer/tree/main/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model)|
|   [Google_Whale](#google_whale)       |   paper   |   [code](https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2)    |   sup l|   CNN| EffNetb0 | included|
|   [HumpbackNET](#humpbacknet) |   [paper](https://pubs.aip.org/asa/jasa/article/155/3/2050/3271347)   |   [code](https://github.com/vskode/acodet)    |   sup l |   CNN | ResNet50| [weights](https://github.com/vskode/acodet/blob/main/acodet/src/models/Humpback_20221130.zip)|
|   [Insect66NET](#insect66net) |   paper   |   [code](https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet)    |   sup l|   CNN | EffNetv2s | [weights](https://gitlab.com/arise-biodiversity/DSI/algorithms/cricket-cicada-detector-capgemini/-/blob/main/src/model_traced.pt?ref_type=heads)|
|   [Insect459NET](#insect459net) |   paper   |   pending    |   sup l|   CNN | EffNetv2s | pending |
|   [Mix2](#mix2)        |   [paper](https://arxiv.org/abs/2403.09598)   |   [code](https://github.com/ilyassmoummad/Mix2/tree/main)    |   sup l|   CNN| MobNetv3 | release pending|
|   [NatureBEATs](#naturebeats)        |   [paper](https://arxiv.org/abs/2411.07186)   |   [code](https://github.com/earthspecies/NatureLM-audio)    | ssl | trafo | BEATs | [weights](https://drive.google.com/file/d/12BrWRbxJsuwZHOkzX8HEpGgSMy5VnwCp/view?usp=sharing) |
|   [Perch_Bird](#perch_bird)       |   [paper](https://www.nature.com/articles/s41598-023-49989-z.epdf)   |   [code](https://github.com/google-research/perch)    |   sup l|   CNN| EffNetb0 | included |
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
- ConvNext architecture
- trained on the xeno-canto large section of BirdSet

This CNN is trained in two phases. The main contribution of this model is its interpretability. It learned prototypes during its second training phase which can be used during inference time to visualize sections of the spectrogram that were most important for classification. It also reaches competitive performance on bird classification tasks. The (include) classifier can distinguish 9736 classes. 

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

### Google_Whale
- CNN
- supervised training model
- trained on 7 whale species

Google_Whale (multispecies_whale) is a EFficientNet B0 model trained on whale vocalizations and other marine sounds.


### HumpbackNET
- CNN
- supervised training model
- trained on humpback whale song

HumpbackNET is a binary classifier based on a ResNet-50 model trained on humpback whale data from different parts in the North Atlantic. 

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





</details>

---

<details>
<summary><b style="font-size: 2em;">Add a new model</b></summary>

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
from bacpipe.model_specific_utils.newmodel.module import MyClass

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

This repository is intended to be a collaborative project for people working in the field of bioacoustics. If you think there is some improvement that could be useful, please raise an **issues**, submit a **PR** or get in touch. 

There are two main intentions for this repository that should always be considered when contributing: 
      
      1. Only add new requirements if truly necessary

Given the large number of different models, there are already a lot of requirements. To ensure that the repository is stable, and installation errors are kept minimal, please only add code with new requirements if truly necessary.
      
      2. The main purpose of bacpipe is quickly generating embeddings from models
      
There should always be a baseline minimal use case, where embeddings are created from different feature extractors and everything else is an add-on.

# Known issues

Given that this repository compiles a large number of very different deep learning models with different requirements, some issues have been noted. 

Please raise issues if there are questions or bugs. 

Previous versions of **bacpipe** included models like animal2vec, but the requirements conflicts led me to remove them. In the future I hope there will be an updated version of those models and then they will be included again.

# Citation

A lot of work has gone into creating these bioacoustic models, both by data collectors and by machine learning practitioners, please cite the authors of the respective models (all models are referenced in the table above).


This work is first described in a conference paper. If you use **bacpipe** for your research, please include the following reference:

```bibtex
@misc{kather2025clusteringnovelclassrecognition,
      title={Clustering and novel class recognition: evaluating bioacoustic deep learning feature extractors}, 
      author={Vincent S. Kather and Burooj Ghani and Dan Stowell},
      year={2025},
      eprint={2504.06710},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.06710}, 
}
```


# Newsletter and Q&A sessions

Reading from the traffic on the repository, there seems to be an interest in bacpipe. I have set up a newsletter under this link: https://buttondown.com/vskode. Once more than 30 people have signed up for the newsletter, I will schedule a Q&A session and post the link in the newsletter. Hopefully I can then help answer questions and address issues that people are running into. 