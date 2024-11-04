# bacpipe
BioAcoustic Collection Pipeline

This repository aims to streamline the generation and testing of embeddings using a large variety of bioacoustic models. 

Models currently include:

## Available models

|   Name|   ref paper|   ref code|   sampling rate|   input length| embedding dimension |
|---|---|---|---|---|
|  Animal2vec_XC|   [paper](https://arxiv.org/abs/2406.01253)   |   [code](https://github.com/livingingroups/animal2vec)    |   8 kHz (?)|   5 s| 768 |
|  Animal2vec_MK|   [paper](https://arxiv.org/abs/2406.01253)   |   [code](https://github.com/livingingroups/animal2vec)    |   8 kHz|   10 s| 1024 |
|   AudioMAE    |   [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)   |   [code](https://github.com/facebookresearch/AudioMAE)    |   16 kHz|   10 s| 768 |
|   Aves        |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s| 768 |
|   BioLingual  |   [paper](https://arxiv.org/abs/2308.04978)   |   [code](https://github.com/david-rx/biolingual)    |   48 kHz|   10 s| |
|   BirdAves    |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s| |
|   BirdNET     |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954121000273)   |   [code](https://github.com/kahst/BirdNET-Analyzer)    |   48 kHz|   3 s| |
|   EchoPASST   |   [paper](https://arxiv.org/abs/2409.15383)   |   code    |   32 kHz|   3 s| |
|   HumpbackNET |   [paper](https://pubs.aip.org/asa/jasa/article/155/3/2050/3271347)   |   [code](https://github.com/vskode/acodet)    |   2 kHz|   3.9124 s| 2048|
|   Insect66NET |   paper   |   [code](https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet)    |   44.1 kHz|   5.5 s| |
|   Mix2        |   [paper](https://arxiv.org/abs/2403.09598)   |   [code](https://github.com/ilyassmoummad/Mix2/tree/main)    |   16 kHz|   3 s| |
|   Perch       |   [paper](https://www.nature.com/articles/s41598-023-49989-z.epdf)   |   [code](https://github.com/google-research/perch)    |   32 kHz|   5 s| |
|   UMAP        |   [paper](https://arxiv.org/abs/1802.03426)   |   [code](https://github.com/lmcinnes/umap)    |   - |   - | |
|   VGGish      |   [paper](https://ieeexplore.ieee.org/document/7952132)   |   [code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)    |   16 kHz|   0.96 s| |

## Installation

Create a virtual environment using python3.11 and virtualenv
`python3.11 -m virtualenv env_bacpipe`

activate the environment

`source env_bacpipe/bin/activate`

before installing the requirements, ensure you have the following:
- for `fairseq` to install you will need python headers:
`sudo apt-get install python3.11-dev`
- pip version 24.0 (`pip install pip==24.0`, omegaconf 2.0.6 has a non-standard dependency specifier PyYAML>=5.1.*. pip 24.1 will enforce this behaviour change and installation will thus fail.)

once the prerequisites are satisfied:

`pip install -r requirements.txt`

