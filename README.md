# bacpipe
BioAcoustic Collection Pipeline

This repository aims to streamline the generation and testing of embeddings using a large variety of bioacoustic models. 

Models currently include:

## Available models

|   Name|   ref paper|   ref code|   sampling rate|   input length|
|---|---|---|---|---|
|  Animal2vec_XC|   paper   |   [code](https://github.com/livingingroups/animal2vec)    |   8 kHz (?)|   5 s|
|  Animal2vec_MK|   paper   |   [code](https://github.com/livingingroups/animal2vec)    |   8 kHz|   10 s|
|   AudioMAE    |   paper   |   [code](https://github.com/facebookresearch/AudioMAE)    |   16 kHz|   10 s|
|   Aves        |   paper   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s|
|   BioLingual  |   paper   |   [code](https://github.com/david-rx/biolingual)    |   48 kHz|   10 s|
|   BirdAves    |   paper   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s|
|   BirdNET     |   paper   |   [code](https://github.com/kahst/BirdNET-Analyzer)    |   48 kHz|   3 s|
|   EchoPASST   |   paper   |   code    |   32 kHz|   3 s|
|   HumpbackNET |   paper   |   [code](https://github.com/vskode/acodet)    |   2 kHz|   3.9124 s|
|   Insect66NET |   paper   |   [code](https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet)    |   44.1 kHz|   5.5 s|
|   Mix2        |   paper   |   [code](https://github.com/ilyassmoummad/Mix2/tree/main)    |   16 kHz|   3 s|
|   Perch       |   paper   |   [code](https://github.com/google-research/perch)    |   32 kHz|   5 s|
|   UMAP        |   paper   |   [code](https://github.com/lmcinnes/umap)    |   - |   - |
|   VGGish      |   paper   |   [code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)    |   16 kHz|   0.96 s|

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

