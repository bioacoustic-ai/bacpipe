# bacpipe
BioAcoustic Collection Pipeline

This repository aims to streamline the generation and testing of embeddings using a large variety of bioacoustic models. 

Models currently include:

## Available models

|   Name|   ref paper|   ref code|   sampling rate|   input length| embedding dimension |
|---|---|---|---|---|---|
|  Animal2vec_XC|   [paper](https://arxiv.org/abs/2406.01253)   |   [code](https://github.com/livingingroups/animal2vec)    |   8 kHz (?)|   5 s| 768 |
|  Animal2vec_MK|   [paper](https://arxiv.org/abs/2406.01253)   |   [code](https://github.com/livingingroups/animal2vec)    |   8 kHz|   10 s| 1024 |
|   AudioMAE    |   [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)   |   [code](https://github.com/facebookresearch/AudioMAE)    |   16 kHz|   10 s| 768 |
|   AVES        |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s| 768 |
|   BioLingual  |   [paper](https://arxiv.org/abs/2308.04978)   |   [code](https://github.com/david-rx/biolingual)    |   48 kHz|   10 s| 512 |
|   BirdAVES    |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   16 kHz|   1 s| 768 |
|   BirdNET     |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954121000273)   |   [code](https://github.com/kahst/BirdNET-Analyzer)    |   48 kHz|   3 s| 1024 |
|   EchoPASST   |   [paper](https://arxiv.org/abs/2409.15383)   |   code    |   32 kHz|   3 s| 768 |
|   HumpbackNET |   [paper](https://pubs.aip.org/asa/jasa/article/155/3/2050/3271347)   |   [code](https://github.com/vskode/acodet)    |   2 kHz|   3.9124 s| 2048|
|   Insect66NET |   paper   |   [code](https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet)    |   44.1 kHz|   5.5 s| 1280 |
|   Mix2        |   [paper](https://arxiv.org/abs/2403.09598)   |   [code](https://github.com/ilyassmoummad/Mix2/tree/main)    |   16 kHz|   3 s| 960 |
|   Perch       |   [paper](https://www.nature.com/articles/s41598-023-49989-z.epdf)   |   [code](https://github.com/google-research/perch)    |   32 kHz|   5 s| 1280 |
|   ProtoCLR     |   [paper](https://arxiv.org/pdf/2409.08589)   |   [code](https://github.com/ilyassmoummad/ProtoCLR)    |   16 kHz|   6 s| 384 |
|   RCL_FS_BSED     |   [paper](https://arxiv.org/abs/2309.08971)   |   [code](https://github.com/ilyassmoummad/RCL_FS_BSED)    |   22.05 kHz|   0.2 s| 2048 |
|   SurfPerch       |   [paper](https://arxiv.org/abs/2404.16436)   |   [code](https://www.kaggle.com/models/google/surfperch)    |   32 kHz|   5 s| 1280 |
|   WhalePerch       |   paper   |   [code](https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2)    |   24 kHz|   5 s| 1280 |
|   UMAP        |   [paper](https://arxiv.org/abs/1802.03426)   |   [code](https://github.com/lmcinnes/umap)    |   - |   - | |
|   VGGish      |   [paper](https://ieeexplore.ieee.org/document/7952132)   |   [code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)    |   16 kHz|   0.96 s| 128 |

## Brief description of models
All information is extracted from the respective repositories and manuscripts. Please refer to them for more details

### Animal2vec_XC 
- raw waveform input
- self-supervised model
- transformer
- trained on bird song data

animal2vec model weights are from self-supervised pretraining on xeno-canto data. The model is based on data2vec2.0 with a number of bioacoustic-specific model implementations. See paper for more details.

### Animal2vec_MK 
- raw waveform input
- self-supervised pretrained model, fine-tuned
- transformer
- trained on meerkat vocalizations

animal2vec model weights are from self-supervised pretraining on meerkat data with fine tuning on a curated meerkat dataset. The model is based on data2vec2.0 with a number of bioacoustic-specific model implementations. See paper for more details.

### AudioMAE
- spectrogram input
- self-supervised pretrained model, fine-tuned
- vision transformer
- trained on general audio

AudioMAE from the facebook research group is a vision transformer pretrained on AudioSet-2M data and fine-tuned on AudioSet-20K.

### AVES
- transformer
- self-supervised pretrained model
- trained on general audio

AVES is short for Animal Vocalization Encoder based on Self-Supervision. The model is based on the HuBERT-base architecture. The model is pretrained on unannotated audio datasets AudioSet-20K, FSD50K and the animal sounds from AudioSet and VGGSound.


### BioLingual
- transformer
- spectrogram input
- contrastive-learning
- self-supervised pretrained model
- trained on animal sound data (primarily bird song)

BioLingual is a language-audio model trained on captioning bioacoustic datasets inlcuding xeno-canto and iNaturalist. The model architecture is based on the [CLAP](https://arxiv.org/pdf/2211.06687) model architecture. 

### BirdAVES
- transformer
- self-supervised pretrained model
- trained on general audio and bird song data

AVES is short for Animal Vocalization Encoder based on Self-Supervision. The model is based on the HuBERT-large architecture. The model is pretrained on unannotated audio datasets AudioSet-20K, FSD50K and the animal sounds from AudioSet and VGGSound as well as bird vocalizations from xeno-canto. 

### BirdNET
- CNN
- supervised training model
- trained on bird song data

BirdNET (v2.4) is based on a EfficientNET(b0) architecture. The model is trained on a large amount of bird vocalizations from the xeno-canto database alongside other bird song databses. 

### EchoPaSST
- transformer
- supervised pretrained model, fine-tuned
- pretrained on general audio and bird song data

EchoPaSST is a vision transformer trained on AudioSet and (deep) fine-tuned on xeno-canto. The model is based on the [PaSST](https://github.com/kkoutini/PaSST) framework. 


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


### Mix2
- CNN
- supervised training model
- trained on frog sounds

Mix2 is a [MobileNet v3](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py) model trained on the [AnuranSet](https://github.com/soundclim/anuraset) which includes sounds of 42 different species of frogs from different regions in Brazil. The model was trained using a mixture of Mixup augmentations to handle the class imbalance of the data.

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


### Perch
- CNN
- supervised training model
- trained on bird song data

Perch is a EFficientNet B1 model trained on the entire Xeno-canto database.

### SurfPerch
- CNN
- supervised training model
- trained on bird song, fine-tuned on tropical reef data

Perch is a EFficientNet B1 model trained on the entire Xeno-canto database and fine tuned on coral reef and unrelated sounds.

### WhalePerch
- CNN
- supervised training model
- trained on 7 whale species

WhalePerch (multispecies_whale) is a EFficientNet B0 model trained on whale sounds.

### UMAP
see [repo](https://github.com/lmcinnes/umap)

### VGGISH
- CNN
- supervised training model
- trained on general audio

VGGish is a model based on the [VGG](https://arxiv.org/pdf/1409.1556) architecture. The model is trained on audio from youtube videos (YouTube-8M)

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

