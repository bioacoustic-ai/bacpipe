# This directory contains all model-specific pipelines to generate embeddings

# Available models

|   Name|   paper|   code|   training|   CNN/Trafo| architecture | checkpoint link |
|---|---|---|---|---|---|---|
|  [Animal2vec_XC](#animal2vec_xc)|   [paper](https://arxiv.org/abs/2406.01253)   |   [code](https://github.com/livingingroups/animal2vec)    |   ssl + ft|   trafo | d2v2.0 | release pending |
|  [Animal2vec_MK](#animal2vec_mk) |   [paper](https://arxiv.org/abs/2406.01253)   |   [code](https://github.com/livingingroups/animal2vec)    |   ssl + ft|   trafo | d2v2.0 | [weights](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.ETPUKU)|
|   [AudioMAE](#audiomae)    |   [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)   |   [code](https://github.com/facebookresearch/AudioMAE)    | ssl + ft|   trafo| ViT | [weights](https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view)|
|   [AVES_ESpecies](#aves_especies)        |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   ssl|   trafo | HuBERT | [weights](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.pt)|
|   [BioLingual](#biolingual)  |   [paper](https://arxiv.org/abs/2308.04978)   |   [code](https://github.com/david-rx/biolingual)    |   ssl|   trafo| CLAP | included |
|   [BirdAVES_ESpecies](#birdaves_especies)    |   [paper](https://arxiv.org/abs/2210.14493)   |   [code](https://github.com/earthspecies/aves)    |   ssl|   trafo | HuBERT | [weights](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-bioxn-large.torchaudio.pt)|
|   [BirdNET](#birdnet)     |   [paper](https://www.sciencedirect.com/science/article/pii/S1574954121000273)   |   [code](https://github.com/kahst/BirdNET-Analyzer)    |   sup l|   CNN | EffNetB0 | [weights](https://github.com/kahst/BirdNET-Analyzer/tree/main/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model)|
|   [AvesEcho_PaSST](#avesecho_passt)   |   [paper](https://arxiv.org/abs/2409.15383)   |   [code](https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1)    |   sup l |   trafo | PaSST | [weights](https://gitlab.com/arise-biodiversity/DSI/algorithms/avesecho-v1/-/blob/main/checkpoints/best_model_passt.pt?ref_type=heads) |
|   [HumpbackNET](#humpbacknet) |   [paper](https://pubs.aip.org/asa/jasa/article/155/3/2050/3271347)   |   [code](https://github.com/vskode/acodet)    |   sup l |   CNN | ResNet50| [weights](https://github.com/vskode/acodet/blob/main/acodet/src/models/Humpback_20221130.zip)|
|   [Insect66NET](#insect66net) |   paper   |   [code](https://github.com/danstowell/insect_classifier_GDSC23_insecteffnet)    |   sup l|   CNN | EffNetv2s | [weights](https://gitlab.com/arise-biodiversity/DSI/algorithms/cricket-cicada-detector-capgemini/-/blob/main/src/model_traced.pt?ref_type=heads)|
|   [Insect459NET](#insect459net) |   paper   |   pending    |   sup l|   CNN | EffNetv2s | pending |
|   [Mix2](#mix2)        |   [paper](https://arxiv.org/abs/2403.09598)   |   [code](https://github.com/ilyassmoummad/Mix2/tree/main)    |   sup l|   CNN| MobNetv3 | release pending|
|   [Perch_Bird](#perch_bird)       |   [paper](https://www.nature.com/articles/s41598-023-49989-z.epdf)   |   [code](https://github.com/google-research/perch)    |   sup l|   CNN| EffNetb0 | included |
|   [ProtoCLR](#protoclr)     |   [paper](https://arxiv.org/pdf/2409.08589)   |   [code](https://github.com/ilyassmoummad/ProtoCLR)    |   sup cl|   trafo| CvT-13 | [weights](https://huggingface.co/ilyassmoummad/ProtoCLR)|
|   [RCL_FS_BSED](#rcl_fs_bsed)     |   [paper](https://arxiv.org/abs/2309.08971)   |   [code](https://github.com/ilyassmoummad/RCL_FS_BSED)    |   sup cl|   CNN| ResNet9 | [weights](https://zenodo.org/records/11353694)|
|   [SurfPerch](#surfperch)       |   [paper](https://arxiv.org/abs/2404.16436)   |   [code](https://www.kaggle.com/models/google/surfperch)    |   sup l|   CNN| EffNetb0 | included |
|   [Google_Whale](#google_whale)       |   paper   |   [code](https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2)    |   sup l|   CNN| EffNetb0 | included|
|   [VGGish](#vggish)      |   [paper](https://ieeexplore.ieee.org/document/7952132)   |   [code](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)    |   sup l|   CNN| VGG | [weights](https://storage.googleapis.com/audioset/vggish_model.ckpt)|

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

### AVES_ESpecies
- transformer
- self-supervised pretrained model
- trained on general audio

AVES_ESpecies is short for Animal Vocalization Encoder based on Self-Supervision by the Earth Species Project. The model is based on the HuBERT-base architecture. The model is pretrained on unannotated audio datasets AudioSet-20K, FSD50K and the animal sounds from AudioSet and VGGSound.


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

### BirdNET
- CNN
- supervised training model
- trained on bird song data

BirdNET (v2.4) is based on a EfficientNET(b0) architecture. The model is trained on a large amount of bird vocalizations from the xeno-canto database alongside other bird song databses. 

### AvesEcho_PaSST
- transformer
- supervised pretrained model, fine-tuned
- pretrained on general audio and bird song data

AvesEcho_PaSST is a vision transformer trained on AudioSet and (deep) fine-tuned on xeno-canto. The model is based on the [PaSST](https://github.com/kkoutini/PaSST) framework. 


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

### Google_Whale
- CNN
- supervised training model
- trained on 7 whale species

Google_Whale (multispecies_whale) is a EFficientNet B0 model trained on whale vocalizations and other marine sounds.


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
from bacpipe.model_specific_utils.newmodel.module import MyClass
from .utils import ModelBaseClass

SAMPLE_RATE = 12345
LENGTH_IN_SAMPLES = int(10 * SAMPLE_RATE)


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

    @torch.inference_mode()
    def __call__(self, input):
        embeddings = self.model(input)
        return embeddings

```

Most of the models are based on pytorch. For tensorflow models, see __birdnet__, __hbdet__ or __vggish__.