# bacpipe
BioAcoustic Collection Pipeline

This repository aims to streamline the generation and testing of embeddings using a large variety of bioacoustic models. 

Models currently include:


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

