# Embeddings directory 

### Embeddings that are created will automatically be saved here unless specified differently in bacpipe/path_settings.yaml

By default the naming conventions for the embedding directories is:

`\_year-month-day_hour-minute\_\_\_modelname\_datasetname`

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
- either the embeddings files (ending on `.npy`) or directories corresponding to subdirectories in the dataset folder

### It is important that the name of this directory remains unchanged, so that it can be found automatically. 
