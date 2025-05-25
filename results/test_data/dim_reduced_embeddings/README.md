# Reduced dimension embeddings directory 

By default the naming conventions for the embedding directories is:

`_year-month-day_hour-minute___DimReductionModelName-DatasetName-ModelName`

Inside this directory you will find:
- `metadata.yml`, which contains metadata in the shape of a dictionary with the following keys:
    - audio_dir: _data_directory_
    - embed_dir: _path_to_embedding_files_
    - embedding_size: _dimension_of_embeddings_
    - files: _dictionary_with_lists_containing_per_file_information_

        - audio_files: _list_of_audio_files_
        - embedding_dimensions: _tuple_of_number_of_embeddings_by_embedding_size_
        - embedding_files: _name_of_embedding_files_
        - file_lengths (s): _file_length_in_seconds_
        - nr_embeds_per_file: _number_of_embeddings_per_file_
    - model_name: _name_of_model_
    - nr_embeds_total: _total_number_of_embeddings_
    - sample_rate (Hz): _sample_rate_
    - segment_length (samples): _length_of_input_segment_in_samples_
    - total_dataset_length (s): _total_length_of_dataset_in_seconds_
- one `.json` file containing the reduced embeddings of the entire dataset
- a plot, visiualizing the reduced embeddings in 2d

### It is important that the name of this directory remains unchanged, so that it can be found automatically. 
