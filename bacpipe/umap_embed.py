import yaml
import json
import logging
import numpy as np
from bacpipe.generate_embeddings import generate_embeddings
from bacpipe.visualize_umaps import plot_embeddings
logger = logging.getLogger('bacpipe')


with open('bacpipe/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

def get_embeddings(check_if_primary_combination_exists=False,
                   check_if_secondary_combination_exists=False):
    
    generate_embeddings(model_name=config['embedding_model'], 
            check_if_combination_exists=check_if_primary_combination_exists)
    ld = generate_embeddings(model_name='umap', 
            check_if_combination_exists=check_if_secondary_combination_exists)
    plot_embeddings(ld.embed_dir)
    embeds, divisions_array = [], []
    for ind, file in enumerate(ld.files):
        d = json.load(open(file))
        arr = np.array([d['x'], d['x']]).reshape([len(d['x']), 2])
        embeds.append(arr)
        # append_timeList(ld.metadata_dict, ind, divisions_array)
        
    return embeds, ld.metadata_dict, divisions_array

def append_timeList(meta_dict, file_idx, divisions_array = []):
    length = meta_dict['files']['embedding_dimensions'][file_idx][0]
    sample_length_in_s = config['preproc']['model_time_length']
    lin_array = np.arange(0, length*sample_length_in_s, sample_length_in_s)
    for t_s in lin_array:
        divisions_array.append(
            f'{int(t_s/60)}:{np.mod(t_s, 60):.2f}s')
    
