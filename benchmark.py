import bacpipe

model_name = 'audioprotopnet'

fnc = bacpipe.core.workflows.make_set_paths_func(bacpipe.config.audio_dir, **vars(bacpipe.settings))
paths = fnc(model_name)


        
from bacpipe.embedding_evaluation.label_embeddings import ground_truth_api_call
gt = ground_truth_api_call(
    model_name, 
    audio_dir=bacpipe.config.audio_dir,
    annotations_filename='annotations.csv',
    single_label=False,
    min_annotation_length=0.05
    )


per_file_classifications_folder = paths.class_path / 'original_classifier_outputs'
annots = per_file_classifications_folder.rglob('*.json')
import json
preds = {}
for annot in annots:
    with open(annot, 'r') as a:
        key = annot.relative_to(paths.class_path / 'original_classifier_outputs')
        preds[key] = json.load(a)
        
p = list(preds.values())[0]
pp = p
pp.pop('head')
model_length = 3
import pandas as pd
import numpy as np
df = pd.DataFrame()
start = []
end = []
audiofilename = []
species = []
for file, values in preds.items():
    for k, v in pp.items():
        if k == 'head':
            continue
        start.extend(np.array(v['time_bins_exceeding_threshold']) * model_length)
        end.extend((np.array(v['time_bins_exceeding_threshold']) + 1) * model_length)
        audiofilename.extend(
            [file] * len(v['time_bins_exceeding_threshold'])
            )
        species.extend([k] * len(v['time_bins_exceeding_threshold']))
df['start'] = start
df['end'] = end
df['audiofilename'] = audiofilename
df['label:species'] = species
l2i = {v: k for k, v in enumerate(df['label:species'].unique())}
meta_dict = {'label:species': l2i}

pred = ground_truth_api_call(
    model_name, 
    label_df = df,
    label_idx_dict=meta_dict,
    audio_dir=bacpipe.config.audio_dir,
    annotations_filename='annotations.csv',
    single_label=False,
    min_annotation_length=0.05
    )

from bacpipe.embedding_evaluation.label_embeddings import get_ground_truth
gt = get_ground_truth(model_name)