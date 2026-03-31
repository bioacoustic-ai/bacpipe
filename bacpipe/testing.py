import pandas as pd
import numpy as np
import json

path = '/mnt/swap/Work/Embeddings/AnuranSet/dim_reduced_embeddings/2025-11-20_15-26___umap-AnuranSet-birdnet'

with open(path+'/AnuranSet_umap.json', 'r') as f:
    embeds = json.load(f)
    
df = pd.DataFrame()
x = embeds['x']
y = embeds['y']

file_names = []
[file_names.extend([f] * ll) for f, ll in zip(embeds['metadata']['audio_files'], embeds['metadata']['nr_embeds_per_file'])]


starts = []
[starts.extend((np.arange(ll)*1000).tolist()) for ll in embeds['metadata']['nr_embeds_per_file']]

duration = [int((embeds['metadata']['segment_length (samples)'] / embeds['metadata']['sample_rate (Hz)']) * 1000)] * len(x)


limit = 731
df['starts'] = starts[:limit]
df['duration'] = duration[:limit]
df['x'] = x[:limit]
df['y'] = y[:limit]
df['file_names'] = file_names[:limit]

df.to_csv('embeddings.csv', sep=' ', index=False)

for file in df.file_names.unique():
    df_temp = df[df.file_names == file]
    df_temp.to_csv(f'{file}_embeddings.csv', sep=' ', index=False)