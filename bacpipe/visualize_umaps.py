from pathlib import Path
import json
import matplotlib.pyplot as plt


def plot_embeddings(umap_embed_path):
    embeds_ar = []
    files = umap_embed_path.iterdir()
    for file in files:
        if file.suffix == '.json':
            with open(file, 'r') as f:
                embeds_ar.append(json.load(f))
    
    plt.figure()
    for embed in embeds_ar:
        embed_x = embed['x']
        embed_y = embed['y']
        file_stem = Path(embed['metadata']['audio_files']).stem
        plt.plot(embed_x, embed_y, 'o', label=file_stem)

    plt.legend()
    plt.title('UMAP embeddings')
    plt.savefig(umap_embed_path.joinpath('umap_embed.png'))