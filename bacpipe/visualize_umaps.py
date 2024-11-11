from pathlib import Path
import json
import matplotlib.pyplot as plt


def plot_embeddings(umap_embed_path):
    embeds_ar = []
    files = umap_embed_path.iterdir()
    for file in files:
        if file.suffix == ".json":
            with open(file, "r") as f:
                embeds_ar.append(json.load(f))

    plt.figure()
    for embed in embeds_ar:
        embed_x = embed["x"]
        embed_y = embed["y"]
        file_stem = Path(embed["metadata"]["audio_files"]).stem
        plt.plot(embed_x, embed_y, "o", label=file_stem, markersize=0.5)

    plt.legend()
    plt.title("UMAP embeddings")
    plt.savefig(umap_embed_path.joinpath("umap_embed.png"))


def append_timeList(meta_dict, file_idx, divisions_array=[]):
    length = meta_dict["files"]["embedding_dimensions"][file_idx][0]
    sample_length_in_s = config["preproc"]["model_time_length"]
    lin_array = np.arange(0, length * sample_length_in_s, sample_length_in_s)
    for t_s in lin_array:
        divisions_array.append(f"{int(t_s/60)}:{np.mod(t_s, 60):.2f}s")