from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from bacpipe.generate_embeddings import Loader

split_by_folder = True


def plot_embeddings(umap_embed_path, axes=False, fig=False):
    files = umap_embed_path.iterdir()
    for file in files:
        if file.suffix == ".json":
            with open(file, "r") as f:
                embeds_dict = json.load(f)
    split_data = data_split_by_labels(embeds_dict)

    if not fig:
        fig, axes = plt.subplots()
        return_axes = False
    else:
        return_axes = True

    # for embed in embeds_ar:
    if split_data is not None:
        for label in split_data:
            axes.plot(
                split_data[label]["x"],
                split_data[label]["y"],
                "o",
                label=label,
                markersize=0.5,
            )
    else:
        embed_x = embeds_dict["x"]
        embed_y = embeds_dict["y"]
        file_stem = Path(embeds_dict["metadata"]["audio_files"]).stem
        axes.plot(embed_x, embed_y, "o", label=file_stem, markersize=0.5)

    if return_axes:
        return axes
    else:
        axes.legend()
        axes.set_title("UMAP embeddings")
        fig.savefig(umap_embed_path.joinpath("umap_embed.png"), dpi=300)


def data_split_by_labels(embeds_dict):
    meta = SimpleNamespace(**embeds_dict["metadata"])
    if split_by_folder:
        index = 0
        split_data = {}
        idx_increment = lambda x: meta.embedding_dimensions[x][0]
        concat = lambda x, x1: np.concatenate([x, x1])
        for idx, file in enumerate(meta.embedding_files):
            parent_dir = Path(file).relative_to(meta.embed_dir).parent.stem
            if not parent_dir in split_data:
                split_data[parent_dir] = {"x": [], "y": []}

            for k in split_data[parent_dir].keys():
                split_data[parent_dir][k] = concat(
                    split_data[parent_dir][k],
                    embeds_dict[k][index : index + idx_increment(idx)],
                )
            index += idx_increment(idx)

        return split_data


def return_rows_cols(num):
    if num > 3 and num <= 6:
        return 2, 3
    elif num > 6 and num <= 9:
        return 3, 3
    elif num > 9 and num <= 16:
        return 4, 4
    elif num > 16 and num <= 20:
        return 4, 5


def plot_comparison(audio_dir, embedding_models, dim_reduction_model):
    rows, cols = return_rows_cols(len(embedding_models))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    for idx, model in enumerate(embedding_models):
        ld = Loader(
            audio_dir, model_name=model, dim_reduction_model=dim_reduction_model
        )
        axes.flatten()[idx] = plot_embeddings(
            ld.embed_dir, axes=axes.flatten()[idx], fig=fig
        )
        axes.flatten()[idx].set_title(f"{model}")
        if idx == 0:
            axes.flatten()[0].legend()
    # fig.tight_layout()
    fig.suptitle(f"Comparison of {dim_reduction_model} embeddings", fontweight="bold")
    fig.savefig(ld.embed_dir.joinpath("comp_fig.png"), dpi=300)
