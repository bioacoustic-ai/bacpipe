from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from bacpipe.generate_embeddings import Loader

split_by_folder = True


def get_centroid(data):
    return np.mean(data["x"]), np.mean(data["y"])


def get_ari_and_ami(split_data, centroids):
    from sklearn.metrics import adjusted_rand_score as ari_score
    from sklearn.metrics import adjusted_mutual_info_score as ami_score
    from sklearn.metrics import silhouette_score

    x = []
    y = []

    preds = []
    labels = []
    acc_idx = 0
    n_labels = np.arange(len(split_data))
    label_dict = {k: v for k, v in zip(split_data.keys(), n_labels)}
    for label in split_data:
        x.append(split_data[label]["x"])
        y.append(split_data[label]["y"])

        for i in range(len(split_data[label]["x"])):
            p = {}
            for c_label, centroid in centroids.items():
                p[c_label] = np.linalg.norm(
                    np.array(centroid)
                    - np.array(
                        [(split_data[label]["x"][i]), (split_data[label]["y"][i])]
                    )
                )
            preds.append(label_dict[list(p.keys())[np.argmin(list(p.values()))]])
            labels.append(label_dict[label])
            acc_idx += 1
    ari = ari_score(labels, preds)
    ami = ami_score(labels, preds)

    # Silhouette Score
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    data = np.column_stack((x, y))
    ss = silhouette_score(data, labels)
    return {"ARI": ari, "AMI": ami, "Silhouette Score": ss}


def plot_embeddings(umap_embed_path, dim_reduction_model, axes=False, fig=False):
    files = umap_embed_path.iterdir()
    centroids = {}
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
            centroids[label] = get_centroid(split_data[label])
            axes.plot(
                centroids[label][0], centroids[label][1], "x", label=f"{label} centroid"
            )
        clustering_dict = get_ari_and_ami(split_data, centroids)
    else:
        embed_x = embeds_dict["x"]
        embed_y = embeds_dict["y"]
        file_stem = Path(embeds_dict["metadata"]["audio_files"]).stem
        axes.plot(embed_x, embed_y, "o", label=file_stem, markersize=0.5)

    if return_axes:
        return axes, clustering_dict
    else:
        print(clustering_dict)
        print(f"See figures at {umap_embed_path.joinpath('embed.png')}")
        axes.legend()
        axes.set_title(f"{dim_reduction_model.upper()} embeddings")
        fig.savefig(umap_embed_path.joinpath("embed.png"), dpi=300)


def data_split_by_labels(embeds_dict):
    meta = SimpleNamespace(**embeds_dict["metadata"])
    if split_by_folder:
        index = 0
        split_data = {}
        idx_increment = lambda x: meta.embedding_dimensions[x][0]
        concat = lambda x, x1: np.concatenate([x, x1])
        for idx, file in enumerate(meta.embedding_files):
            parent_dir = Path(file).parent.stem
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
    if num <= 3:
        return 1, 3
    elif num > 3 and num <= 6:
        return 2, 3
    elif num > 6 and num <= 9:
        return 3, 3
    elif num > 9 and num <= 16:
        return 4, 4
    elif num > 16 and num <= 20:
        return 4, 5


def plot_comparison(audio_dir, embedding_models, dim_reduction_model):
    rows, cols = return_rows_cols(len(embedding_models))
    clust_dict = {}
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9
    )
    for idx, model in enumerate(embedding_models):
        ld = Loader(
            audio_dir, model_name=model, dim_reduction_model=dim_reduction_model
        )
        axes.flatten()[idx], clust_dict[model] = plot_embeddings(
            ld.embed_dir, axes=axes.flatten()[idx], fig=fig
        )
        metric_str = ", ".join([f"{k}={v:.3f}" for k, v in clust_dict[model].items()])
        axes.flatten()[idx].set_title(f"{model}\n{metric_str}")
    # fig.tight_layout()
    new_order = [
        k[0] for k in sorted(clust_dict.items(), key=lambda kv: kv[1]["ARI"])[::-1]
    ]
    positions = {mod: ax.get_position() for mod, ax in zip(new_order, axes.flatten())}
    for model, ax in zip(embedding_models, axes.flatten()):
        ax.set_position(positions[model])
    # plt.legend(loc='lower left', ncol=6, bbox_to_anchor=(0, 0, 1, 1))
    fig.suptitle(f"Comparison of {dim_reduction_model} embeddings", fontweight="bold")
    fig.savefig(ld.embed_dir.joinpath("comp_fig.png"), dpi=300)
