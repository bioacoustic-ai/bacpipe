from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def calc_distances(paths, all_embeds):
    if not paths.distances_path.joinpath("distances.npy").exists():
        distances = {}
        for model, embeds in tqdm(
            all_embeds.items(), desc="calculating distances", position=0, leave=False
        ):
            distances[model] = {}
            for metric in ["euclidean"]:  # "cosine",
                # d_all.append(pairwise_distances(embeds["all"], metric=metric).flatten())
                distances[model][metric] = {}
                for ind, lab in tqdm(
                    enumerate(np.unique(embeds["labels"])),
                    desc="calculating intra and inter distances",
                    position=1,
                    leave=False,
                ):
                    if lab == -1:
                        continue
                    cluster = embeds["all"][embeds["labels"] == lab]
                    if len(cluster) == 0:
                        continue
                    label_key = [
                        k for k, v in embeds["label_dict"].items() if v == lab
                    ][0]
                    distances[model][metric].update({label_key: {}})
                    all_without_cluster = embeds["all"][embeds["labels"] != lab]

                    distances[model][metric][label_key].update(
                        {
                            "intra": pairwise_distances(cluster, metric=metric)
                            .flatten()
                            .tolist()
                        }
                    )

                    distances[model][metric][label_key].update(
                        {
                            "inter": np.sort(
                                pairwise_distances(
                                    cluster, all_without_cluster, metric=metric
                                )
                            )[:, :15]
                            .flatten()
                            .tolist()
                        }
                    )

                    distances[model][metric][label_key].update(
                        {
                            "ratio": np.mean(
                                distances[model][metric][label_key]["intra"]
                            )
                            / np.mean(distances[model][metric][label_key]["inter"])
                        }
                    )

        np.save(paths.distances_path.joinpath("distances.npy"), distances)
    else:
        distances = np.load(
            paths.distances_path.joinpath("distances.npy"), allow_pickle=True
        ).item()
    return distances
