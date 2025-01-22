import numpy as np
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import silhouette_score


def get_centroid(data):
    return np.mean(data["x"]), np.mean(data["y"])


def get_ari_and_ami(split_data, centroids):

    x = []
    y = []

    preds = []
    labels = []
    acc_idx = 0
    cluster_score_dict = {}
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
    cluster_score_dict["ARI"] = ari
    cluster_score_dict["AMI"] = ami

    # Silhouette Score
    if np.unique(labels).shape[0] > 1:
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        data = np.column_stack((x, y))
        ss = silhouette_score(data, labels)
        cluster_score_dict["Silhouette Score"] = ss
    return cluster_score_dict
