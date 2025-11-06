import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bacpipe.main import model_specific_embedding_creation
from bacpipe import config, settings

config.overwrite = False
loader_dict = model_specific_embedding_creation(
        # check_if_primary_combination_exists=False,
        # check_if_secondary_combination_exists=False,
        **vars(config), **vars(settings)
)

clust_dict = {}

for model_name, model_dict in loader_dict.items():
        # clustering = np.load(model_dict.paths.clust_path / 'clust_labels.npy', allow_pickle=True).item()
        ground_truth = np.load(model_dict.paths.labels_path / 'ground_truth.npy', allow_pickle=True).item()
        clust_dict.update({
                # model_name: {**clustering, **ground_truth}
                model_name: {**ground_truth}
        })

embeds = {}
for model_name, model_dict in loader_dict.items():
        embeds[model_name] = np.vstack(list(model_dict.embedding_dict().values()))

## compute clusterings

# kmeans
from sklearn.cluster import KMeans, SpectralClustering
from hdbscan import HDBSCAN

kmeans = KMeans(n_clusters=16) # because 15 species + noise for the within and diff file ...?
hdb = HDBSCAN(min_cluster_size=5, min_samples=8)
spec = SpectralClustering(n_clusters=16)

for model_name in loader_dict:
        clust_dict[model_name]['kmeans'] = kmeans.fit_predict(embeds[model_name])
        clust_dict[model_name]['hdb'] = hdb.fit_predict(embeds[model_name])
        # clust_dict[model_name]['spec'] = spec.fit_predict(embeds[model_name])




from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI

clust_results = {}

for model_name, clusterings in clust_dict.items():
        clust_results[model_name] = {}
        for name, clust in clust_dict[model_name].items():
                clust_results[model_name].update({
                                name: AMI(clust, clusterings['label:species'])
                })
        # for clust_name in ['kmeans', 'hdbscan']:
        #         clust_results[model_name].update({
        #                         clust_name: AMI(clusterings[clust_name], 
        #                                         clusterings['label:species'])
        #         })
        # for clust_name in ['kmeans_no_noise', 'hdbscan_no_noise']:
        #         clust_results[model_name].update({
        #                         clust_name: AMI(clusterings[clust_name], 
        #                                         clusterings['label:species'][clusterings['label:species']>=0])
        #         })
                
print(clust_results)