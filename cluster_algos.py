import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN, KMeans, kmeans_plusplus

# import matplotlib.pyplot as plt
# from sklearn.datasets import make_moons
# # ==========================================
# # 1. Generate a large, noisy dataset
# # ==========================================
# print("Generating dataset...")
# # X, _ = make_moons(n_samples=100000, noise=0.08, random_state=42)
# # X = X * 5.0  # Scale up for visualization

# # Inject 5,000 random background noise points
# # noise = np.random.uniform(low=-10, high=15, size=(5000, 2))
# # X = np.vstack([X, noise])
# X = embeds['birdnet']['snr=0']
# print(f"-> Dataset size: {X.shape} points\n")

# # ==========================================
# # 2. Compress data with Mini-Batch K-Means
# # ==========================================
# print("Step 1: Compressing with Mini-Batch K-Means...")
# n_centroids = X.shape[0]//100  # Compress 105,000 points into 1,000 representative centroids
# kmeans = MiniBatchKMeans(n_clusters=n_centroids, random_state=42, batch_size=2048)
# kmeans.fit(X)

# # ==========================================
# # 3. Calculate weights (counts per centroid)
# # ==========================================
# print("Step 2: Extracting centroids and computing weights...")
# centroids = kmeans.cluster_centers_

# # Count how many original points were assigned to each centroid index
# weights = np.bincount(kmeans.labels_)

# # ==========================================
# # 4. Run HDBSCAN on the weighted centroids
# # ==========================================
# print("Step 3: Running HDBSCAN on weighted centroids...")
# # NOTE: Since we compressed 105,000 points into 1,000 centroids, 
# # each centroid represents ~105 original points on average.
# # Adjust min_cluster_size to reflect this "centroid-scale" (e.g., 15 centroids ≈ 1500 points)
# hdb = DBSCAN(eps=0.5, min_samples=5)
# hdb.fit(centroids, sample_weight=weights)

# # ==========================================
# # 5. Map the centroid labels back to raw points
# # ==========================================
# print("Step 4: Mapping labels back to the original dataset...")
# # This elegant, vectorized NumPy indexing maps the centroid's label 
# # to every original point assigned to that centroid in O(1) time.
# filtered_labels = weights < 30
# # final_labels = hdb.labels_[kmeans.labels_]
# labels = hdb.labels_
# labels[~filtered_labels] = -1
# final_labels = labels[kmeans.labels_]

# # ==========================================
# # 6. Analyze and Plot the Results
# # ==========================================
# unique_labels = set(final_labels)
# n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
# n_noise = np.sum(final_labels == -1)

# print("\n=== CLUSTERING RESULTS ===")
# print(f"Discovered Clusters: {n_clusters}")
# print(f"Noise Points Rejected: {n_noise} ({n_noise / len(X) * 100:.2f}% of data)")

# # Quick check on how fast it was
# # Plotting the raw points colored by their mapped HDBSCAN cluster
# plt.figure(figsize=(10, 6))
# # plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', s=1, alpha=0.5)
# plt.scatter(umaps['birdnet']['snr=0']['x'], umaps['birdnet']['snr=0']['y'], c=final_labels, cmap='viridis', s=1, alpha=0.5)
# plt.title(f"Hybrid K-Means + HDBSCAN\n({n_clusters} clusters found, noise colored dark purple/blue)")
# plt.colorbar(label="Cluster ID")
# plt.savefig('test.png')




class Clustering_Approach:
    def __init__(
        self, 
        max_cluster_size, 
        n_centroids, 
        filter_centroids=False, 
        ### kmeans params
        initial_clustering='minibatchkmeans', 
        tol=1e-4, 
        max_iter=300, 
        n_init=10,
        init='k-means++', 
        ### general params
        random_state=42, 
        ### minibatch params
        batch_size=2048, 
        ### dbscan params
        dcscan_eps=0.5, 
        dbscan_min_samples=5, 
        ### agglomerative clustering metrics
        agglomerative_clustering = False,
        n_neighbors=5
        ):
        """
        init{‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’

            Method for initialization:

                ‘k-means++’ : selects initial cluster centroids using sampling based on an empirical probability distribution of the points’ contribution to the overall inertia. This technique speeds up convergence. The algorithm implemented is “greedy k-means++”. It differs from the vanilla k-means++ by making several trials at each sampling step and choosing the best centroid among them.

                ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.

                If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

                If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.

            For an example of how to use the different init strategies, see A demo of K-Means clustering on the handwritten digits data.

            For an evaluation of the impact of initialization, see the example Empirical evaluation of the impact of k-means initialization.
        n_init‘auto’ or int, default=’auto’

            Number of times the k-means algorithm is run with different centroid seeds. The final results is the best output of n_init consecutive runs in terms of inertia. Several runs are recommended for sparse high-dimensional problems (see Clustering sparse data with k-means).

            When n_init='auto', the number of runs depends on the value of init: 10 if using init='random' or init is a callable; 1 if using init='k-means++' or init is an array-like.
        
        """
        ### old umap code
        # umap4clust = dict()
        # for key, val in embeds.items():
        #     umap4clust[key] = dict()
        #     for snr_key, ems in val.items():
        #         umap4clust[key][snr_key] = umap_kmeans(ems, n_centroids, 42)

        # clust_df, cluster_booleans, clust_results = fetch_clustering(umap4clust, clustering_dict, overwrite=True)

        
        # n_centroids = X.shape[0]//100  # Compress 105,000 points into 1,000 representative centroids
        self.max_cluster_size = max_cluster_size
        self.filter_centroids = filter_centroids
        
        self.agglomerative_clustering = agglomerative_clustering
        self.n_neighbors = n_neighbors
        
        if initial_clustering == 'minibatchkmeans':
            self.kmeans = MiniBatchKMeans(n_clusters=n_centroids, random_state=random_state, batch_size=batch_size)
        elif initial_clustering == 'kmeans':
            self.kmeans = KMeans(n_clusters=n_centroids, random_state=random_state, tol=tol, max_iter=max_iter, init=init, n_init=n_init)
        elif initial_clustering == 'kmeans+umap':
            import umap
            def umap_kmeans(X, n_clusters, random_state): 
                clusterer = umap.UMAP(
                **{"n_neighbors": 15, 
                   "min_dist": 0.1, 
                   "n_components": n_clusters, 
                   "metric": "euclidean", 
                   "random_state": random_state}
                )
                X = X.swapaxes(0, 1)
                centroids = clusterer.fit_transform(X)
                return centroids.swapaxes(0, 1)
            self.kmeans = KMeans(n_clusters=n_centroids, random_state=random_state, tol=tol, max_iter=max_iter, init=umap_kmeans, n_init=3)
            
        self.dbscan = DBSCAN(eps=dcscan_eps, min_samples=dbscan_min_samples)
        
    def fit_predict(self, X):
        print("Step 1: Compressing with K-Means...")
        self.kmeans.fit(X)

        
        print("Step 2: Extracting centroids and computing weights...")
        centroids = self.kmeans.cluster_centers_

        # Count how many original points were assigned to each centroid index
        weights = np.bincount(self.kmeans.labels_)


        
        if self.filter_centroids:
            from sklearn.metrics import pairwise_distances
            pp = pairwise_distances(centroids)
            bool_smaller_than_3std = [True if any(p<(np.mean(pp)-3*np.std(pp)) * (p>0)) else False for p in pp]
            labels[bool_smaller_than_3std] = -3 
        
        if self.agglomerative_clustering:
            print("Step 3: Running DBSCAN on weighted centroids...")
            final_labels = self.agglomerative_clust(centroids)
        
        else:
            print("Step 3: Running kNN and then agglomerative clustering on weighted centroids...")

            self.dbscan.fit(centroids, sample_weight=weights)
            labels = self.dbscan.labels_
            filtered_labels = weights < self.max_cluster_size
            labels[~filtered_labels] = -2
            
            final_labels = labels[self.kmeans.labels_]
        # ==========================================
        # 5. Map the centroid labels back to raw points
        # ==========================================
        print("Step 4: Mapping labels back to the original dataset...")
        # This elegant, vectorized NumPy indexing maps the centroid's label 
        # to every original point assigned to that centroid in O(1) time.
        
        # final_labels = hdb.labels_[kmeans.labels_]

        # ==========================================
        # 6. Analyze and Plot the Results
        # ==========================================
        unique_labels = set(final_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(final_labels == -1)

        print("\n=== CLUSTERING RESULTS ===")
        print(f"Discovered Clusters: {n_clusters}")
        print(f"Noise Points Rejected: {n_noise} ({n_noise / len(X) * 100:.2f}% of data)")
        
        return final_labels, centroids, weights
    

    def agglomerative_clust(self, centroids):    
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import NearestNeighbors
        
        import numpy as np
        from scipy.spatial.distance import pdist

        # Returns a 1D vector of length (N * (N - 1)) / 2
        # Contains no 0 diagonals and no duplicate pairs!
        distances = pdist(centroids)

        threshold = np.mean(distances) - 2 * np.std(distances)
        
        # 1. Build the k-NN graph directly from NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(centroids)
        knn_graph = nbrs.kneighbors_graph(centroids, mode='distance')
        
        # 2. Cluster without setting n_clusters
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            connectivity=knn_graph,
            linkage='single'         # 'single' matches connected components behavior
        ).fit(centroids)
        
        labels = clustering.labels_
        n_classes = len(set(labels))
        
        print(f"Discovered {n_classes} classes.")
        
        new_labels = self.kmeans.labels_
        for idx_centroid in np.arange(len(centroids)):
            new_labels[new_labels == idx_centroid] = labels[idx_centroid]
            
        for idx_centroid in np.arange(len(centroids)):
            if len(new_labels[new_labels == idx_centroid]) > self.max_cluster_size:
                new_labels[new_labels == idx_centroid] = -2
        
        return new_labels

# def plot_embeds_with_centroids():
#     from matplotlib import pyplot as plt
    
#     from sklearn.metrics import pairwise_distances
#     pp = pairwise_distances(centroids)#[tl_mask])
    
#     counts, bins = np.histogram(pp, bins=25)
#     plt.figure()
#     plt.hist(bins[:-1], bins, weights=counts)
#     plt.savefig('test-hist.png')
    
#     # pp[pp<4] = -1
#     plt.figure()
#     plt.imshow(pp)
#     plt.colorbar()
#     plt.savefig('test_heat1.png')

# def plot_embeds_with_centroids():
#     from matplotlib import pyplot as plt
    
#     plt.figure()
#     plt.scatter(umap1.embedding_[:, 0], umap1.embedding_[:, 1], label='embeds')
#     plt.scatter(cen[:, 0], cen[:, 1], label='centroids')
#     plt.savefig('test.png')

# ####
# from matplotlib import pyplot as plt

# plt.figure()
# plt.scatter(umap1.embedding_[:, 0], umap1.embedding_[:, 1], s=0.5, c=new_labels, label='embeds')
# mask = self.dbscan.labels_ >= 0
# # plt.scatter(cen[mask][:, 0], cen[mask][:, 1], s=8, c='blue', label='clust')
# # plt.scatter(cen[~mask][:, 0], cen[~mask][:, 1], s=12, c='green', label='neg')
# # plt.scatter(cen[tl_mask][:, 0], cen[tl_mask][:, 1], s=12, c='red', label='tl')
# plt.scatter(cen[:, 0], cen[:, 1], s=12, c=labels, label='tl')

# plt.legend()
# plt.savefig('test.png')
# plt.close()

####
    