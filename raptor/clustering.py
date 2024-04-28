import numpy as np
from sklearn.mixture import GaussianMixture
import logging
from typing import List, Optional
import umap


class GMMClustering:
    def __init__(self, n_init: int = 5, max_cluster_tokens: int = 2048):
        self.n_init = n_init
        self.max_cluster_tokens = max_cluster_tokens
        # self.max_cluster_size =max_cluster_size

    def find_optimal_components(self, vectors: np.ndarray, min_components: int = 1, max_components: int = 10, n_iter: int = 10) -> int:
        """
        Find the optimal number of components for the Gaussian mixture model using the Bayesian information criterion (BIC).

        Args:
            min_components (int, optional): The minimum number of components to consider. Defaults to 1.
            max_components (int, optional): The maximum number of components to consider. Defaults to 10.
            n_iter (int, optional): The number of iterations to perform for each number of components. Defaults to 10.

        Returns:
            The optimal number of components.
        """
        bic = []
        max_components = min(max_components, len(vectors) - 1)
        for n in range(min_components, max_components + 1):
            gmm = GaussianMixture(n_components=n, n_init=self.n_init)
            gmm.fit(vectors)
            bic.append(gmm.bic(vectors))
            logging.debug(f"BIC for {n} components: {bic[-1]}")
        return np.argmin(bic) + min_components

    def reduce_dimensions(self, vectors: np.ndarray, n_components: int, n_neighbors: Optional[int] = None, n_iter: Optional[int] = 10) -> np.ndarray:
        """
        Reduce the dimensionality of the input vectors using UMAP.

        Args:
            n_components (int): The number of components to use for the UMAP projection.
            n_neighbors (int): The number of nearest neighbors to consider for each point.
            n_iter (int, optional): The number of iterations to perform for each number of components. Defaults to 10.

        Returns:
            The reduced dimensionality of the input vectors.
        """
        n_components = min(n_components, len(vectors) - 2)
        if n_neighbors is None:
            n_neighbors = int((len(vectors) - 1) ** 0.5)
        umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
        return umap_model.fit_transform(vectors)


    def cluster_vectors(self, vectors: np.ndarray, reduced_dimension: int, threshold: float = None):
        """
        Perform recursive GMM clustering on the input vectors.

        Args:
            max_membership (int): The maximum membership of each cluster.
            depth (int, optional): The current depth of recursion. Defaults to 0.
            parent_cluster (int, optional): The index of the parent cluster. Defaults to None.

        Returns:
            A dictionary containing the clusters and their members.
        """
        vectors = self.reduce_dimensions(vectors, reduced_dimension)
        n_components = self.find_optimal_components(vectors)
        gmm = GaussianMixture(n_components=n_components, n_init=self.n_init)
        gmm.fit(vectors)
        # Vectors can be assigned to multiple clusters
        probs = gmm.predict_proba(vectors)
        if threshold is None:
            logging.info(f"Threshold is None, using better-than-uniform probability")
            threshold = 1 / n_components

        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_components


    def cluster_nodes(self, nodes: List["Node"], embedding_model_name: str, recursion_level=0, reduced_dimension=10):
        cluster_token_count = sum([node.token_count for node in nodes])

        if cluster_token_count <= self.max_cluster_tokens:
            print(f"Cluster has {cluster_token_count} tokens across {len(nodes)} nodes. Not clustering anymore.")
            return [nodes]
        # if len(nodes) <= self.max_cluster_size:
        #     print(f"Cluster has {len(nodes)} nodes. Not clustering anymore.")
        #     return [nodes]

        vectors = np.array([node.embedding[embedding_model_name] for node in nodes])
        labels, n_components = self.cluster_vectors(vectors, reduced_dimension)
        print(f"\n[LVL {recursion_level}] ### Clustered {len(vectors)} vectors into {n_components} base clusters ###")

        final_clusters = []
        for cluster_id in range(n_components):
            membership_mask = [cluster_id in label for label in labels]
            member_nodes = [node for c, node in enumerate(nodes) if membership_mask[c]]
            sub_cluster_nodes = self.cluster_nodes(member_nodes, embedding_model_name, recursion_level=recursion_level + 1)
            final_clusters.extend(sub_cluster_nodes)
        print(f"N final_clusters: {len(final_clusters)}\n")

        return final_clusters
