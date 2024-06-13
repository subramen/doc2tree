import umap
import numpy as np
import logging
from sklearn.mixture import GaussianMixture
from typing import List, Optional


class GMMClustering:
    def __init__(
        self, n_init, max_cluster_size, max_cluster_tokens, reduced_dim: int = 10
    ):
        """
        Initialize the GMMClustering object.

        Args:
            n_init (int): The number of initializations to perform for the Gaussian mixture model.
            max_cluster_size (int): The maximum number of nodes in a cluster before it needs to be reclustered.
            max_cluster_tokens (int): The maximum number of tokens in a cluster before it needs to be reclustered.
            reduced_dim (int, optional): The number of dimensions to reduce the input vectors to. Defaults to 10.
        """
        self.n_init = n_init
        self.max_cluster_tokens = (
            max_cluster_tokens  # clusters over this size will need to be reclustered
        )
        self.max_cluster_size = (
            max_cluster_size  # clusters over this size will need to be reclustered
        )
        self.reduced_dim = reduced_dim

    def find_optimal_components(
        self,
        vectors: np.ndarray,
        min_components: int = 1,
        max_components: int = 10,
        n_iter: int = 10,
    ) -> int:
        """
        Find the optimal number of components for the Gaussian mixture model using the Bayesian information criterion (BIC).

        Args:
            vectors (np.ndarray): The input vectors to fit the Gaussian mixture model to.
            min_components (int, optional): The minimum number of components to consider. Defaults to 1.
            max_components (int, optional): The maximum number of components to consider. Defaults to 10.
            n_iter (int, optional): The number of iterations to perform for each number of components. Defaults to 10.

        Returns:
            int: The optimal number of components.
        """
        bic = []
        max_components = min(max_components, len(vectors) - 1)
        for n in range(min_components, max_components + 1):
            gmm = GaussianMixture(n_components=n, n_init=self.n_init)
            gmm.fit(vectors)
            bic.append(gmm.bic(vectors))
            logging.debug(f"BIC for {n} components: {bic[-1]}")
        return np.argmin(bic) + min_components

    def reduce_dimensions(
        self,
        vectors: np.ndarray,
        n_components: int,
        n_neighbors: Optional[int] = None,
        n_iter: Optional[int] = 10,
    ) -> np.ndarray:
        """
        Reduce the dimensionality of the input vectors using UMAP.

        Args:
            vectors (np.ndarray): The input vectors to reduce dimensions.
            n_components (int): The number of components to use for the UMAP projection.
            n_neighbors (int): The number of nearest neighbors to consider for each point.
            n_iter (int, optional): The number of iterations to perform for each number of components. Defaults to 10.

        Returns:
            np.ndarray: The reduced-dim vectors.
        """
        n_components = min(n_components, len(vectors) - 2)
        if n_neighbors is None:
            n_neighbors = int((len(vectors) - 1) ** 0.5)
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors, n_components=n_components, metric="cosine"
        )
        return umap_model.fit_transform(vectors)

    def cluster_vectors(
        self, vectors: np.ndarray, reduced_dim: int, threshold: float = None
    ):
        """
        Perform GMM clustering on the input vectors.

        Args:
            vectors (np.ndarray): The input vectors to cluster.
            reduced_dim (int): The number of dimensions to reduce the input vectors to.
            threshold (float, optional): The threshold for assigning a vector to a cluster. Defaults to 1/n_components.

        Returns:
            labels (List[np.ndarray]): A list of arrays, where each array contains the indices of the clusters that the corresponding vector belongs to.
            n_components (int): The optimal number of components for the Gaussian mixture model.
        """
        vectors = self.reduce_dimensions(vectors, reduced_dim)
        n_components = self.find_optimal_components(vectors)
        gmm = GaussianMixture(n_components=n_components, n_init=self.n_init)
        gmm.fit(vectors)
        # Vectors can be assigned to multiple clusters
        probs = gmm.predict_proba(vectors)
        if threshold is None:
            threshold = 1 / n_components

        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_components

    def cluster_nodes(self, nodes: List["Node"], recursion_level=0):
        """
        Find semantic clusters among the input nodes.

        Args:
            nodes (List[Node]): The nodes to cluster.
            recursion_level (int, optional): The current level of recursion. Defaults to 0.

        Returns:
            List[List[Node]]: A list of lists of nodes, where each inner list represents a single cluster.
        """
        cluster_token_count = sum([node.token_count for node in nodes])
        if cluster_token_count <= self.max_cluster_tokens:
            logging.debug(
                f"Cluster has {cluster_token_count} tokens across {len(nodes)} nodes. Not clustering anymore."
            )
            return [nodes]

        # TOO FEW NODES
        if len(nodes) <= self.max_cluster_size:
            # To avoid tiny clusters that can't be UMAPped properly
            logging.debug(f"Cluster has {len(nodes)} nodes. Not clustering anymore.")
            return [nodes]

        vectors = np.array([node.text_emb for node in nodes])
        labels, n_components = self.cluster_vectors(vectors, self.reduced_dim)

        final_clusters = []
        for cluster_id in range(n_components):
            membership_mask = [cluster_id in label for label in labels]
            member_nodes = [node for c, node in enumerate(nodes) if membership_mask[c]]
            sub_cluster_nodes = self.cluster_nodes(
                member_nodes, recursion_level=recursion_level + 1
            )
            final_clusters.extend(sub_cluster_nodes)

        return final_clusters
