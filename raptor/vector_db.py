import os
import faiss
import numpy as np
import uuid
from typing import List, Tuple
import warnings
import unittest

class BaseVectorDatabase:
    def __init__(self):
        pass

    def add_embeddings(self, ids: List[str], embeddings: np.ndarray):
        pass

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        pass

class FaissVectorDatabase(BaseVectorDatabase):
    def __init__(self, dims: int, index_file: str = None):
        """
        Initialize the FaissVectorDatabase object with a specified number of dimensions and an optional index file.

        Args:
            dims (int): The number of dimensions in the embeddings.
            index_file (str, optional): The path to the FAISS index file. If provided, the index will be loaded from this file. Otherwise, a new index will be created.
        """
        self.dims = dims
        self.index_file = index_file or "index_{}.faiss".format(uuid.uuid4())
        self.index = self.load_index()


    def load_index(self):
        if not os.path.exists(self.index_file):
            warnings.warn(f"Index file doesn't exist, creating a new index at {self.index_file}...")
            # index = faiss.IndexFlatL2(self.dims)
            index = faiss.index_factory(self.dims, "IDMap,HNSW,Flat")
        else:
            try:
                index = faiss.read_index(self.index_file)
            except Exception as e:
                raise ValueError("Invalid index file: {}".format(e))
        return index

    def save(self):
        faiss.write_index(self.index, self.index_file)


    def add_embeddings(self, ids: List[str], embeddings: np.ndarray):
        """
        Add document embeddings to the index, indexed by their corresponding node elementID strings.

        Args:
            ids (List[str]): A list of id strings for each embedding.
            embeddings (np.ndarray): A 2D numpy array of shape (n_docs, dims) containing the document embeddings.
        """
        self.index.add_with_ids(embeddings, ids)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search the index for the k nearest neighbors of a given query embedding.

        Args:
            query_embedding (np.ndarray): A 1D numpy array of shape (dims,) containing the query embedding.
            k (int, optional): The number of nearest neighbors to return. Defaults to 10.

        Returns:
            A tuple containing two lists: the first list contains the hash strings of the k nearest neighbors, and the second list contains the corresponding distances.
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return indices[0].tolist(), distances[0].tolist()

    def persist_tree(self, tree):
        ids = []
        embeddings = []
        for node in tree.all_nodes:
            ids.append(node.hash_id)
            embeddings.append(node.embedding["vector"])
        self.add_embeddings(ids, np.array(embeddings))
