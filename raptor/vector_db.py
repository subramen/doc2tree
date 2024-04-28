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
        self.index = self.load()


    def load(self):
        if not os.path.exists(index_file):
            warnings.warn(f"Creating a new index at {self.index_file}...")
            self.index = faiss.IndexFlatL2(dims)
        else:
            try:
                self.index = faiss.read_index(index_file)
            except Exception as e:
                raise ValueError("Invalid index file: {}".format(e))

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


class TestFaissVectorDatabase(unittest.TestCase):
    def setUp(self):
        self.dims = 3
        self.index_file = "test_index.faiss"
        self.ids = ["id1", "id2", "id3"]
        self.embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_init(self):
        # Test creating a new index
        vdb = FaissVectorDatabase(self.dims)
        self.assertIsInstance(vdb.index, faiss.IndexFlatL2)

        # Test loading an existing index
        vdb = FaissVectorDatabase(self.dims, self.index_file)
        self.assertIsInstance(vdb.index, faiss.IndexFlatL2)

    def test_add_embeddings(self):
        vdb = FaissVectorDatabase(self.dims)
        vdb.add_embeddings(self.ids, self.embeddings)
        self.assertEqual(vdb.index.ntotal, len(self.ids))

    def test_search(self):
        vdb = FaissVectorDatabase(self.dims)
        vdb.add_embeddings(self.ids, self.embeddings)
        query_embedding = np.array([1, 2, 3])
        k = 2
        indices, distances = vdb.search(query_embedding, k)
        self.assertEqual(len(indices), k)
        self.assertEqual(len(distances), k)
        self.assertEqual(indices, ["id1", "id2"])
        self.assertEqual(distances, [np.linalg.norm(query_embedding - self.embeddings[0]), np.linalg.norm(query_embedding - self.embeddings[1])])
