import logging
import os
import faiss
import numpy as np
import uuid
from typing import List, Tuple
import warnings


class FaissVectorDatabase():
    def __init__(self, embedding_model, index_file: str = None):
        """
        Initialize the FaissVectorDatabase object with an embedding model and an optional index file.

        Args:
            embedding_model (EmbeddingModel): Model used to generate embeddings for text
            index_file (str, optional): The path to the FAISS index file. If provided, the index will be loaded from this file. Otherwise, a new index will be created.
        """
        self.embedding_model = embedding_model
        self.dims = embedding_model.dims
        self.index_file = index_file
        self.index = self.load_index()


    def load_index(self):
        if self.index_file is None:
            self.index_file = "index_{}.faiss".format(uuid.uuid4())
            logging.info(f"creating a new index at {self.index_file}...")
            index = faiss.index_factory(self.dims, "IDMap,HNSW,Flat")
        else:
            try:
                index = faiss.read_index(self.index_file)
            except Exception as e:
                raise ValueError("Invalid index file: {}".format(e))
        return index

    def save(self):
        faiss.write_index(self.index, self.index_file)

    def add_embeddings(self, ids: np.ndarray, embeddings: np.ndarray):
        """
        Add document embeddings to the index, indexed by their corresponding node hash_id strings.

        Args:
            ids (List[str]): A list of id strings for each embedding.
            embeddings (np.ndarray): A 2D numpy array of shape (n_docs, dims) containing the document embeddings.
        """
        self.index.add_with_ids(embeddings, ids)

    def search(self, query: str, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search the index for the k nearest neighbors of a given query embedding.

        Args:
            query (str): The query to lookup
            k (int, optional): The number of nearest neighbors to return. Defaults to 10.

        Returns:
            A tuple containing two lists: the first list contains the hash strings of the k nearest neighbors, and the second list contains the corresponding distances.
        """
        query_embedding = self.embedding_model.get_text_embedding(query)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        indices = list(map(str, indices[0].tolist()))
        return indices, distances[0].tolist()

    def persist_tree(self, tree):
        """
        Persist all nodes in the tree to the vector database.
        """
        ids = []
        txt_embeddings = []
        q_embeddings = []
        for node in tree.all_nodes:
            ids.append(node.hash_id)
            txt_embeddings.append(node.text_emb)
            q_embeddings.append(node.questions_emb)

        ids = np.array(ids, dtype=np.int64)
        txt_emb = np.array(txt_embeddings, dtype=np.float32)
        q_emb = np.array(q_embeddings, dtype=np.float32)
        logging.info(f"Persisting {len(txt_embeddings)} text embeddings...")
        self.add_embeddings(ids, txt_emb)
        logging.info(f"Persisting {len(q_embeddings)} question embeddings...")
        self.add_embeddings(ids, q_emb)
