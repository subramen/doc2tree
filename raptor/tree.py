import json
import pickle
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer

from text import Document, SentencePreservingChunker, chunk_document
from clustering import GMMClustering
from vector_db import FaissVectorDatabase

class Node:
    """
    A class representing a node in a tree.

    Attributes:
    - text (str): The text content of the node.
    - embedding (Dict[str, np.ndarray]): A dictionary containing the embeddings of the node.
    - token_count (int): The number of tokens in the node's text.
    - breadcrumb (str): The breadcrumb of the node.
    - page_label (str): The page label of the node.
    - bbox (List[float]): The bounding box of the node on the page.
    - children (List[Node]): A list of the node's children.
    - hash_id (int): The hash id of the node text.
    """
    def __init__(self, text: str, embedding: Dict[str, np.ndarray], token_count: int, breadcrumb: str = None, page_label: str = None, bbox: List[float] = None, children = None):
        self.text = text
        self.embedding = embedding
        self.token_count = token_count
        self.breadcrumb = breadcrumb
        self.page_label = page_label
        self.bbox = bbox
        self.children = children
        self.hash_id = hash(self.text)


class Tree:
    def __init__(self, root_nodes: List[Node]):
        self.root_nodes = root_nodes
        self.layer_nodes = self.layer_to_node()

    def __str__(self):
        levels = max(self.layer_nodes.keys())
        n_nodes = len(sum(list(self.layer_nodes.values()), []))
        return f"""Tree with:
        - {len(layer_nodes[0])} leaves
        - {len(levels)} levels
        - {n_nodes} nodes
        """

    def layer_to_node(self):
        d = {}
        def helper(node, level):
            d.get(level, []).append(node)
            if node.children:
                for child in node.children:
                    helper(child, level + 1)
        for node in self.root_nodes:
            helper(node, 0)
        max_level = max(layer_to_node.keys())
        return {max_level - k : d[k] for k in range(max_level)}

    def to_json(self, skip_keys=[], vector_index=""):
        def helper(node, skip_keys=[]):
            subtree = node.__dict__
            subtree = {k:subtree[k] for k in subtree if k not in skip_keys}
            if node.children:
                subtree['children'] = [helper(child, skip_keys) for child in node.children]
            return subtree
        return {"tree": [helper(node, skip_keys) for node in self.root_nodes], "vector_index": vector_index }

    def from_json(self, json_dict):
        def helper(node_dict):
            node = Node(**{k:v for k, v in node_dict.items() if k != 'children'})
            if 'children' in node_dict:
                node.children = [helper(child) for child in node_dict['children']]
            return node

        root_nodes = [helper(node) for node in json_dict['tree']]
        return Tree(root_nodes)

    def save(self, directory='.'):
        with open(f"{directory}/tree.pkl", 'wb') as f:
            pickle.dump(self, f)
        with open(f"{directory}/tree.json", 'w') as f:
            json.dump(self.to_json(), f, indent=2)


class TreeBuilder:
    def __init__(self,
        tokenizer_id,
        clusterer,
        embedding_model,
        summarization_model,
        leaf_text_tokens,
        parent_text_tokens,
        max_layers,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.clusterer = clusterer
        self.leaf_text_tokens = leaf_text_tokens
        self.parent_text_tokens = parent_text_tokens
        self.max_layers = max_layers
        self.chunker = SentencePreservingChunker(self.tokenizer, self.leaf_text_tokens)


    def create_leaf_node(self, document_chunk: Dict) -> Node:
        text = document_chunk["text"].strip()
        embedding = self.embedding_model.create_embedding(text)
        return Node(
            text=text,
            embedding={self.embedding_model.name: embedding},
            token_count=len(self.tokenizer.encode(text)),
            breadcrumb=document_chunk["breadcrumb"],
            page_label=document_chunk["page_label"],
            bbox=document_chunk["bbox"]
        )


    def create_parent_node(self, cluster: List[Node]) -> Node:
        """
        Create a parent node for a list of nodes.

        Args:
            nodes (List[Node]): A list of nodes to create a parent node for.

        Returns:
            A Node object representing the parent node.
        """
        all_text = "\n".join([node.text for node in cluster])
        summary = self.summarization_model.summarize(all_text, max_tokens=self.parent_text_tokens)
        assert summary is not None
        embedding = self.embedding_model.create_embedding(summary)
        return Node(
            text=summary,
            embedding={self.embedding_model.name: embedding},
            token_count=len(self.tokenizer.encode(summary)),
            children=cluster
        )


    def build_tree_from_document(
        self,
        document_path: str,
        start_end: tuple = (0, None,),
    ) -> Tree:
        document = Document(document_path)
        logging.info(f"Building tree for {document_path}: pages {start_end[0]} to {start_end[1]}")
        chunks_with_metadata = list(chunk_document(document, self.chunker, start_end))

        current_layer = [self.create_leaf_node(chunk) for chunk in tqdm(chunks_with_metadata, desc=f"Building leaf nodes")]
        layer_to_nodes = {0: current_layer}

        for i in tqdm(range(1, self.max_layers + 1), desc="Building layers"):
            clusters = self.clusterer.cluster_nodes(current_layer, self.embedding_model.name)
            parents = [self.create_parent_node(cluster) for cluster in tqdm(clusters, desc=f"Building parents for level {i}")]
            current_layer = parents
            layer_to_nodes[i] = current_layer
            # stopping criteria
            if len(current_layer) == 1:
                break

        tree = Tree(current_layer)
        return tree


    def cache_embeddings_to_vectordb(self, tree: Tree, vector_index_file: str = None):
        """
        Persist the embeddings of the nodes in the tree to a FAISS index.

        Args:
            tree (Tree): The tree to persist.
            vector_index_file (str): The path to the FAISS index file.
        """
        dims = self.embedding_model.dims
        index = FaissVectorDatabase(dims, vector_index_file)

        text = []
        embeddings = []
        for node in tqdm(tree.all_nodes(), desc="Persisting embeddings"):
            text.append(node.text)
            embeddings.append(node.embedding[self.embedding_model.name])
        embeddings = np.array(embeddings)
        index.add_embeddings(text, embeddings)

        index.save()
