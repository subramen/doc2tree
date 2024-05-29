import os
import json
import pickle
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict,Union
from transformers import AutoTokenizer

from clustering import GMMClustering
from text import Document, SentencePreservingChunker, get_document_chunks

class Node:
    """
    A class representing a node in a tree.

    Attributes:
    - text (str): The text content of the node.
    - questions(str): The questions extracted from the node's text.
    - token_count (int): The number of tokens in the node's text.
    - breadcrumb (str): The breadcrumb of the node.
    - page_label (str): The page label of the node.
    - bbox (List[float]): The bounding box of the node on the page.
    - children (List[Node]): A list of the node's children.
    - hash_id (int): The hash id of the node text.
    """
    def __init__(self, text: str, token_count: int, questions: str = "", breadcrumb: str = "", page_label: str = "", bbox: List[float] = [], children = None, text_emb=None, questions_emb=None):
        self.text = text
        self.token_count = token_count
        self.questions = questions
        self.breadcrumb = breadcrumb
        self.page_label = page_label
        self.bbox = bbox
        self.children = children
        self.hash_id = hash(self.text)
        self.text_emb = text_emb
        self.questions_emb = questions_emb


class Tree:
    def __init__(self, root_nodes: List[Node], layer_to_nodes: Dict[int, List]=None, metadata: Dict[str, str] = {}):
        self.root_nodes = root_nodes
        self.layer_nodes = layer_to_nodes or self.layer_to_node()
        self.metadata = metadata

    def __str__(self):
        levels = max(self.layer_nodes.keys())
        n_nodes = len(sum(list(self.layer_nodes.values()), []))
        counts = f"{len(self.layer_nodes[0])} leaves || {levels} levels || {n_nodes} nodes"
        meta = f"Metadata: {self.metadata}"
        return f"{counts}\n{meta}"

    @property
    def all_nodes(self):
        return (node for node_list in self.layer_nodes.values() for node in node_list)

    def layer_to_node(self):
        d = {}
        def helper(node, level):
            try:
                d[level].append(node)
            except KeyError:
                d[level] = [node]
            if node.children:
                for child in node.children:
                    helper(child, level + 1)

        for node in self.root_nodes:
            helper(node, 0)
        max_level = max(d.keys())
        assert max_level > 0
        return {max_level - k : d[k] for k in range(max_level + 1)}

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
        # with open(f"{directory}/tree.json", 'w') as f:
        #     json.dump(self.to_json(), f, indent=2)


class TreeBuilder:
    def __init__(self,
        tokenizer_id,
        clusterer,
        language_model,
        embedding_model,
        leaf_text_tokens,
        parent_text_tokens,
        max_layers,
    ):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=os.environ['HF_TOKEN'])

        self.clusterer = clusterer
        self.language_model = language_model
        self.embedding_model = embedding_model
        self.leaf_text_tokens = leaf_text_tokens
        self.parent_text_tokens = parent_text_tokens
        self.max_layers = max_layers
        self.chunker = SentencePreservingChunker(self.tokenizer, self.leaf_text_tokens)

    def create_leaf_node_batched(self, chunks):
        text_batch = [chunk['text'] for chunk in chunks]
        text_emb_batch = self.embedding_model.get_text_embedding(text_batch)
        questions_batch = self.language_model.extract_questions(text_batch)
        questions_emb_batch = self.embedding_model.get_text_embedding(questions_batch)
        return [
            Node(
                text=text_batch[i],
                token_count=len(self.tokenizer.encode(text_batch[i])),
                questions=questions_batch[i],
                breadcrumb=chunks[i]["breadcrumb"],
                page_label=chunks[i]["page_label"],
                bbox=chunks[i]["bbox"],
                text_emb=text_emb_batch[i],
                questions_emb=questions_emb_batch[i]
            ) for i in range(len(chunks))
        ]


    def create_parent_node_batched(self, clusters: List[List[Node]]):
        text_batch = ["\n".join([node.text for node in cluster]) for cluster in clusters]
        facts_batch = self.language_model.extract_facts(text_batch)
        questions_batch = self.language_model.extract_questions(text_batch)
        facts_emb_batch = self.embedding_model.get_text_embedding(facts_batch)
        questions_emb_batch = self.embedding_model.get_text_embedding(questions_batch)
        return [
            Node(
                text=facts_batch[i],
                token_count=len(self.tokenizer.encode(facts_batch[i])),
                questions=questions_batch[i],
                children=clusters[i],
                text_emb=facts_emb_batch[i],
                questions_emb=questions_emb_batch[i]
            ) for i in range(len(clusters))
        ]



    def build_tree_from_document(
        self,
        document_path: str,
        start_end: tuple = (0, None,),
    ) -> Tree:
        document = Document(document_path)
        document.metadata.update({'start_page': start_end[0], 'end_page': start_end[1] or -1})
        logging.info(f"Building tree for {document_path}: pages {start_end[0]} to {start_end[1]}")
        chunks_with_metadata = list(get_document_chunks(document, self.chunker, start_end))

        logging.info(f"Built layer 0 with {len(chunks_with_metadata)} leaf nodes")
        current_layer = self.create_leaf_node_batched(chunks_with_metadata)
        layer_to_nodes = {0: current_layer}

        for i in tqdm(range(1, self.max_layers + 1), desc=f"Building layers"):
            logging.info(f"Clustering {len(current_layer)} nodes in layer {i-1}...")
            clusters = self.clusterer.cluster_nodes(current_layer)
            logging.info(f"Building layer {i} from {len(clusters)} clusters")
            parents = self.create_parent_node_batched(clusters)
            current_layer = parents
            layer_to_nodes[i] = current_layer
            # stopping criteria
            if len(current_layer) == 1:
                break

        tree = Tree(current_layer, layer_to_nodes=layer_to_nodes, metadata=document.metadata)
        return tree
