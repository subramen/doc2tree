import os
from typing import Type
from pydantic import BaseModel


class TreeBuilderConfig(BaseModel):
    """
    Configuration for the tree builder.
    """

    tokenizer_id: str = 'mistralai/Mistral-7B-v0.1'
    # The maximum number of tokens to consider for a leaf node.
    leaf_text_tokens: int = 256
    # The maximum number of tokens to consider for a parent node.
    parent_text_tokens: int = 1024  # 4 * leaf_text_tokens
    # The maximum number of layers in the tree
    max_layers: int = 3


class GMMClusteringConfig(BaseModel):
    """
    Configuration for the GMM clustering.
    """
    n_init: int = 3
    max_cluster_tokens: int = 2048  # 8 * leaf_text_tokens


class Neo4JDriverConfig(BaseModel):
    """
    Configuration for the Neo4J driver.
    """
    uri: str = os.environ['NEO4J_DB']
    user: str = os.environ['NEO4J_USER']
    password: str = os.environ['NEO4J_PASSWD']
