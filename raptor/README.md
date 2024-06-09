# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

<!-- ![banner]()

![badge]()
![badge]()
[![license](https://img.shields.io/github/license/:user/:repo.svg)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme) -->

This repository adapts the [RAPTOR](https://arxiv.org/abs/2401.18059) technique of structuring documents as a tree optimized for retrieval by an LLM.

This is NOT a pure implementation of the paper! The code here uses:
* Neo4J for the graph database
* FAISS for the vector database
* Meta Llama 3 served on VLLM

The code here augments each node in the RAPTOR tree with additional LLM-generated interpretations.

The original paper encounters an increased bias towards retrieving leaf nodes. To rectify this, we make use of 2-step retrieval via a reranker model.


## Prerequisites
* A Neo4J graph URI with write access
* Access to Meta Llama 3 models on the HuggingFace hub (if using Llama models)



## Usage
To render a document as a tree, run the following:

`python raptor/main.py upload_document --document_path '/path/to/document/raptor_arxiv.pdf'`

Adjust configuration parameters such as size of tree, graph database endpoints, Llama inference endpoints and batch size for inferece in [](config.yaml)
This step will take some time depending on the length of the document and the size of the tree.

This step will generate a tree in Neo4J and a FAISS index file. Note that FAISS indexes are platform-specific so if you build one on MacOS, the same index will likely not work on a Linux system.

To run inference, run the following:
`python raptor/main.py ask --query 'What are the advantages of using RAPTOR over other retrieval methods?' --vector_index_file index_00112233.faiss`




## RAPTOR in (high-level) detail

### Document Preprocessing
* Given a document, we create a `Document` instance that wraps the file along with other metadata.
* The document is then divided into chunks of text. Each chunk contains the raw text along with metadata identifying where it is in the document.

### Create Leaf Nodes
* The chunks are then converted to `Node` instances. Along with the raw text and metadata, the `Node` contains
    * a list of questions that can be answered by the raw text
    * embeddings of the raw text
    * embeddings of the question
* These leaf nodes form `layer_0` or the base of the tree.

### Identify Semantic Clusters
* There might be subsets of the leaf nodes that are about the same topics.
* The text embeddings are first reduced to a smaller number of dimensions (using UMAP) and clustered (using the Gaussian Markov Mixture clustering algorithm) to yield these subsets.
* Each cluster of semantically-similar nodes will be processed into a single higher-layer node.

### Cluster to Parent Node
* Similar to leaf nodes, each parent node contains:
    * A summary of all the text contained in the cluster (extracted as "facts" by the LLM)
    * A list of questions that can be answered by the summary
    * Embeddings of the summary
    * Embeddings of the questions
* Every node in `layer_i` (where `i>0`) will have children in `layer_i-1`. This parent node represents a high-level representation of the raw text in the children.

### Q&A via Vector Lookups
* All the embeddings are mapped to the nodes they belong in a vector database.
* At inference time, given a query, the most relevant embeddings are looked up. The contents of these retrieved nodes form the context based on which the LLM constructs a response.





<!-- ## Table of Contents

- [Security](#security)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contributing](#contributing)
- [License](#license)

## Security

### Any optional sections

## Background

### Any optional sections

## Install

This module depends upon a knowledge of [Markdown]().

```
```

### Any optional sections

## Usage

```
```

Note: The `license` badge image link at the top of this file should be updated with the correct `:user` and `:repo`.

### Any optional sections

## API

### Any optional sections

## More optional sections

## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

Small note: If editing the Readme, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

### Any optional sections

## License
 -->
