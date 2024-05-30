import os
import fire
import logging
from omegaconf import OmegaConf
from models import EmbeddingModel, Llama3, RerankerModel
from clustering import GMMClustering
from tree import TreeBuilder
from graph_db import Neo4JDriver
from vector_db import FaissVectorDatabase

logging.basicConfig(format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.INFO)
config = OmegaConf.load("raptor/config.yaml")
os.environ["HF_TOKEN"] = config.HF_TOKEN

def upload_document(
    document_path: str,
    start_page= 0,
    end_page= None,
    vector_index_file=None,
    save_tree = False
    ):
    """
    Main entry point for the CLI.
    """
    logging.info(f"Starting document upload...")
    clusterer = GMMClustering(**config.clustering)
    embedding_model = EmbeddingModel(**config.embedding_model)
    language_model = Llama3(**config.language_model)
    graph_client = Neo4JDriver(**config.neo4j)

    tree_builder = TreeBuilder(
        clusterer=clusterer,
        language_model=language_model,
        embedding_model=embedding_model,
        **config.tree_builder)

    tree = tree_builder.build_tree_from_document(document_path, start_end=(start_page, end_page))

    if save_tree:
        output_dir = f"outputs/tree_{document_path.split('/')[-1]}_{start_page}_{end_page}"
        os.makedirs(output_dir, exist_ok=True)
        tree.save(output_dir)

    graph_client.upload_tree(tree)
    faiss_client = FaissVectorDatabase(embedding_model, vector_index_file)
    faiss_client.persist_tree(tree)
    faiss_client.save()

    return tree


def query_graph(
    query: str,
    vector_index_file: str,
    ):

    embedding_model = EmbeddingModel(**config.embedding_model)
    reranker_model = RerankerModel(**config.reranker_model)
    language_model = LanguageModel(**config.language_model)

    faiss_client = FaissVectorDatabase(embedding_model, vector_index_file=vector_index_file)
    neighbor_idx, _ = faiss_client.search(query, k=config.retriever_k)
    neighbor_idx = list(set(neighbor_idx))

    graph_client = Neo4JDriver(**config.neo4j)
    neighbor_nodes = graph_client.get_nodes_by_hash_ids(neighbor_idx)

    ranked_idx = reranker_model.rerank(query, [n.text for n in neighbor_nodes])
    retrieved_context = '\n'.join([neighbor_nodes[idx].text for idx in ranked_idx[:config.reranker_k]])
    response = language_model.write_response(query, retrieved_context)

    return response


if __name__ == "__main__":
    fire.Fire()
    # python raptor/main.py upload_document --vector_index_file 'loy.faiss' --document_path 'documents/28LettersOnYoga-I.pdf' --start_page 380 --end_page 400
