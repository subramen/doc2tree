import os
import fire
import logging
from omegaconf import OmegaConf
from clustering import GMMClustering
from tree import TreeBuilder
from graph_db import Neo4JDriver
from vector_db import FaissVectorDatabase
from models import EmbeddingModel, Llama3, RerankerModel

logging.basicConfig(format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.INFO)
config = OmegaConf.load("config.yaml")
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


def ask(
    query: str,
    vector_index_file: str,
    use_reranker: bool = False,
    ):

    embedding_model = EmbeddingModel(**config.embedding_model)
    language_model = Llama3(**config.language_model)
    graph_client = Neo4JDriver(**config.neo4j)
    faiss_client = FaissVectorDatabase(embedding_model, index_file=vector_index_file)
    
    retrieval_k = config.retrieval.step1_k if use_reranker else config.retrieval.step2_k
    neighbor_ids, _ = faiss_client.search(query, k=retrieval_k)
    neighbor_ids = list(dict.fromkeys(neighbor_ids)) # deduplicate
    neighbor_nodes = graph_client.get_nodes_by_hash_ids(neighbor_ids)
    # neighbor_nodes = graph_client.nodes_in_paths(neighbor_ids)

    if use_reranker:
        reranker_model = RerankerModel(**config.reranker_model)
        ranked_idx = reranker_model.rerank(query, [n.text for n in neighbor_nodes])
        context = [neighbor_nodes[idx].text for idx in ranked_idx[:config.retrieval.step2_k]]
    else: 
        context = [n.text for n in neighbor_nodes]

    response = {
        'answer': language_model.write_response(query, context),
        'step1_neighbors': neighbor_ids[:config.retrieval.step2_k],
        'step2_neighbors': [neighbor_ids[i] for i in ranked_idx[:config.retrieval.step2_k]] if use_reranker else None,
        'context': context,
    }  
    return response


if __name__ == "__main__":
    fire.Fire()
