import fire
import logging
from omegaconf import OmegaConf
from models import EmbeddingModel, LanguageModel
from clustering import GMMClustering
from tree import TreeBuilder
from graph_db import Neo4JDriver
from vector_db import FaissVectorDatabase

logging.basicConfig(format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.DEBUG)

def upload_document(
    document_path: str,
    tokenizer_id: str,
    start_page= 0,
    end_page= None,
    vector_index_file=None,
    tree_builder_config = TreeBuilderConfig(),
    clustering_config = GMMClusteringConfig(),
    neo4j_config = Neo4JDriverConfig(),
    save_tree = False
    ):
    """
    Main entry point for the CLI.
    """
    config = OmegaConf.load("config.yaml")

    clusterer = GMMClustering(**config.clustering)
    embedding_model = EmbeddingModel(**config.embedding_model)
    language_model = LanguageModel(**config.language_model)

    tree_builder = TreeBuilder(
        clusterer=clusterer,
        language_model=language_model,
        **config.tree_builder)

    tree = tree_builder.build_tree_from_document(document_path, start_end=(start_page, end_page))

    if save_tree:
        output_dir = f"outputs/tree_{document_path.split('/')[-1]}_{start_page}_{end_page}"
        os.makedirs(output_dir, exist_ok=True)
        tree.save(output_dir)

    graph_client = Neo4JDriver(**config.neo4j)
    graph_client.upload_tree(tree)

    faiss_client = FaissVectorDatabase(embedding_model, vector_index_file)
    faiss_client.persist_tree(tree)
    faiss_client.save()

    return tree


if __name__ == "__main__":
    fire.Fire()
    # python raptor/main.py upload_document --tokenizer_id 'NousResearch/Nous-Hermes-Llama2-70b' --vector_index_file 'loy.faiss' --document_path 'documents/28LettersOnYoga-I.pdf' --start_page 380 --end_page 400
