import logging
from config import TreeBuilderConfig, GMMClusteringConfig, Neo4JDriverConfig
from models import JinaEmbeddingModel, AzureLlamaSummarizationModel
from clustering import GMMClustering
from tree_builder import TreeBuilder
from neo4j_driver import Neo4JDriver

logging.basicConfig(format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.DEBUG)

def parse_document(
    document_path: str,
    tokenizer_id: str,
    start_pg= 0,
    end_pg= None,
    vector_index_file=None,
    tree_builder_config = TreeBuilderConfig(),
    clustering_config = GMMClusteringConfig(),
    neo4j_config = Neo4JDriverConfig(),
    save_tree = False
    ):
    """
    Main entry point for the CLI.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    clusterer = GMMClustering(**clustering_config.dict())
    embedding_model = JinaEmbeddingModel()
    summarization_model = AzureLlamaSummarizationModel()
    tree_builder = TreeBuilder(tokenizer, clusterer, embedding_model, summarization_model, **tree_builder_config.dict())
    tree = tree_builder.build_tree_from_document(document_path, start_end=(start_pg, end_pg))

    if save_to_json:
        output_dir = f"outputs/tree_{document_path.split('/')[-1]}_{start_pg}_{end_pg}"
        os.makedirs(output_dir, exist_ok=True)
        tree.save(output_dir)

    graph_client = Neo4JDriver(**neo4j_config.dict())
    graph_client.upload_tree(document_path, tree)
    tree_builder.persist_embeddings(tree, vector_index_file)

    return tree
