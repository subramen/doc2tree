import os
import fire
import logging
from omegaconf import OmegaConf
from clustering import GMMClustering
from tree import TreeBuilder
from graph_db import Neo4JDriver
from vector_db import FaissVectorDatabase
from models import EmbeddingModel, Llama3, RerankerModel

logging.basicConfig(
    format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.INFO
)
config = OmegaConf.load("config.yaml")
os.environ["HF_TOKEN"] = config.HF_TOKEN


def upload_document(
    document_path: str,
    start_page=0,
    end_page=None,
    vector_index_file=None,
    save_tree=False,
):
    """
    Uploads a document to the graph database and saves it to a vector index file.

    Args:
        document_path (str): The path to the document to be uploaded.
        start_page (int, optional): The starting page of the document to be uploaded. Defaults to 0.
        end_page (int, optional): The ending page of the document to be uploaded. Defaults to None.
        vector_index_file (str, optional): The path to the vector index file. Defaults to None.
        save_tree (bool, optional): Whether to save the tree to a file. Defaults to False.

    Returns:
        Tree: The tree that was uploaded to the graph database.
    """

    logging.info(f"Starting document upload...")

    # Initialize the clustering, embedding, language, and graph clients using configuration parameters
    clusterer = GMMClustering(**config.clustering)
    embedding_model = EmbeddingModel(**config.embedding_model)
    language_model = Llama3(**config.language_model)
    graph_client = Neo4JDriver(**config.neo4j)

    # Build a tree from the document using the TreeBuilder class and the initialized clients
    tree_builder = TreeBuilder(
        clusterer=clusterer,
        language_model=language_model,
        embedding_model=embedding_model,
        **config.tree_builder,
    )
    tree = tree_builder.build_tree_from_document(
        document_path, start_end=(start_page, end_page)
    )

    # If the save_tree flag is set, save the tree to a file in the outputs directory
    if save_tree:
        output_dir = (
            f"outputs/tree_{document_path.split('/')[-1]}_{start_page}_{end_page}"
        )
        os.makedirs(output_dir, exist_ok=True)
        tree.save(output_dir)

    # Upload the tree to the graph database using the graph client
    graph_client.upload_tree(tree)

    # Initialize the FAISS vector database using the embedding model and vector index file
    faiss_client = FaissVectorDatabase(embedding_model, vector_index_file)

    # Persist the tree to the FAISS vector database and save the database to disk
    faiss_client.persist_tree(tree)
    faiss_client.save()

    return tree


def ask(
    query: str,
    vector_index_file: str,
    use_reranker: bool = False,
):
    """
    Asks a question and returns an LLM-generated response via RAG.

    Args:
        query (str): The question to be asked.
        vector_index_file (str): The file containing the vector index.
        use_reranker (bool, optional): Whether to use the reranker model. Defaults to False.

    Returns:
        dict: A dictionary containing the answer, step1_neighbors, step2_neighbors, and context.
    """

    # Initialize the embedding model, language model, graph client, and FAISS client
    embedding_model = EmbeddingModel(**config.embedding_model)
    language_model = Llama3(**config.language_model)
    graph_client = Neo4JDriver(**config.neo4j)
    faiss_client = FaissVectorDatabase(embedding_model, index_file=vector_index_file)

    # Determine the number of neighbors to retrieve based on whether the reranker model is being used
    retrieval_k = config.retrieval.step1_k if use_reranker else config.retrieval.step2_k

    # Search the FAISS index for the k nearest neighbors to the query
    neighbor_ids, _ = faiss_client.search(query, k=retrieval_k)

    # Remove any duplicate neighbor IDs and get the corresponding nodes from the graph database
    neighbor_ids = list(dict.fromkeys(neighbor_ids))
    neighbor_nodes = graph_client.get_nodes_by_hash_ids(neighbor_ids)

    # If the reranker model is being used, rerank the neighbors and get the context
    if use_reranker:
        reranker_model = RerankerModel(**config.reranker_model)
        ranked_idx = reranker_model.rerank(query, [n.text for n in neighbor_nodes])
        context = [
            neighbor_nodes[idx].text for idx in ranked_idx[: config.retrieval.step2_k]
        ]
    # Otherwise, just get the context from the neighbor nodes
    else:
        context = [n.text for n in neighbor_nodes]

    # Generate a response using the language model and the context
    response = {
        "answer": language_model.write_response(query, context),
        "step1_neighbors": neighbor_ids[: config.retrieval.step2_k],
        "step2_neighbors": (
            [neighbor_ids[i] for i in ranked_idx[: config.retrieval.step2_k]]
            if use_reranker
            else None
        ),
        "context": context,
    }

    return response


"""
* data_sample_p = {question, oracle_docs, distractor_docs, oracle_cot_answer}
* data_sample_(1-p) = {question, oracle_docs=None, distractor_docs, oracle_cot_answer}
* cot_answer_template: ##Reason: {reason} ##Answer: {answer}
* SFT on data_samples => (oracle_cot_answer | docs, question)

Given a RAPTOR graph:
    - traverse top-down
    - collect dissimilar questions i.e. if the nearest question has a similarity score < `threshold=0.85`
    - for each question:
        - oracle_docs = (self, parent, min(3, all_children))
        - synthesize COT_answer for each question
        - distractor_docs = (find most dissimilar nodes)
        - return sample

Require:
    - prompt to generate COT answer
"""



def raft(
    neodoc_id: str,
    vector_index_file: str,
    threshold: float = 0.85
):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity


    embedding_model = EmbeddingModel(**config.embedding_model)
    language_model = Llama3(**config.language_model)
    graph_client = Neo4JDriver(**config.neo4j)
    faiss_client = FaissVectorDatabase(embedding_model, index_file=vector_index_file)

    @dataclass
    class Question:
        node_hash: int
        question: str

    # Get all the nodes in the tree
    graph_nodes = graph_client.download_tree_nodes(neodoc_id)

    # Get all the questions in the tree
    questions = []
    for c, n in enumerate(graph_nodes):
        ques = n.questions.split('\n')
        questions.extend([Question(c, q) for q in ques])

    # Get embeddings of all questions
    embs = embedding_model.get_text_embedding([q.question for q in questions])

    # Filter out questions that are pretty similar
    similarity_matrix = cosine_similarity(embs, embs)
    distinct_ix = [i for i in range(1, len(similarity_matrix)) if all(similarity_matrix[i, :i] < threshold)]
    filtered_questions = [questions[i] for i in distinct_ix]
    
    def create_raft_sample(Q):
        question = Q.question
        oracle_nodes = graph_client.get_node_family([Q.node_hash])
        oracle_docs = [oracle_nodes['self_node'].text, oracle_nodes['parent'].text, [c.text for c in oracle_nodes['children'][:3]]]
        distractor_nodes = graph_client.get_distractor_nodes([Q.node_hash])
        distractor_docs = [c.text for c in distractor_nodes]
        cot_answer = language_model.raft_qa(question, oracle_docs[:3] + distractor_docs[:3])
        return dict(question=question, oracle_docs=oracle_docs, distractor_docs=distractor_docs, cot_answer=cot_answer)
    


if __name__ == "__main__":
    fire.Fire()
