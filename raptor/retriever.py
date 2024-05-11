

class TreeRetriever:
    def __init__(self,
        embedding_model,
        reranker_model,
        vector_db_client,
        graph_db_client,
    ):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.vector_db_client = vector_db_client
        self.graph_db_client = graph_db_client

    def search(self, query, k=10):
        query_embedding = self.embedding_model.get_text_embedding(query, to_numpy=True)
        indices, distance = self.vector_db_client.search(query_embedding, k)
        return indices, distance

    def retrieve_from_graph(self, query):
        indices, distance = self.search(query)
        nodes = self.graph_db_client.get_nodes_by_hash_ids(indices)
        result = [{"text": n.text, "questions": n.questions, "breadcrumb": n.breadcrumb} for n in nodes]
        return results

    def rerank(self, query, k=10):
        retrieval_results = self.retrieve_from_graph(query)
        candidates = [f"TEXT: {r['text']} QUESTIONS: {r['questions']}" for r in retrieval_results]
        ranked_indices = self.reranker_model.rerank(candidates)
        return [retrieval_results[i] for i in ranked_indices]
