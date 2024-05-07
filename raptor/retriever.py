

class TreeRetriever:
    def __init__(self,
        embedding_model,
        vector_db_client,
        graph_db_client
    ):
        self.embedding_model = embedding_model
        self.vector_db_client = vector_db_client
        self.graph_db_client = graph_db_client

    def search(self, query, k=10):
        query_embedding = self.embedding_model.create_embedding(query, to_numpy=True)['vector']
        indices, distance = self.vector_db_client.search(query_embedding, k)
        return indices, distance


    def retrieve_from_graph(self, query):
        indices, distance = self.search(query)
        nodes = self.graph_db_client.get_nodes(indices)
        return nodes
