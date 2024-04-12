
import logging

logging.basicConfig(format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.DEBUG)

class BaseChunker(ABC):
    @abstractmethod
    def chunk_spans(self, text: str) -> List[Tuple(int, int)]:
        pass

class SentencePreservingChunker:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int):
        """
        Initialize the Chunker object with a tokenizer.

        Args:
            tokenizer (AutoTokenizer): A tokenizer object used to tokenize text.
            max_seq_len (int): The maximum number of tokens allowed in a sequence.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def sentence_spans(self, text, sentence_max_len=-1):
        """
        Get the sentence boundaries in the given text.

        Args:
            text (str): The text to find sentence boundaries in.
            sentence_max_len (int, optional): The maximum number of characters allowed in a sentence. Defaults to -1.

        Returns:
            list: A list of tuples representing the start and end indices of each sentence in the text.
        """
        sentence_splits = [match.start() for match in re.finditer(r'(?<=[.!?])\s?', text)]
        sentence_idx = []
        c = 0
        for idx in sentence_splits:
            # somtimes a wayward sentence might be longer than the permissible max_len
            if sentence_max_len > 0 and idx - c >= sentence_max_len:
                sentence_idx.append((c, c + sentence_max_len - 1,))
                sentence_idx.append((c + sentence_max_len - 1, idx,))
            else:
                sentence_idx.append((c, idx,))
            c = idx + 1
        return sentence_idx

    def chunk_spans(text: str) -> List[Tuple(int, int)]:
        """
        Get the chunk boundaries in the given text.

        Args:
            text (str): The text to find chunk boundaries in.

        Returns:
            list: A list of tuples representing the start and end indices of each chunk in the text.
        """

        # Get the sentence boundaries in the text
        sentence_idx = sentence_spans(text, self.max_seq_len)
        # Initialize empty list to store chunk indice
        chunk_idx = []
        chunk = [0, 0]
        current_length = 0

        # Iterate over the sentence boundaries
        for start_ix, end_ix in sentence_idx:
            seq_len = len(self.tokenizer.encode(text[start_ix:end_ix]))

            if current_length + seq_len < self.max_seq_len:
                chunk[-1] = end_ix
                current_length += seq_len

            else:
                chunk_idx.append(chunk)
                chunk = [start_ix, end_ix]
                current_length = seq_len

        chunk_idx.append(chunk)
        return chunk_idx



class Node:
    def __init__(self, text: str, embedding: np.ndarray, span: Union[List, Tuple[int, int]], token_count: int, children = None):
        self.text = text
        self.embedding = embedding
        self.span = span
        self.children = children
        self.token_count = token_count

    @property
    def hash_id(self):
        return hash(self.text)

    def __dict__(self):
        attr = {"hash_id": self.hash_id}
        return self.__dict__.update(attr)

class Tree:
    def __init__(self, document_url, root_nodes: List[Node], layer_to_nodes = None):
        self.document_url = document_url
        self.root_nodes = root_nodes
        self.layer_to_nodes = layer_to_nodes


class BaseEmbeddingModel(ABC):
    def __init__(self, name: str):
        self.name = name
    @abstractmethod
    def create_embedding(self, text):
        pass

class JinaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_id="jinaai/jina-embeddings-v2-base-en") -> None:
        super().__init__(name="jina")
        self.model = SentenceTransformer(model_id, trust_remote_code=True) # trust_remote_code is needed to use the encode method

    def create_embedding(self, text):
        return self.model.encode(text)




class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass

class AzureLlamaSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name='llama-2-70b-chat') -> None:
        self.endpoint = "https://Llama-2-70b-chat-suraj-demo-serverless.eastus2.inference.ai.azure.com/v1/completions"
        self.key = os.environ['KEY_70B']


    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        message = f"Write a summary of the following, including as many key details as possible: {context}:"

        payload = {
            "prompt": message,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "stop": stop_sequence,
        }
        headers = {'Content-Type':'application/json', 'Authorization':(self.key)}
        response = requests.post(self.endpoint, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()['choices'][0]['text']
        else:
            print(f"Request failed with status code {response.status_code}")




import numpy as np
from sklearn.mixture import GaussianMixture

class GMMClustering:
    def __init__(self, n_init: int = 5, max_token_length: int = None):
        self.n_init = n_init
        self.max_token_length = max_token_length

    def find_optimal_components(self, vectors, min_components: int = 1, max_components: int = 10, n_iter: int = 10):
        """
        Find the optimal number of components for the Gaussian mixture model using the Bayesian information criterion (BIC).

        Args:
            min_components (int, optional): The minimum number of components to consider. Defaults to 1.
            max_components (int, optional): The maximum number of components to consider. Defaults to 10.
            n_iter (int, optional): The number of iterations to perform for each number of components. Defaults to 10.

        Returns:
            The optimal number of components.
        """
        bic = []
        for n in range(min_components, max_components + 1):
            gmm = GaussianMixture(n_components=n, n_init=self.n_init)
            gmm.fit(vectors)
            bic.append(gmm.bic(vectors))
            logging.debug(f"BIC for {n} components: {bic[-1]}")
        return np.argmin(bic) + min_components

    def cluster_vectors(self, vectors, threshold: float = None):
        """
        Perform recursive GMM clustering on the input vectors.

        Args:
            max_membership (int): The maximum membership of each cluster.
            depth (int, optional): The current depth of recursion. Defaults to 0.
            parent_cluster (int, optional): The index of the parent cluster. Defaults to None.

        Returns:
            A dictionary containing the clusters and their members.
        """
        n_components = self.find_optimal_components()
        gmm = GaussianMixture(n_components=n_components, n_init=self.n_init)
        gmm.fit(vectors)
        # Vectors can be assigned to multiple clusters
        probs = gmm.predict_proba(vectors)
        if self.threshold is None:
            logging.info(f"Threshold is None, using better-than-uniform probability")
            self.threshold = 1 / n_components

        labels = [np.where(prob > self.threshold)[0] for prob in probs]
        return labels, n_components

    def cluster_nodes(self, nodes):
        cluster_token_count = sum([node.token_count for node in nodes])
        if cluster_token_count < self.max_token_length:
            return [[nodes]]

        labels, n_components = self.cluster_vectors([node.embedding for node in nodes])
        final_clusters = []
        for cluster_id in range(n_components):
            mask = [cluster_id in label for label in labels]
            member_nodes = [node for c, node in enumerate(nodes) if mask[c]]
            sub_cluster_nodes = self.cluster_nodes(member_nodes)
            final_clusters.extend(sub_cluster_nodes)

        return final_clusters







class FaissVectorDatabase:
    def __init__(self, dims: int, index_file: str = None):
        """
        Initialize the FaissVectorDatabase object with a specified number of dimensions and an optional index file.

        Args:
            dims (int): The number of dimensions in the embeddings.
            index_file (str, optional): The path to the FAISS index file. If provided, the index will be loaded from this file. Otherwise, a new index will be created.
        """
        self.dims = dims
        if index_file is not None:
            self.index = faiss.read_index(index_file)
        else:
            self.index = faiss.IndexFlatL2(dims)

    def add_embeddings(self, ids: List[str], embeddings: np.ndarray):
        """
        Add document embeddings to the index, indexed by their corresponding node elementID strings.

        Args:
            ids (List[str]): A list of id strings for each embedding.
            embeddings (np.ndarray): A 2D numpy array of shape (n_docs, dims) containing the document embeddings.
        """
        self.index.add_with_ids(embeddings, ids)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search the index for the k nearest neighbors of a given query embedding.

        Args:
            query_embedding (np.ndarray): A 1D numpy array of shape (dims,) containing the query embedding.
            k (int, optional): The number of nearest neighbors to return. Defaults to 10.

        Returns:
            A tuple containing two lists: the first list contains the hash strings of the k nearest neighbors, and the second list contains the corresponding distances.
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return indices[0].tolist(), distances[0].tolist()





from neomodel import config

class Neo4JDatabase:
    def __init__(self,
        uri: str,
        user: str,
        password: str,
        ):
        """
        Initialize the Neo4JDatabase object with a graph URL, username, and password.

        Args:
            uri (str): The URL of the Neo4J graph database.
            user (str): The username for the Neo4J graph database.
            password (str): The password for the Neo4J graph database.
        """
        config.DATABASE_URL = f'neo4j+s://{user}:{password}@{uri}'

    def add_tree(self, tree: Tree):
        # Create a node for the source Document
        doc_node = NeoDoc(filepath=document_filepath, name=document_name).save()
        # Create a dictionary to store the Neo4J nodes
        neo_nodes = {}
        # Create all nodes in Neo4J
        for layer, layer_nodes in tree.layer_to_nodes.items():
            for node in layer_nodes:
                neo_node = NeoNode(text=node.text, layer=layer, embeddings=node.embeddings['JinaAI']).save()
                neo_nodes[node.index] = neo_node
                if layer == 0:
                    neo_node.contained_in.connect(doc_node)
                    doc_node.contains.connect(neo_node)
        # Create the relationships in Neo4J
        for node in tree.all_nodes:
            # relationship with children (summary) nodes
            for child_index in node.children:
                child_neo_node = neo_nodes[child_index]
                neo_nodes[node.index].is_summary_of.connect(child_neo_node)




class TreeBuilder:
    def __init__(self,
        tokenizer: AutoTokenizer,
        chunker: SentencePreservingChunker,
        clusterer: RecursiveGMMClustering,
        embedding_model: BaseEmbeddingModel,
        summarization_model: BaseSummarizationModel,
        max_level: int = 3
    ):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.chunker = chunker
        self.clusterer = clusterer
        self.max_layer = max_layer


    def make_leaf_layer(self, all_text: str) -> List[Node]:
        leaf_nodes = []
        chunks_spans = chunker.chunk_spans(all_text)
        for chunk_indices in chunks_spans:
            text = all_text[chunk_indices[0]:chunk_indices[1]]
            text = text.strip()
            embedding = self.embedding_model.create_embedding(text)
            leaf_nodes.append(Node(
                text=text,
                embedding={self.embedding_model.name: embedding},
                span=chunk_indices,
                token_count=len(self.tokenizer.encode(text)),
            ))
        return leaf_nodes


    def make_parent(self, nodes):
        """
        Create a parent node for a list of nodes.

        Args:
            nodes (List[Node]): A list of nodes to create a parent node for.

        Returns:
            A Node object representing the parent node.
        """
        all_text = "\n\n".join([node.text for node in nodes])
        summary = self.summarization_model.summarize(all_text)
        embedding = self.embedding_model.create_embedding(summary)
        parent = Node(
            text=summary,
            embedding={self.embedding_model.name: embedding},
            span=[n.span for n in nodes],
            token_count=len(self.tokenizer.encode(summary)),
            children=nodes
        )
        return parent


    def build_tree_from_document(self, document_path: str):
        with open(self.document_path, "r") as f:
            text = f.read()
        current_layer = self.make_leaf_layer(text)
        layer_to_nodes = {0: current_layer}

        for i in range(1, self.max_level):
            clusters = self.clusterer.cluster_node(current_layer)
            parents = [make_parent(c) for c in tqdm(clusters, desc=f"Building parents for level {i}")]
            current_layer = parents
            layer_to_nodes[i] = current_layer

        return Tree(document_path, current_layer, layer_to_nodes)
