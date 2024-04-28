from neomodel import StructuredNode, StringProperty, ArrayProperty, IntProperty, RelationshipTo


class NeoDoc(StructuredNode):
    filepath = StringProperty(required=True)
    title = StringProperty()
    author = StringProperty()
    subject = StringProperty()
    has_root = RelationshipTo(NeoNode, "has_root")


class NeoNode(StructuredNode):
    text = StringProperty()
    embedding = ArrayProperty()
    span = ArrayProperty()
    token_count = IntProperty()
    hash_id = StringProperty()
    refers_to = RelationshipTo(NeoDoc, "refers_to")
    expands_to = RelationshipTo(NeoNode, "expands_to")


class Neo4JDriver:
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
        self.uri = uri
        self.user = user
        self.password = password
        self.set_database_url()

    def set_database_url(self):
        config.DATABASE_URL = f'neo4j+s://{self.user}:{self.password}@{self.uri}'

    def upload_tree(self, document_path: str, tree: Tree, embedding_name: str = 'JinaAI', document_metadata: Optional[Dict[str, str]]={}):
        def recursive_upload(node: Node, element_ids: List[str], doc_node: NeoDoc, parent_node: NeoNode = None):
            neo_node = NeoNode(text=node.text, embedding=node.embedding[embedding_name], span=node.span, token_count=node.token_count, hash_id=node.hash_id).save()
            neo_node.refers_to.connect(doc_node)
            if parent_node is None:  # only for the root node
                doc_node.has_root.connect(neo_node)
            else:
                parent_node.expands_to.connect(neo_node)
            for child in node.children:
                recursive_upload(child, doc_node, neo_node)

        doc_node = NeoDoc(filepath=tree.document_url, **document_metadata).save()
        for root_node in tree.root_nodes:
            recursive_upload(root_node, doc_node)
        return doc_node
