from neomodel import StructuredNode, StringProperty, ArrayProperty, IntegerProperty, RelationshipTo, config
from typing import List

class NeoDoc(StructuredNode):
    filepath = StringProperty(required=True)
    title = StringProperty()
    author = StringProperty()
    subject = StringProperty()
    # has_root = RelationshipTo(NeoNode, "has_root")


class NeoNode(StructuredNode):
    text = StringProperty()
    questions = StringProperty()
    token_count = IntegerProperty()
    breadcrumb = StringProperty()
    page_label = StringProperty()
    bbox = ArrayProperty()
    hash_id = StringProperty()
    refers_to = RelationshipTo('NeoDoc', "refers_to")
    expands_to = RelationshipTo('NeoNode', "expands_to")


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

    def upload_tree(self, tree):
        def recursive_upload(node, doc_node: NeoDoc, parent_node: NeoNode = None, is_root: bool = False):
            neo_node = NeoNode(text=node.text, questions=node.questions, token_count=node.token_count, breadcrumb=node.breadcrumb, page_label=node.page_label, bbox=node.bbox, hash_id=node.hash_id).save()
            if is_root:
                neo_node.refers_to.connect(doc_node)
            if parent_node:
                parent_node.expands_to.connect(neo_node)
            if node.children:
                for child in node.children:
                    recursive_upload(child, doc_node, neo_node)

        doc_node = NeoDoc(**tree.metadata).save()
        for root_node in tree.root_nodes:
            recursive_upload(root_node, doc_node, is_root=True)
        return doc_node

    def get_nodes_by_hash_ids(hash_ids: List[str]) -> List[NeoNode]:
        query = "MATCH (n:NeoNode) WHERE n.hash_id IN {hash_ids} RETURN n"
        params = {"hash_ids": hash_ids}
        results, meta = db.cypher_query(query, params)
        nodes = [NeoNode.inflate(row[0]) for row in results]
        return nodes
