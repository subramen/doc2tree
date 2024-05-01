# %load_ext autoreload
# %autoreload 2

from tree import Tree, Node
from config import Neo4JDriverConfig
from graph_db import Neo4JDriver

config = Neo4JDriverConfig()
n1 = Node('test1', {'vector':[1,2]}, 1)
n2 = Node('test2', {'vector':[3,4]}, 2)
metadata = {'filepath': 'xyz', 'title': 'xyz', 'author':'xyz', 'subject':'xyz'}
tree = Tree([n1, n2], metadata)

db = Neo4JDriver(**config.dict())

db.upload_tree(tree)
