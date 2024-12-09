import networkx as nx

G = nx.DiGraph()

G.add_node(1, time='5pm')
G.add_nodes_from([3], time='2pm')
print(G.nodes[3])
print(G)