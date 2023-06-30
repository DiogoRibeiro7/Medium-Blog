import networkx as nx
import matplotlib.pyplot as plt

# Number of nodes in the initial network
m0 = 5

# Number of edges to attach from a new node to existing nodes
m = 2

# Create an empty graph
ba_network = nx.Graph()

# Add the initial nodes to the graph
ba_network.add_nodes_from(range(m0))

# Iterate to add new nodes with preferential attachment
for i in range(m0, 50):
    # Select m nodes from the existing network to which the new node will connect
    selected_nodes = list(ba_network.nodes())
    new_edges = []

    # Perform preferential attachment by connecting the new node to existing nodes
    while len(new_edges) < m:
        selected_node = nx.utils.random.choice(selected_nodes)
        if not ba_network.has_edge(i, selected_node):
            new_edges.append((i, selected_node))

    # Add the new edges to the graph
    ba_network.add_edges_from(new_edges)

# Print the degree of each node in the generated Barabási–Albert network
print("Node\tDegree")
for node, degree in ba_network.degree():
    print(node, "\t", degree)

# Draw the network graph
nx.draw(ba_network, with_labels=True, node_size=200, alpha=0.8)
plt.title("Barabási–Albert Network")
plt.show()
