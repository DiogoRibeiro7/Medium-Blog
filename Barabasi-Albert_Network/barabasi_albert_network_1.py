import networkx as nx
import matplotlib.pyplot as plt

def generate_barabasi_albert_network(m0, m, num_nodes):
    # Create an empty graph
    ba_network = nx.Graph()

    # Add the initial nodes to the graph
    ba_network.add_nodes_from(range(m0))

    # Iterate to add new nodes with preferential attachment
    for i in range(m0, num_nodes):
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

    return ba_network

def plot_network(network):
    # Draw the network graph
    nx.draw(network, with_labels=True, node_size=200, alpha=0.8)
    plt.title("Barabási–Albert Network")
    plt.show()

# Generate a Barabási–Albert network with m0 = 5, m = 2, and 50 total nodes
m0 = 5
m = 2
num_nodes = 50
ba_network = generate_barabasi_albert_network(m0, m, num_nodes)

# Print the degree of each node in the generated Barabási–Albert network
print("Node\tDegree")
for node, degree in ba_network.degree():
    print(node, "\t", degree)

# Plot the network graph
plot_network(ba_network)
