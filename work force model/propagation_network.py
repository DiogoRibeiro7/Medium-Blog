import os
import shutil
from PIL import Image
import imageio.v2 as imageio
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy


class Network:

    def create_source(self):
        """Create the source node for the network."""
        self.G = nx.DiGraph()
        self.G.add_node(0, Informed=1)

    def add_node(self):
        """Add a new node to the network."""
        index = len(self.G.nodes)
        self.G.add_node(index, Informed=0)

    def add_connection(self, node1, node2):
        """Add a connection between node1 and node2."""
        self.G.add_edge(node1, node2)
        self.G[node1][node2]["Connection"] = nx.degree_centrality(self.G)[node1]

    def propagate_information(self, resistance):
        """Propagate information through the network based on the resistance level."""
        # Create a temporary dictionary to hold updated information status
        new_informed_status = {node: self.G.nodes[node]["Informed"] for node in self.G.nodes}
        
        for edge in self.G.edges:
            rand = np.random.uniform(0, resistance)
            if self.G.nodes[edge[0]]["Informed"] == 1 and rand < self.G[edge[0]][edge[1]]["Connection"]:
                new_informed_status[edge[1]] = 1
        
        # Update the information status of all nodes
        for node in self.G.nodes:
            self.G.nodes[node]["Informed"] = new_informed_status[node]


# def visualize_network(network):
#     """Visualize the network."""
#     pos = nx.kamada_kawai_layout(network)
#     color_dict = {0: "red", 1: "green"}
#     colors = [color_dict[network.nodes[node]["Informed"]] for node in network.nodes]
#     nx.draw(network, node_color=colors, arrowsize=20, pos=pos)
    
    
def visualize_network(network, ax):
    """
    Visualize the network.

    Parameters:
    network (networkx.Graph): The network graph.
    ax (matplotlib.axes.Axes): The axis object to draw the network on.

    """
    ax.clear()  # Clear previous drawing
    pos = nx.kamada_kawai_layout(network)
    color_dict = {0: "red", 1: "green"}
    colors = [color_dict[network.nodes[node]["Informed"]] for node in network.nodes]
    nx.draw(network, node_color=colors, arrowsize=20, pos=pos, ax=ax)


# Create and setup the network
network = Network()
network.create_source()
for _ in range(100):
    network.add_node()
    
# Test the propagate_information function
initial_state = deepcopy(network.G)
network.propagate_information(0.3)

# Randomly add connections
nodes = list(network.G.nodes)
for _ in range(300):
    node1, node2 = np.random.choice(nodes, 2, replace=False)
    network.add_connection(node1, node2)
    
    
# Initialize plot
fig, ax = plt.subplots()

def update(num):
    network.propagate_information(0.3)
    visualize_network(network.G, ax)
    # plt.show()

ani = FuncAnimation(fig, update, frames=100)
ani.save('network_animation.gif', writer='pillow', fps=10)

# # Create a directory to store the frames
# os.makedirs("frames", exist_ok=True)

# # Simulate and visualize the spread of information
# informed = []
# for i in range(100):
#     print(f"Time Step {i}")
#     network.propagate_information(0.3)
#     informed.append(sum([data for node, data in network.G.nodes(data="Informed")]))
    
#     plt.clf()
#     visualize_network(network.G)
#     plt.savefig(f"frames/frame_{i:03d}.png", bbox_inches='tight')
#     plt.close()

# # Create an animated GIF
# with imageio.get_writer('network_animation.gif', mode='I', duration=0.1) as writer:
#     for i in range(100):
#         image = imageio.imread(f"frames/frame_{i:03d}.png")
#         writer.append_data(image)

# # Delete the "frames" folder
# shutil.rmtree("frames")

