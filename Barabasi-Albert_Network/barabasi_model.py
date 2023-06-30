import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Union

# If you are using Jupyter Notebook you may find following two lines useful:
warnings.filterwarnings('ignore')
# % matplotlib inline

def plot_degree_distribution(graph: nx.Graph, scale: str='lin', colour: str='#40a6d1', 
                             alpha: float=.8, expct_lo: int=1, expct_hi: int=10, 
                             expct_const: int=1) -> None:
    """
    This function plots the degree distribution of a network. 
    It can plot the distribution in linear ('lin') or logarithmic ('log') scale.
    
    Parameters:
    graph: nx.Graph: input graph
    scale: str: 'lin' for linear or 'log' for logarithmic scale
    colour: str: colour for the data points
    alpha: float: transparency level for the data points
    expct_lo: int: lower bound for expected degree
    expct_hi: int: upper bound for expected degree
    expct_const: int: constant for expected degree
    
    Returns: None
    """
    try:
        if not isinstance(graph, nx.Graph):
            raise ValueError("The graph must be an instance of networkx.Graph.")
        
        if not isinstance(scale, str) or scale not in ['lin', 'log']:
            raise ValueError("Scale must be either 'lin' or 'log'.")
        
        plt.close()
        degrees = [graph.degree(n) for n in graph.nodes()]
        max_degree = max(degrees)
        x = list(range(max_degree+1))
        y_tmp = [degrees.count(i) for i in x]
        y = [i/graph.number_of_nodes() for i in y_tmp]

        plt.plot(x, y, label='Degree distribution', linewidth=0, marker='o', markersize=8, color=colour, alpha=alpha)

        if scale == 'log':
            plt.xscale('log')
            plt.yscale('log')
            plt.title('Degree distribution (log-log scale)')
            w = list(range(expct_lo, expct_hi))
            z = [(i**-3) * expct_const for i in w] 
            plt.plot(w, z, 'k-', color='#7f7f7f')
        else:
            plt.title('Degree distribution (linear scale)')

        plt.ylabel('P(k)')
        plt.xlabel('k')
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def choose_node_by_degree_prob(graph: nx.Graph) -> int:
    """
    This function selects a node from the graph with a probability proportional 
    to its degree.
    
    Parameters:
    graph: nx.Graph: input graph
    
    Returns: int: node chosen
    """
    try:
        if not isinstance(graph, nx.Graph):
            raise ValueError("The graph must be an instance of networkx.Graph.")
        
        nodes_probs = [graph.degree(node) / (2 * len(graph.edges())) for node in graph.nodes()]
        return np.random.choice(graph.nodes(), p=nodes_probs)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def add_edge_to_new_node(graph: nx.Graph, new_node: int) -> None:
    """
    This function adds an edge to the new node, considering the existing edges.
    
    Parameters:
    graph: nx.Graph: input graph
    new_node: int: the new node that the edge will be added to
    
    Returns: None
    """
    try:
        if not isinstance(graph, nx.Graph):
            raise ValueError("The graph must be an instance of networkx.Graph.")
        if not isinstance(new_node, int):
            raise ValueError("New node must be an integer.")

        if len(graph.edges()) == 0:
            random_node = 0
        else:
            random_node = choose_node_by_degree_prob(graph)
        new_edge = (random_node, new_node)

        if new_edge in graph.edges():
            add_edge_to_new_node(graph, new_node)
        else:
            graph.add_edge(new_node, random_node)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Main flow of the program
print("***\nWelcome to Barabási–Albert (BA) model simulation\nAuthor: Aleksander Molak (2017)\n!איזה כיף\n\n")

init_nodes = int(input("Please type in the initial number of nodes (m_0): "))
final_nodes = int(input("\nPlease type in the final number of nodes: "))
m_parameter = int(input("\nPlease type in the value of m parameter (m<=m_0): "))

G = nx.complete_graph(init_nodes)
print("Graph created. Number of nodes: {}".format(len(G.nodes())))
print("Adding nodes...")

for count in range(final_nodes - init_nodes):
    print("----------> Step {} <----------".format(count))
    G.add_node(init_nodes + count)
    print("Node added: {}".format(init_nodes + count + 1))

    for e in range(0, m_parameter):
        add_edge_to_new_node(G, init_nodes + count)

print(f"\nFinal number of nodes ({len(G.nodes())}) reached")

# Now let's visualize the degree distribution
print("\nVisualizing degree distribution...")
plot_degree_distribution(G, scale='log')

print("\n\nThanks! !תודה רבה :)\n")

