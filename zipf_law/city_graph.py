import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_city_populations(num_cities, population_range):
    # Generate random city populations within a given range
    return np.random.randint(population_range[0], population_range[1], num_cities)

def create_city_graph(city_populations):
    # Create a graph with cities as nodes and population as node attributes
    G = nx.Graph()
    num_cities = len(city_populations)
    G.add_nodes_from(range(num_cities))
    nx.set_node_attributes(G, {i: pop for i, pop in enumerate(city_populations)}, 'population')
    
    return G

def calculate_concentration_index(city_graph, num_selected_cities):
    # Calculate the concentration index based on the population of selected cities
    sorted_populations = sorted(nx.get_node_attributes(city_graph, 'population').items(), key=lambda x: x[1], reverse=True)
    selected_populations = [pop for _, pop in sorted_populations[:num_selected_cities]]
    total_population = sum(pop for _, pop in sorted_populations)
    selected_population = sum(selected_populations)

    concentration_index = selected_population / total_population

    return concentration_index

def plot_city_graph(city_graph):
    # Plot the city graph
    pos = nx.spring_layout(city_graph)
    node_sizes = [pop / 10000 for _, pop in nx.get_node_attributes(city_graph, 'population').items()]
    node_colors = node_sizes.copy()

    nx.draw_networkx(city_graph, pos, node_size=node_sizes, node_color=node_colors, cmap='coolwarm', alpha=0.7)
    plt.title('City Graph')
    plt.axis('off')
    plt.show()

# Parameters
num_cities = 50
population_range = (100000, 10000000)
num_selected_cities = 5

# Generate random city populations
city_populations = generate_city_populations(num_cities, population_range)

# Create the city graph
city_graph = create_city_graph(city_populations)

# Plot the city graph
plot_city_graph(city_graph)

# Calculate the concentration index
concentration_index = calculate_concentration_index(city_graph, num_selected_cities)
print(f"Concentration Index: {concentration_index}")
