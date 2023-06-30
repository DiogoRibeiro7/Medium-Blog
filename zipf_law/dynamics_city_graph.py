import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_initial_city_populations(num_cities, population_range):
    # Generate random initial city populations within a given range
    return np.random.randint(population_range[0], population_range[1], num_cities)

def generate_next_city_populations(previous_populations, growth_rate_range):
    # Generate next city populations based on the previous populations and a growth rate range
    growth_rates = np.random.uniform(growth_rate_range[0], growth_rate_range[1], len(previous_populations))
    next_populations = np.round(previous_populations * (1 + growth_rates))
    
    return next_populations

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

def plot_city_graph(city_graph, time_step):
    # Plot the city graph at a specific time step
    pos = nx.spring_layout(city_graph)
    node_sizes = [pop / 10000 for _, pop in nx.get_node_attributes(city_graph, 'population').items()]
    node_colors = node_sizes.copy()

    nx.draw_networkx(city_graph, pos, node_size=node_sizes, node_color=node_colors, cmap='coolwarm', alpha=0.7)
    plt.title(f'City Graph - Time Step {time_step}')
    plt.axis('off')
    plt.show()

# Parameters
num_cities = 50
population_range = (100000, 10000000)
growth_rate_range = (-0.05, 0.05)
num_selected_cities = 5
num_time_steps = 10

# Generate random initial city populations
city_populations = generate_initial_city_populations(num_cities, population_range)

# Create the initial city graph
city_graph = create_city_graph(city_populations)

# Plot the initial city graph
plot_city_graph(city_graph, 0)

# Perform the evolution of city populations over multiple time steps
for t in range(1, num_time_steps + 1):
    # Generate next city populations
    next_populations = generate_next_city_populations(city_populations, growth_rate_range)
    
    # Update city populations in the graph
    nx.set_node_attributes(city_graph, {i: pop for i, pop in enumerate(next_populations)}, 'population')
    
    # Calculate the concentration index at each time step
    concentration_index = calculate_concentration_index(city_graph, num_selected_cities)
    print(f"Concentration Index at Time Step {t}: {concentration_index}")
    
    # Plot the city graph at each time step
    plot_city_graph(city_graph, t)
