import numpy as np

def generate_next_city_populations(previous_populations, transition_matrix):
    # Generate next city populations based on the transition matrix
    num_cities = len(previous_populations)
    growth_rates = np.random.choice(np.arange(len(transition_matrix)), size=num_cities, p=transition_matrix)
    next_populations = np.round(previous_populations * (1 + growth_rates))
    
    return next_populations

# Parameters
num_cities = 50
population_range = (100000, 10000000)
transition_matrix = [0.2, 0.5, 0.3]  # Example transition probabilities

# Generate random initial city populations
city_populations = generate_initial_city_populations(num_cities, population_range)

# Perform the evolution of city populations over multiple time steps
for t in range(1, num_time_steps + 1):
    # Generate next city populations based on the transition matrix
    next_populations = generate_next_city_populations(city_populations, transition_matrix)
    
    # Update city populations for the next time step
    city_populations = next_populations

    # Calculate the concentration index at each time step
    concentration_index = calculate_concentration_index(city_graph, num_selected_cities)
    print(f"Concentration Index at Time Step {t}: {concentration_index}")
