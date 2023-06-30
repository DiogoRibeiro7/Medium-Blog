import matplotlib.pyplot as plt
import numpy as np

def generate_city_populations(num_cities, population_range):
    # Generate city populations following Zipf's Law
    ranks = np.arange(1, num_cities + 1)
    populations = np.round(population_range[0] / (ranks ** 0.8))

    return populations

def calculate_concentration_index(city_populations, num_selected_cities):
    # Calculate the concentration index based on the population distribution
    sorted_populations = np.sort(city_populations)[::-1]
    selected_populations = sorted_populations[:num_selected_cities]
    total_population = np.sum(sorted_populations)
    selected_population = np.sum(selected_populations)

    concentration_index = selected_population / total_population

    return concentration_index

def plot_population_distribution(city_populations):
    # Plot the population distribution
    ranks = np.arange(1, len(city_populations) + 1)
    plt.scatter(ranks, city_populations, color='blue')
    plt.xlabel('Rank')
    plt.ylabel('Population')
    plt.title('Population Distribution of Cities')
    plt.show()

# Parameters
num_cities = 1000
population_range = (100000, 10000000)
num_selected_cities = 10

# Generate city populations based on Zipf's Law
city_populations = generate_city_populations(num_cities, population_range)

# Plot the population distribution
plot_population_distribution(city_populations)

# Calculate the concentration index
concentration_index = calculate_concentration_index(city_populations, num_selected_cities)
print(f"Concentration Index: {concentration_index}")
