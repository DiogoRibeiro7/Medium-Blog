import matplotlib.pyplot as plt
import numpy as np

def generate_city_populations(num_cities, population_range):
    # Generate random city populations within a given range
    return np.random.randint(population_range[0], population_range[1], num_cities)

def plot_zipf_law(city_populations):
    # Rank the cities based on population
    ranked_populations = np.sort(city_populations)[::-1]
    ranks = np.arange(1, len(ranked_populations) + 1)

    # Calculate the exponent 'alpha' of Zipf's Law
    alpha = np.polyfit(np.log(ranks), np.log(ranked_populations), 1)[0]

    # Generate the predicted population based on Zipf's Law
    predicted_populations = ranked_populations[0] / (ranks ** alpha)

    # Plot the observed and predicted populations
    plt.scatter(ranks, ranked_populations, color='blue', label='Observed Populations')
    plt.plot(ranks, predicted_populations, color='red', linestyle='--', label='Zipf\'s Law Prediction')
    plt.xlabel('Rank')
    plt.ylabel('Population')
    plt.title('Zipf\'s Law: City Population Distribution')
    plt.legend()
    plt.show()

# Example usage
num_cities = 100
population_range = (100000, 10000000)

# Generate random city populations
city_populations = generate_city_populations(num_cities, population_range)

# Plot the distribution based on Zipf's Law
plot_zipf_law(city_populations)
