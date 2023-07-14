import numpy as np
import matplotlib.pyplot as plt

def simulate_economic_diffusion(population_size, initial_infection_rates, contact_matrix, recovery_rates, time_steps):
    num_groups = len(initial_infection_rates)

    s = np.zeros((num_groups, time_steps))
    i = np.zeros((num_groups, time_steps))
    r = np.zeros((num_groups, time_steps))

    s[:, 0] = population_size - np.array(initial_infection_rates)
    i[:, 0] = initial_infection_rates

    for t in range(1, time_steps):
        for g in range(num_groups):
            new_infections = np.sum(i[:, t-1] * contact_matrix[g, :]) * recovery_rates[g]
            new_recoveries = i[g, t-1] * (1 - recovery_rates[g])

            s[g, t] = s[g, t-1] - new_infections
            i[g, t] = new_infections
            r[g, t] = r[g, t-1] + new_recoveries

    return s, i, r

# Parameters
population_size = 10000
initial_infection_rates = [100, 200, 300]  # Initial infection rates for each economic group
recovery_rates = [0.1, 0.2, 0.3]  # Recovery rates for each economic group
contact_matrix = np.array([[0.2, 0.3, 0.1], [0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])  # Contact rates between economic groups
time_steps = 50

# Simulate economic diffusion
s, i, r = simulate_economic_diffusion(population_size, initial_infection_rates, contact_matrix, recovery_rates, time_steps)

# Plot results
groups = ['Group 1', 'Group 2', 'Group 3']

for g in range(len(groups)):
    plt.plot(range(time_steps), s[g], label='Susceptible ' + groups[g])
    plt.plot(range(time_steps), i[g], label='Infectious ' + groups[g])
    plt.plot(range(time_steps), r[g], label='Recovered ' + groups[g])

plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.title('Economic Diffusion Model')
plt.legend()
plt.show()
