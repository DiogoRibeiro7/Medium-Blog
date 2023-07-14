import numpy as np
import matplotlib.pyplot as plt

def simulate_wealth_distribution(initial_wealth, savings_rate, growth_rate, time_steps):
    wealth = [initial_wealth]

    for t in range(1, time_steps):
        new_wealth = (1 + growth_rate) * wealth[t-1] + savings_rate * wealth[t-1]
        wealth.append(new_wealth)

    return wealth

# Parameters
initial_wealth = 10000
savings_rate = 0.2
growth_rate = 0.03
time_steps = 50

# Simulate wealth distribution
wealth = simulate_wealth_distribution(initial_wealth, savings_rate, growth_rate, time_steps)

# Plot results
plt.plot(range(time_steps), wealth)
plt.xlabel('Time Steps')
plt.ylabel('Wealth')
plt.title('Wealth Distribution Model')
plt.show()
