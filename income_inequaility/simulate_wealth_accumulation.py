import numpy as np
import matplotlib.pyplot as plt

def simulate_wealth_accumulation(borrowing_limit, interest_rate, productivity_distribution, num_agents, time_steps):
    # Initialize arrays
    wealth = np.zeros((num_agents, time_steps))
    savings = np.zeros((num_agents, time_steps))

    for t in range(1, time_steps):
        # Generate random productivity shocks
        productivity_shocks = np.random.choice(productivity_distribution, size=num_agents)

        # Calculate savings and update wealth
        for i in range(num_agents):
            savings[i, t] = max(0, wealth[i, t-1] * interest_rate + productivity_shocks[i] - borrowing_limit)
            wealth[i, t] = wealth[i, t-1] + savings[i, t]

    return wealth

# Parameters
borrowing_limit = 0  # Borrowing constraint
interest_rate = 0.03  # Annual interest rate
productivity_distribution = [-0.1, 0, 0.1]  # Productivity shocks distribution
num_agents = 1000
time_steps = 50

# Simulate wealth accumulation
wealth = simulate_wealth_accumulation(borrowing_limit, interest_rate, productivity_distribution, num_agents, time_steps)

# Plot wealth trajectories for a few agents
agents_to_plot = np.random.choice(num_agents, size=5, replace=False)

for agent_idx in agents_to_plot:
    plt.plot(range(time_steps), wealth[agent_idx])

plt.xlabel('Time Steps')
plt.ylabel('Wealth')
plt.title('Wealth Accumulation - Aiyagari Model')
plt.show()
