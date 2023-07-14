import matplotlib.pyplot as plt

def calculate_consumption(income, consumption_propensity):
    return consumption_propensity * income

def simulate_economy(initial_income, consumption_propensity, iterations):
    incomes = [initial_income]
    consumptions = [calculate_consumption(initial_income, consumption_propensity)]

    for _ in range(iterations):
        income = incomes[-1]
        consumption = calculate_consumption(income, consumption_propensity)

        new_income = income + (0.6 * income)  # Assumption: 60% increase in income
        incomes.append(new_income)
        consumptions.append(calculate_consumption(new_income, consumption_propensity))

    return incomes, consumptions

# Parameters
initial_income = 1000
consumption_propensity = 0.8  # Consumption propensity of 0.8 (80%)
iterations = 10

# Simulate economy
incomes, consumptions = simulate_economy(initial_income, consumption_propensity, iterations)

# Plot results
plt.plot(range(iterations + 1), incomes, label='Income')
plt.plot(range(iterations + 1), consumptions, label='Consumption')
plt.xlabel('Iterations')
plt.ylabel('Amount')
plt.title('Income and Consumption over Iterations')
plt.legend()
plt.show()
