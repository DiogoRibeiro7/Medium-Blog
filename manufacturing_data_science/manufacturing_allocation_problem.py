from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus, LpInteger

# Dynamic data: task names and their respective time requirements
tasks = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5']
time_requirements = {
    'Task1': {'Laser': 2, 'Iron Mark': 3, 'Ink': 4},
    'Task2': {'Laser': 3, 'Iron Mark': 2, 'Ink': 5},
    'Task3': {'Laser': 4, 'Iron Mark': 1, 'Ink': 6},
    'Task4': {'Laser': 1, 'Iron Mark': 5, 'Ink': 3},
    'Task5': {'Laser': 2, 'Iron Mark': 4, 'Ink': 2},
}

# Dynamic data: number of machines available for each type
machines_available = {
    'Laser': 1,
    'Iron Mark': 5,
    'Ink': 5,
}

# Dynamic data: stock availability for each material
stock_availability = {
    'Material1': 10,
    'Material2': 8,
    'Material3': 15,
}

# Dynamic data: machine efficiency or speed
machine_efficiency = {
    'Laser': 1.2,
    'Iron Mark': 1.0,
    'Ink': 0.8,
}

# Dynamic data: setup time matrix (in minutes)
setup_time = {
    ('Laser', 'Iron Mark'): 10,
    ('Iron Mark', 'Laser'): 15,
    ('Laser', 'Ink'): 5,
    ('Ink', 'Laser'): 8,
    ('Iron Mark', 'Ink'): 3,
    ('Ink', 'Iron Mark'): 4,
}

# Create the LP problem
problem = LpProblem("MachineTimeAllocation", LpMaximize)

# Define decision variables
allocation = LpVariable.dicts('allocation', [(task, machine) for task in tasks for machine in machines_available], 0, 1, LpInteger)

# Define the objective function
problem += lpSum([allocation[task, machine] for task in tasks for machine in machines_available]), "Total Production"

# Add the constraints
for task in tasks:
    problem += lpSum([allocation[task, machine] for machine in machines_available]) == 1, f"AllocationConstraint_{task}"

for machine, available_count in machines_available.items():
    problem += lpSum([allocation[task, machine] for task in tasks]) <= available_count, f"MachineAvailability_{machine}"

for task in tasks:
    for machine in machines_available:
        problem += allocation[task, machine] * time_requirements[task][machine] <= machine_efficiency[machine] * lpSum([allocation[task, m] * time_requirements[task][m] for m in machines_available]), f"MachineEfficiency_{task}_{machine}"

for task1 in tasks:
    for task2 in tasks:
        for machine1 in machines_available:
            for machine2 in machines_available:
                if (machine1, machine2) in setup_time:
                    problem += allocation[task1, machine1] + allocation[task2, machine2] <= 1 + allocation[task1, machine2], f"SetupTime_{task1}_{machine1}_{task2}_{machine2}"

# Solve the problem
problem.solve()

# Print the status of the solution
print("Status:", LpStatus[problem.status])

# Print the optimal allocation of machine time
for task in tasks:
    for machine in machines_available:
        if allocation[task, machine].varValue == 1:
            print(f"Task '{task}' allocated to {machine}")

# Print the total production achieved
print("Total Production:", problem.objective.value())

# Print the allocation of machine time for each task
print("Machine Time Allocation:")
for task in tasks:
    allocation_per_machine = []
for machine in machines_available:
    allocation_value = allocation[task, machine].varValue
    allocation_per_machine.append((machine, allocation_value))
print(f"Task '{task}': {allocation_per_machine}")

