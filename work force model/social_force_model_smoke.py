# Importing required libraries
from scipy.spatial.distance import euclidean
import math
import numpy as np


def solve_advection_diffusion_eq(grid_points, diffusion_constant, velocity_field, source_term, dt, dx, dy):
    """
    Solves the advection-diffusion equation using a finite-difference method.

    Parameters:
    - grid_points: 2D numpy array representing the concentration at each grid point
    - diffusion_constant: Diffusion constant (D)
    - velocity_field: 2D array containing the velocity field [vx, vy]
    - source_term: Source term (S)
    - dt: Time step
    - dx, dy: Grid spacing in the x and y directions

    Returns:
    - Updated grid_points after one time step
    """
    # Extract the velocity components
    vx, vy = velocity_field

    # Compute the Laplacian term
    laplacian_term = (
        np.roll(grid_points, -1, axis=0) + np.roll(grid_points, 1, axis=0) +
        np.roll(grid_points, -1, axis=1) + np.roll(grid_points, 1, axis=1) -
        4 * grid_points
    ) / (dx * dy)

    # Compute the advection term (ignoring boundary points)
    adv_x = -(grid_points - np.roll(grid_points, -1, axis=0)) / dx
    adv_y = -(grid_points - np.roll(grid_points, -1, axis=1)) / dy
    advection_term = vx * adv_x + vy * adv_y

    # Update the concentration at each grid point
    grid_points += dt * (diffusion_constant *
                         laplacian_term - advection_term + source_term)

    return grid_points

# # Test the function with sample parameters
# grid_points = np.random.rand(5, 5)
# diffusion_constant = 0.1
# velocity_field = [0.2, 0.1]
# source_term = 0.05
# dt = 0.01
# dx = dy = 0.1

# # Run one time step of the advection-diffusion equation
# updated_grid_points = solve_advection_diffusion_eq(grid_points, diffusion_constant, velocity_field, source_term, dt, dx, dy)
# print(updated_grid_points)


def compute_visibility_distance(person, extinction_coefficient, smoke_mass):
    """
    Computes the visibility distance of a person using equation 7 and 12.

    Parameters:
    - person: Dictionary containing information about the person
    - extinction_coefficient: Extinction coefficient of smoke (kappa)
    - smoke_mass: Smoke mass at time t (M)

    Returns:
    - Updated 'person' dictionary with the computed 'visibility_distance'
    """
    # Given constant 7.6 m^2/g for light-reflecting objects (equation 12)
    constant = 7.6

    # Compute visibility distance using equation 7
    visibility_distance = constant / (extinction_coefficient * smoke_mass)

    # Update the 'person' dictionary with the computed 'visibility_distance'
    person['visibility_distance'] = visibility_distance

    return person


# # Test the function with sample parameters
# person = {'id': 1}
# extinction_coefficient = 0.5
# smoke_mass = 0.3

# # Compute the visibility distance
# updated_person = compute_visibility_distance(
#     person, extinction_coefficient, smoke_mass)
# print(updated_person)


def compute_pedestrian_density(person, persons, radius):
    """
    Computes the pedestrian density around a person in a ball of given radius.

    Parameters:
    - person: Dictionary containing information about the person
    - persons: List of dictionaries, each containing information about a person
    - radius: Radius of the ball within which to calculate pedestrian density

    Returns:
    - Updated 'person' dictionary with the computed 'pedestrian_density'
    """
    # Initialize pedestrian count within the radius
    count = 0

    # Coordinates of the person for whom we are calculating the density
    person_coordinates = person['position']

    # Loop through each person to check if they are within the radius of 'person'
    for other_person in persons:
        if person != other_person:
            other_person_coordinates = other_person['position']
            print(other_person_coordinates)
            distance = euclidean(person_coordinates, other_person_coordinates)

            if distance <= radius:
                count += 1

    # Compute pedestrian density: Number of persons within the radius divided by the area of the circle
    pedestrian_density = count / (math.pi * radius ** 2)

    # Update the 'person' dictionary with the computed 'pedestrian_density'
    person['pedestrian_density'] = pedestrian_density

    return person


# # Test the function with sample parameters
# person = {'id': 1, 'position': [0, 0]}
# persons = [
#     {'id': 1, 'position': [0, 0]},
#     {'id': 2, 'position': [0.1, 0.1]},
#     {'id': 3, 'position': [0.5, 0.5]},
#     {'id': 4, 'position': [1, 1]}
# ]
# radius = 0.6

# # Compute the pedestrian density
# updated_person = compute_pedestrian_density(person, persons, radius)
# print(updated_person)


def solve_eikonal_eq(person, simulation_domain, arrival_time, moving_speed):
    """
    Solves the Eikonal equation (equation 11). 
    This is a simplified example, as solving the Eikonal equation can be quite complex.

    Parameters:
    - person: Dictionary containing information about the person
    - simulation_domain: 2D list defining the corners of the simulation domain
    - arrival_time: Arrival time of the front crossing the point
    - moving_speed: Moving speed of the front

    Returns:
    - Updated 'person' dictionary with the solution of the Eikonal equation ('travel_cost')
    """
    # Coordinates of the person
    x, y = person['position']

    # Coordinates defining the simulation domain
    x_min, x_max = simulation_domain[0]
    y_min, y_max = simulation_domain[1]

    # Calculate travel cost (this is a simplified calculation; actual solutions could involve numerical methods)
    travel_cost = arrival_time + ((x - x_min) + (y - y_min)) / moving_speed

    # Update the 'person' dictionary with the computed 'travel_cost'
    person['travel_cost'] = travel_cost

    return person

# # Test the function with sample parameters
# person = {'id': 1, 'position': [0.5, 0.5]}
# simulation_domain = [[0, 1], [0, 1]]
# arrival_time = 2.0
# moving_speed = 1.0

# # Solve the Eikonal equation for the person
# updated_person = solve_eikonal_eq(person, simulation_domain, arrival_time, moving_speed)
# print(updated_person)


def set_intended_speed(person, max_speed, pedestrian_density, visibility_distance):
    """
    Sets the intended speed of a person based on equation 2.

    Parameters:
    - person: Dictionary containing information about the person
    - max_speed: Maximum speed a person can achieve (v_max)
    - pedestrian_density: Pedestrian density around the person (rho)
    - visibility_distance: Visibility distance of the person (d)

    Returns:
    - Updated 'person' dictionary with the computed 'intended_speed'
    """
    # Calculate the intended speed based on equation 2
    intended_speed = max_speed * (1 - pedestrian_density / visibility_distance)

    # Ensure that the intended speed is non-negative
    intended_speed = max(0, intended_speed)

    # Update the 'person' dictionary with the computed 'intended_speed'
    person['intended_speed'] = intended_speed

    return person

# # Test the function with sample parameters
# person = {'id': 1}
# max_speed = 1.5
# pedestrian_density = 0.884
# visibility_distance = 50.67

# # Set the intended speed of the person
# updated_person = set_intended_speed(person, max_speed, pedestrian_density, visibility_distance)
# print(updated_person)


def compute_desire_force(person, intended_speed):
    """
    Computes the desire force for a person based on equation 1.

    Parameters:
    - person: Dictionary containing information about the person
    - intended_speed: Intended speed of the person (v_0)

    Returns:
    - Updated 'person' dictionary with the computed 'desire_force'
    """
    # Constants (for simplicity, assuming mass m=1 and relaxation time tau=0.5)
    mass = 1
    tau = 0.5

    # Extract current velocity of the person
    current_velocity = np.array(person.get('velocity', [0, 0]))

    # Calculate the desire force based on equation 1
    # Adjusting intended_speed to be a vector in the direction of the current_velocity
    intended_speed_vector = intended_speed * \
        current_velocity / np.linalg.norm(current_velocity)
    desire_force = mass * (intended_speed_vector - current_velocity) / tau

    # Update the 'person' dictionary with the computed 'desire_force'
    person['desire_force'] = desire_force.tolist()

    return person


# # Test the function with sample parameters
# person = {'id': 1, 'velocity': [1, 1]}
# intended_speed = 1.474

# # Compute the desire force for the person
# updated_person = compute_desire_force(person, intended_speed)
# print(updated_person)


def compute_repulsive_force(person, other_person):
    """
    Computes the repulsive social force between two persons based on equation 16.

    Parameters:
    - person: Dictionary containing information about the person
    - other_person: Dictionary containing information about the other person

    Returns:
    - Updated 'person' dictionary with the computed 'repulsive_force'
    """
    # Constants (for simplicity, assuming interaction strength A=2000 and range B=0.08)
    A = 2000
    B = 0.08

    # Extract positions of the persons
    position1 = np.array(person.get('position', [0, 0]))
    position2 = np.array(other_person.get('position', [0, 0]))

    # Calculate distance between the persons
    distance = np.linalg.norm(position1 - position2)

    # Calculate the normalized direction vector
    direction_vector = (position1 - position2) / distance

    # Calculate the repulsive force based on equation 16
    repulsive_force = A * np.exp(-distance / B) * direction_vector

    # Update the 'person' dictionary with the computed 'repulsive_force'
    person.setdefault('repulsive_forces', []).append(repulsive_force.tolist())

    return person

# # Test the function with sample parameters
# person1 = {'id': 1, 'position': [0, 0]}
# person2 = {'id': 2, 'position': [0.1, 0.1]}

# # Compute the repulsive force between the persons
# updated_person = compute_repulsive_force(person1, person2)
# print(updated_person)


def compute_physical_force(person, other_person):
    """
    Computes the physical interaction force between two persons based on the provided equations.

    Parameters:
    - person: Dictionary containing information about the person
    - other_person: Dictionary containing information about the other person

    Returns:
    - Updated 'person' dictionary with the computed 'physical_force'
    """
    # Constants (for simplicity, assuming normal and tangential constants k_n=1.2e5 and k_t=2.4e5)
    k_n = 1.2e5
    k_t = 2.4e5

    # Extract positions and velocities of the persons
    position1 = np.array(person.get('position', [0, 0]))
    position2 = np.array(other_person.get('position', [0, 0]))
    velocity1 = np.array(person.get('velocity', [0, 0]))
    velocity2 = np.array(other_person.get('velocity', [0, 0]))

    # Calculate distance and direction between the persons
    distance = np.linalg.norm(position1 - position2)
    direction_vector = (position1 - position2) / distance

    # Calculate the tangential direction
    tangential_vector = np.array([-direction_vector[1], direction_vector[0]])

    # Calculate the tangential velocity difference
    delta_velocity_t = np.dot((velocity1 - velocity2), tangential_vector)

    # Calculate the physical interaction force
    normal_force = k_n * distance * direction_vector
    tangential_force = k_t * distance * delta_velocity_t * tangential_vector
    physical_force = normal_force + tangential_force

    # Update the 'person' dictionary with the computed 'physical_force'
    person.setdefault('physical_forces', []).append(physical_force.tolist())

    return person

# # Test the function with sample parameters
# person1 = {'id': 1, 'position': [0, 0], 'velocity': [1, 1]}
# person2 = {'id': 2, 'position': [0.1, 0.1], 'velocity': [0.9, 0.9]}

# # Compute the physical interaction force between the persons
# updated_person = compute_physical_force(person1, person2)
# print(updated_person)


def solve_social_force_model(person):
    """
    Solves the social force model equations (equations 1 and 2). 
    This is a simplified example as solving the social force model can be complex.

    Parameters:
    - person: Dictionary containing information about the person

    Returns:
    - Updated 'person' dictionary with the solution of the social force model equations
    """
    # Constants (for simplicity, assuming mass m=1, time step dt=0.1, and relaxation time tau=0.5)
    mass = 1
    dt = 0.1
    tau = 0.5

    # Extract current velocity and forces from the person dictionary
    current_velocity = np.array(person.get('velocity', [0, 0]))
    desire_force = np.array(person.get('desire_force', [0, 0]))
    repulsive_forces = np.array(person.get('repulsive_forces', [[0, 0]]))
    physical_forces = np.array(person.get('physical_forces', [[0, 0]]))

    # Sum up the repulsive and physical forces
    total_other_forces = np.sum(
        repulsive_forces, axis=0) + np.sum(physical_forces, axis=0)

    # Calculate the total force based on equation 1
    total_force = desire_force + total_other_forces

    # Update the velocity and position based on equations 1 and 2
    new_velocity = current_velocity + (total_force / mass) * dt
    new_position = np.array(person.get('position', [0, 0])) + new_velocity * dt

    # Update the 'person' dictionary with the new velocity and position
    person['velocity'] = new_velocity.tolist()
    person['position'] = new_position.tolist()

    return person

# # Test the function with sample parameters
# person = {
#     'id': 1,
#     'position': [0, 0],
#     'velocity': [1, 1],
#     'desire_force': [0.5, 0.5],
#     'repulsive_forces': [[0.2, 0.2], [0.1, 0.1]],
#     'physical_forces': [[0.05, 0.05]]
# }

# # Solve the social force model for the person
# updated_person = solve_social_force_model(person)
# print(updated_person)


def update_positions_velocities(person):
    """
    Updates the positions and velocities of a person based on the social force model's solutions.

    Parameters:
    - person: Dictionary containing information about the person

    Returns:
    - Updated 'person' dictionary with the new 'position' and 'velocity'
    """
    # Constants (for simplicity, assuming time step dt=0.1)
    dt = 0.1

    # Extract current velocity from the person dictionary
    current_velocity = np.array(person.get('velocity', [0, 0]))

    # Update the position based on the current velocity and time step
    new_position = np.array(person.get(
        'position', [0, 0])) + current_velocity * dt

    # Update the 'person' dictionary with the new position and velocity
    person['position'] = new_position.tolist()

    return person

# # Test the function with sample parameters
# person = {
#     'id': 1,
#     'position': [0, 0],
#     'velocity': [1, 1]
# }

# # Update the positions and velocities of the person
# updated_person = update_positions_velocities(person)
# print(updated_person)


# def main_update_algorithm(input_params, grid_points, persons):
#     """
#     Main update algorithm for pedestrian evacuation simulation.

#     Parameters:
#     - input_params: Dictionary containing input parameters like time_steps, high_smoke_density, etc.
#     - grid_points: 2D array representing the simulation domain grid
#     - persons: List of dictionaries, each containing information about a person

#     Returns:
#     - Updated state of persons
#     """

#     # Loop through time steps
#     for t in range(input_params['time_steps']):

#         # Solve the advection-diffusion equation for each grid point
#         for point in grid_points:
#             solve_advection_diffusion_eq(grid_points, input_params['diffusion_constant'],
#                                          input_params['velocity_field'], input_params['source_term'])

#         # Find grid points with high smoke density
#         high_smoke_density_points = [
#             point for point in grid_points if point >= input_params['high_smoke_density']]

#         # Loop through each person to update their states
#         for i, person in enumerate(persons):

#             # Compute the visibility distance of person i
#             compute_visibility_distance(
#                 person, input_params['extinction_coefficient'], input_params['smoke_mass'])

#             # Compute the pedestrian density around person i
#             compute_pedestrian_density(person, persons, input_params['radius'])

#             # Solve the Eikonal equation for person i
#             solve_eikonal_eq(person, input_params['simulation_domain'],
#                              input_params['arrival_time'], input_params['moving_speed'])

#             # Set the intended speed of person i
#             set_intended_speed(
#                 person, input_params['max_speed'], person['pedestrian_density'], person['visibility_distance'])

#             # Compute the desire force for person i
#             compute_desire_force(person, person['intended_speed'])

#             # Compute repulsive and physical forces for person i with respect to other persons
#             for j, other_person in enumerate(persons):
#                 if i != j:
#                     compute_repulsive_force(person, other_person)
#                     compute_physical_force(person, other_person)

#             # Solve the social force model for person i
#             solve_social_force_model(person)

#             # Update the positions and velocities of person i
#             update_positions_velocities(person)

#     return persons


# Correcting the main_update_algorithm function to pass the missing arguments to solve_advection_diffusion_eq

def main_update_algorithm(input_params, grid_points, persons):
    """
    Main update algorithm for pedestrian evacuation simulation.

    Parameters:
    - input_params: Dictionary containing input parameters like time_steps, high_smoke_density, etc.
    - grid_points: 2D array representing the simulation domain grid
    - persons: List of dictionaries, each containing information about a person

    Returns:
    - Updated state of persons
    """

    # Extract additional required parameters for solve_advection_diffusion_eq
    dt = input_params['dt']
    dx = input_params['dx']
    dy = input_params['dy']

    # Loop through time steps
    for t in range(input_params['time_steps']):

        # Solve the advection-diffusion equation for each grid point
        for point in grid_points:
            solve_advection_diffusion_eq(grid_points, input_params['diffusion_constant'],
                                         input_params['velocity_field'], input_params['source_term'], dt, dx, dy)

        # # Find grid points with high smoke density
        # high_smoke_density_points = [
        #     point for point in grid_points if point >= input_params['high_smoke_density']]

        # Loop through each person to update their states
        for i, person in enumerate(persons):

            # Compute the visibility distance of person i
            compute_visibility_distance(
                person, input_params['extinction_coefficient'], input_params['smoke_mass'])

            # Compute the pedestrian density around person i
            compute_pedestrian_density(person, persons, input_params['radius'])

            # Solve the Eikonal equation for person i
            solve_eikonal_eq(person, input_params['simulation_domain'],
                             input_params['arrival_time'], input_params['moving_speed'])

            # Set the intended speed of person i
            set_intended_speed(
                person, input_params['max_speed'], person['pedestrian_density'], person['visibility_distance'])

            # Compute the desire force for person i
            compute_desire_force(person, person['intended_speed'])

            # Compute repulsive and physical forces for person i with respect to other persons
            for j, other_person in enumerate(persons):
                if i != j:
                    compute_repulsive_force(person, other_person)
                    compute_physical_force(person, other_person)

            # Solve the social force model for person i
            solve_social_force_model(person)

            # Update the positions and velocities of person i
            update_positions_velocities(person)

    return persons

# Function signature corrected.


# Example usage
input_params = {
    'time_steps': 10,
    'high_smoke_density': 0.8,
    'diffusion_constant': 0.1,
    'velocity_field': [0.1, 0.1],
    'source_term': 0.2,
    'extinction_coefficient': 0.5,
    'smoke_mass': 0.3,
    'radius': 1.0,
    'simulation_domain': [[0, 1], [0, 1]],
    'arrival_time': 0.0,
    'moving_speed': 1.0,
    'max_speed': 1.5,
    'dt': 0.1,  # Time step size
    'dx': 0.1,  # Grid spacing in x-direction
    'dy': 0.1  # Grid spacing in y-direction
}

grid_points = np.array([[0.1, 0.2], [0.3, 0.4]])
persons = [{'id': i, 'position': [0.0, 0.0], 'velocity': [0.0, 0.0]}
           for i in range(5)]

# Run the main update algorithm
updated_persons = main_update_algorithm(input_params, grid_points, persons)
updated_persons
