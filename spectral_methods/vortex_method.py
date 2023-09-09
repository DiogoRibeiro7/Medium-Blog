import numpy as np
import matplotlib.pyplot as plt

class Vortex:
    """Class to represent a point vortex."""
    def __init__(self, x, y, strength):
        """
        Initialize the vortex.

        Parameters:
        x (float): x-coordinate of the vortex
        y (float): y-coordinate of the vortex
        strength (float): strength of the vortex
        """
        self.x = x
        self.y = y
        self.strength = strength

def update_positions(vortices, dt, box_size):
    """
    Update the positions of the vortices.

    Parameters:
    vortices (list of Vortex): List of vortex objects
    dt (float): Time step for Euler integration
    box_size (tuple): Tuple specifying the dimensions of the bounding box (x_max, y_max)

    Returns:
    list of Vortex: Updated list of vortex objects
    """
    for i, vortex_i in enumerate(vortices):
        # Initialize net velocity components for each vortex
        Vx_net = 0
        Vy_net = 0

        for j, vortex_j in enumerate(vortices):
            if i != j:
                dx = vortex_j.x - vortex_i.x
                dy = vortex_j.y - vortex_i.y
                r_squared = dx**2 + dy**2

                # Compute velocity components induced by each vortex
                Vx = vortex_j.strength / (2 * np.pi) * dy / r_squared
                Vy = -vortex_j.strength / (2 * np.pi) * dx / r_squared

                # Update net velocity components
                Vx_net += Vx
                Vy_net += Vy

        # Update vortex positions using Euler integration
        vortex_i.x += Vx_net * dt
        vortex_i.y += Vy_net * dt

        # Ensure the vortex stays within the bounding box
        vortex_i.x = np.clip(vortex_i.x, 0, box_size[0])
        vortex_i.y = np.clip(vortex_i.y, 0, box_size[1])

    return vortices

from random import uniform

def initialize_vortices(n, box_size, strength=1):
    """
    Initialize n vortices at random positions within the box.

    Parameters:
    n (int): Number of vortices to initialize
    box_size (tuple): Tuple specifying the dimensions of the bounding box (x_max, y_max)
    strength (float): Strength of each vortex

    Returns:
    list of Vortex: List of initialized vortex objects
    """
    vortices = []
    for _ in range(n):
        x = uniform(0, box_size[0])
        y = uniform(0, box_size[1])
        vortex = Vortex(x, y, strength)
        vortices.append(vortex)
    return vortices

def simulate_vortices(n_steps, dt, box_size):
    """
    Simulate the movement of vortices within a bounded box.

    Parameters:
    n_steps (int): Number of time steps for the simulation
    dt (float): Time step for Euler integration
    box_size (tuple): Tuple specifying the dimensions of the bounding box (x_max, y_max)

    Returns:
    list of list of Vortex: List of vortex positions at each time step
    """
    # Initialize three vortices
    vortices = initialize_vortices(3, box_size)
    
    # List to store vortex positions at each time step
    history = []
    
    for _ in range(n_steps):
        # Update vortex positions
        vortices = update_positions(vortices, dt, box_size)
        
        # Store current positions
        history.append([Vortex(v.x, v.y, v.strength) for v in vortices])
        
    return history

# Test case
def test_simulate_vortices():
    n_steps = 10
    dt = 0.1
    box_size = (5, 5)
    
    history = simulate_vortices(n_steps, dt, box_size)
    
    # Print the positions of the vortices at the last time step for validation
    last_positions = history[-1]
    for i, vortex in enumerate(last_positions):
        print(f"Vortex {i+1} final position: ({vortex.x}, {vortex.y})")

test_simulate_vortices()


def compute_velocity_field(vortices, grid_x, grid_y):
    """
    Compute the velocity field induced by the vortices.

    Parameters:
    vortices (list of Vortex): List of vortex objects
    grid_x (2D numpy array): x-coordinates of the grid points
    grid_y (2D numpy array): y-coordinates of the grid points

    Returns:
    2D numpy arrays: Velocity components (Vx, Vy) at each grid point
    """
    Vx = np.zeros_like(grid_x)
    Vy = np.zeros_like(grid_y)
    
    for vortex in vortices:
        dx = grid_x - vortex.x
        dy = grid_y - vortex.y
        r_squared = dx**2 + dy**2

        # Compute the velocity components induced by the vortex at grid points
        Vx += vortex.strength / (2 * np.pi) * dy / r_squared
        Vy += -vortex.strength / (2 * np.pi) * dx / r_squared
        
    return Vx, Vy

def plot_velocity_field_and_vortices(vortices, box_size, n_grid=20):
    """
    Plot the velocity field and vortex positions.

    Parameters:
    vortices (list of Vortex): List of vortex objects
    box_size (tuple): Tuple specifying the dimensions of the bounding box (x_max, y_max)
    n_grid (int): Number of grid points along each dimension
    """
    # Generate a grid
    x = np.linspace(0, box_size[0], n_grid)
    y = np.linspace(0, box_size[1], n_grid)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Compute the velocity field at grid points
    Vx, Vy = compute_velocity_field(vortices, grid_x, grid_y)
    
    plt.figure(figsize=(10, 10))
    
    # Plot velocity field using quiver plot
    plt.quiver(grid_x, grid_y, Vx, Vy, scale=20, color='b', alpha=0.5, label='Velocity Field')
    
    # Plot vortex positions
    x_positions = [v.x for v in vortices]
    y_positions = [v.y for v in vortices]
    plt.scatter(x_positions, y_positions, c='r', label='Vortex Centers')
    
    plt.xlim(0, box_size[0])
    plt.ylim(0, box_size[1])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Velocity Field and Vortex Positions')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run a simulation for 1 time step and plot the velocity field and vortex positions
n_steps = 1
dt = 0.1
box_size = (5, 5)
history = simulate_vortices(n_steps, dt, box_size)

# Plot the velocity field and vortex positions at the first time step
plot_velocity_field_and_vortices(history[0], box_size)

