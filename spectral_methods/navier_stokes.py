from scipy.fft import fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2


def fft_derivative(f, k):
    """
    Compute the Fourier derivative of a 2D field.

    Parameters:
        f (numpy.ndarray): The function values on a 2D grid.
        k (numpy.ndarray): The wave numbers.

    Returns:
        numpy.ndarray: The Fourier derivative.

    Example:
        >>> f = np.random.rand(5, 5)
        >>> k = np.array([1, 2, 3, 4, 5])
        >>> fft_derivative(f, k)
    """
    f_hat = fft2(f)
    return np.real(ifft2(1j * k * f_hat))


def fft_derivative_2D(f, kx, ky):
    """
    Compute the Fourier derivatives along x and y of a 2D field.

    Parameters:
        f (numpy.ndarray): The function values on a 2D grid.
        kx, ky (numpy.ndarray): The wave numbers along x and y.

    Returns:
        dfdx, dfdy (numpy.ndarray): The Fourier derivatives along x and y.

    Example:
        >>> f = np.random.rand(5, 5)
        >>> kx = np.array([1, 2, 3, 4, 5])
        >>> ky = np.array([1, 2, 3, 4, 5])
        >>> fft_derivative_2D(f, kx, ky)
    """
    f_hat = fft2(f)
    kx, ky = np.meshgrid(kx, ky)  # Create 2D grid of wave numbers
    dfdx = np.real(ifft2(1j * kx * f_hat))
    dfdy = np.real(ifft2(1j * ky * f_hat))
    return dfdx, dfdy



def navier_stokes_2D_RK4(u, v, p, rho, nu, dx, dy, dt, steps):
    """
    Solve 2D Navier-Stokes equations using spectral methods and 4th order Runge-Kutta.

    Parameters:
        u, v, p (numpy.ndarray): Initial velocity and pressure fields.
        rho (float): Density of the fluid.
        nu (float): Viscosity of the fluid.
        dx, dy (float): Grid spacing in x and y directions.
        dt (float): Time step.
        steps (int): Number of time steps to solve for.

    Returns:
        u, v, p (numpy.ndarray): Velocity and pressure fields at final time.

    Example:
        >>> u = np.random.rand(5, 5)
        >>> v = np.random.rand(5, 5)
        >>> p = np.random.rand(5, 5)
        >>> navier_stokes_2D_RK4(u, v, p, 1.0, 0.1, 0.01, 0.01, 0.001, 10)
    """
    # Wave numbers for Fourier derivative
    N, M = u.shape
    kx = np.fft.fftfreq(N, dx)
    ky = np.fft.fftfreq(M, dy)
    
    for t in range(steps):
        # Compute derivatives using FFT
        ux, uy = fft_derivative_2D(u, kx, ky)
        vx, vy = fft_derivative_2D(v, kx, ky)
        px, py = fft_derivative_2D(p, kx, ky)
        # Runge-Kutta 4th order for u
        k1_u = -u*ux - v*uy + (-px/rho) + nu*(ux + vy)
        k2_u = -u*(ux + k1_u*dt/2) - v*(uy + k1_u*dt/2) + \
            (-px/rho) + nu*(ux + vy)
        k3_u = -u*(ux + k2_u*dt/2) - v*(uy + k2_u*dt/2) + \
            (-px/rho) + nu*(ux + vy)
        k4_u = -u*(ux + k3_u*dt) - v*(uy + k3_u*dt) + (-px/rho) + nu*(ux + vy)
        u = u + dt/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u)

        # Runge-Kutta 4th order for v
        k1_v = -u*vx - v*vy + (-py/rho) + nu*(vx + vy)
        k2_v = -u*(vx + k1_v*dt/2) - v*(vy + k1_v*dt/2) + \
            (-py/rho) + nu*(vx + vy)
        k3_v = -u*(vx + k2_v*dt/2) - v*(vy + k2_v*dt/2) + \
            (-py/rho) + nu*(vx + vy)
        k4_v = -u*(vx + k3_v*dt) - v*(vy + k3_v*dt) + (-py/rho) + nu*(vx + vy)
        v = v + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return u, v, p


def test_fft_derivative():
    """
    Test the fft_derivative function.
    """
    # Test 1: Constant function, derivative should be zero
    f_const = np.ones((5, 5))
    k = np.array([1, 2, 3, 4, 5])
    assert np.allclose(fft_derivative(f_const, k),
                       np.zeros((5, 5)), atol=1e-10)

    # Test 2: Linear function in x, derivative should be constant
    f_linear_x = np.outer(np.ones(5), np.linspace(0, 4, 5))
    assert np.allclose(fft_derivative(f_linear_x, k),
                       np.ones((5, 5)), atol=1e-10)

    # Test 3: Sine function, derivative should be cosine
    f_sine = np.sin(np.outer(np.linspace(0, 2*np.pi, 5), np.ones(5)))
    f_cosine = np.cos(np.outer(np.linspace(0, 2*np.pi, 5), np.ones(5)))
    assert np.allclose(fft_derivative(f_sine, k), f_cosine, atol=1e-10)

# test_fft_derivative()


def heat_equation_2D(u, T, rho, cp, k, kx, ky, dt, steps):
    """
    Solve the 2D heat equation coupled with Navier-Stokes equations.

    Parameters:
        u (numpy.ndarray): Initial velocity field along x.
        T (numpy.ndarray): Initial temperature field.
        rho (float): Density of the fluid.
        cp (float): Specific heat at constant pressure.
        k (float): Thermal conductivity of the fluid.
        kx, ky (numpy.ndarray): Wave numbers along x and y directions.
        dt (float): Time step.
        steps (int): Number of time steps to solve for.

    Returns:
        T (numpy.ndarray): Temperature field at the final time step.

    Example:
        >>> u = np.random.rand(5, 5)
        >>> T = np.random.rand(5, 5)
        >>> kx = np.array([1, 2, 3, 4, 5])
        >>> ky = np.array([1, 2, 3, 4, 5])
        >>> heat_equation_2D(u, T, 1.0, 1000, 0.1, kx, ky, 0.001, 10)
    """
    for t in range(steps):
        # Compute Fourier derivatives
        dfdx, _ = fft_derivative_2D(u, kx, ky)
        dTdx, dTdy = fft_derivative_2D(T, kx, ky)
        
        # Compute Laplacian of T in spectral space
        laplacian_T = fft_derivative_2D(dTdx, kx, ky)[0] + fft_derivative_2D(dTdy, kx, ky)[1]
        
        # Time-stepping for heat equation
        dT_dt = (k / (rho * cp)) * laplacian_T - u * dTdx
        T += dt * dT_dt

    return T

# Initialize fields for testing
N = 5
u = np.ones((N, N))
T_initial = np.random.rand(N, N)

# Wave numbers for Fourier derivative
kx = np.fft.fftfreq(N, 0.01)
ky = np.fft.fftfreq(N, 0.01)

# Parameters
rho = 1.0  # Density
cp = 1000  # Specific heat at constant pressure
k = 0.1  # Thermal conductivity
dt = 0.001  # Time step
steps = 10  # Number of time steps

# Solve heat equation coupled with fluid flow
T_final = heat_equation_2D(u, T_initial, rho, cp, k, kx, ky, dt, steps)
print(T_final)