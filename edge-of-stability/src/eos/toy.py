"""Scalar two-layer factorisation f(u, v) = u * v with loss L(u, v) = (u v - 1)^2 / 2.

Everything about this model can be written down by hand:

* the minimisers form the hyperbola u v = 1;
* at a minimiser the Hessian has eigenvalues 0 and u^2 + v^2;
* gradient flow conserves the imbalance D = u^2 - v^2, so its endpoint is fixed
  by the initialisation;
* gradient descent with step eta changes the imbalance by an exact factor,
  D_{t+1} = D_t (1 - eta^2 r_t^2), where r_t = u_t v_t - 1 is the residual.

On the balanced line u = v = s the gradient-descent map reduces to the cubic map
s -> s (1 + eta - eta s^2), which is gradient descent on (s^2 - 1)^2 / 4.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def residual(u, v):
    return u * v - 1.0


def loss(u, v):
    return 0.5 * residual(u, v) ** 2


def grad(u, v):
    r = residual(u, v)
    return r * v, r * u


def hessian(u: float, v: float) -> np.ndarray:
    r = residual(u, v)
    return np.array([[v * v, u * v + r], [u * v + r, u * u]])


def sharpness(u: float, v: float) -> float:
    """Largest Hessian eigenvalue at (u, v)."""
    return float(np.linalg.eigvalsh(hessian(u, v))[-1])


def imbalance(u, v):
    return u * u - v * v


def minimum_sharpness(imb):
    """Sharpness of the minimiser (u v = 1) whose imbalance is `imb`: sqrt(imb^2 + 4)."""
    return np.sqrt(np.asarray(imb) ** 2 + 4.0)


def flow_endpoint(u0: float, v0: float) -> tuple[float, float]:
    """Limit of gradient flow from a start in the open positive quadrant.

    The flow conserves u^2 - v^2, so the limit is the point of u v = 1 with the
    same imbalance.
    """
    if u0 <= 0 or v0 <= 0:
        raise ValueError("flow_endpoint is implemented for the positive quadrant only")
    d = imbalance(u0, v0)
    u = np.sqrt((d + np.sqrt(d * d + 4.0)) / 2.0)
    return float(u), float(1.0 / u)


def stable_imbalance_bound(eta: float) -> float:
    """Largest |D| of a minimiser that gradient descent with step eta can converge to.

    Linear stability needs eta * sqrt(D^2 + 4) < 2, i.e. |D| < 2 sqrt(1/eta^2 - 1).
    No minimiser is stable when eta >= 1.
    """
    if eta >= 1.0:
        return float("nan")
    return float(2.0 * np.sqrt(1.0 / eta**2 - 1.0))


def gd_step(u, v, eta):
    r = residual(u, v)
    return u - eta * r * v, v - eta * r * u


def gd_trajectory(u0: float, v0: float, eta: float, steps: int, blowup: float = 1e8):
    """Iterates of gradient descent. Returns (u, v) arrays of length steps + 1.

    If the iterates leave a ball of radius `blowup`, the rest of the arrays is NaN.
    """
    u = np.full(steps + 1, np.nan)
    v = np.full(steps + 1, np.nan)
    u[0], v[0] = u0, v0
    for t in range(steps):
        u[t + 1], v[t + 1] = gd_step(u[t], v[t], eta)
        if not (abs(u[t + 1]) < blowup and abs(v[t + 1]) < blowup):
            u[t + 1 :] = np.nan
            v[t + 1 :] = np.nan
            break
    return u, v


def gradient_flow(u0: float, v0: float, t_max: float, n: int = 400):
    """Gradient flow du/dt = -dL/du, dv/dt = -dL/dv, integrated with tight tolerances."""

    def rhs(_, z):
        gu, gv = grad(z[0], z[1])
        return [-gu, -gv]

    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(rhs, (0.0, t_max), [u0, v0], t_eval=t_eval, rtol=1e-11, atol=1e-13, method="DOP853")
    return sol.t, sol.y[0], sol.y[1]


def balanced_map(s, eta):
    """Gradient descent restricted to u = v = s."""
    return s * (1.0 + eta - eta * s * s)


def balanced_map_derivative(s, eta):
    return 1.0 + eta - 3.0 * eta * s * s


def orbit(eta: float, s0: float, burn: int, keep: int, blowup: float = 1e6) -> np.ndarray:
    """Points of the balanced map after a burn-in. Empty array if the orbit escapes."""
    s = s0
    for _ in range(burn):
        s = balanced_map(s, eta)
        if abs(s) > blowup:
            return np.array([])
    out = np.empty(keep)
    for i in range(keep):
        s = balanced_map(s, eta)
        if abs(s) > blowup:
            return np.array([])
        out[i] = s
    return out


def lyapunov_exponent(eta: float, s0: float, burn: int, n: int) -> float:
    """Mean of log|g'(s_t)| along the orbit of the balanced map (NaN if it escapes)."""
    pts = orbit(eta, s0, burn, n)
    if pts.size == 0:
        return float("nan")
    d = np.abs(balanced_map_derivative(pts, eta))
    return float(np.mean(np.log(np.maximum(d, 1e-300))))


def period_of(points: np.ndarray, tol: float = 1e-8, max_period: int = 64) -> int:
    """Smallest p <= max_period with s_{t+p} == s_t over the stored orbit, else 0."""
    if points.size == 0:
        return -1
    for p in range(1, max_period + 1):
        if points.size > p and np.max(np.abs(points[p:] - points[:-p])) < tol:
            return p
    return 0
