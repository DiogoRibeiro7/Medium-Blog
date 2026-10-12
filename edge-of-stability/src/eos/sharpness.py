"""Top Hessian eigenvalue ("sharpness") and a rescaling-invariant variant.

Both are computed matrix-free with Lanczos (scipy's eigsh, ARPACK) on exact
Hessian-vector products, so the Hessian is never formed.

Adaptive sharpness is the top eigenvalue of  D H D  with D = diag(|theta|).
If theta' = A theta for a positive diagonal A that leaves the network function
unchanged (for example the layer rescaling of a ReLU network), then
H' = A^{-1} H A^{-1} and D' = A D, so D' H' D' = D H D: the quantity is exactly
invariant. This is the element-wise normalisation used by Kwon et al. (2021)
and studied by Andriushchenko et al. (2023).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .mlp import Problem


def _top_eig(matvec, n, v0=None, tol=1e-6):
    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    vals, vecs = eigsh(op, k=1, which="LA", tol=tol, v0=v0, maxiter=5000)
    return float(vals[0]), vecs[:, 0]


def sharpness(prob: Problem, flat, v0=None, tol=1e-6):
    """Largest algebraic eigenvalue of the training-loss Hessian at `flat`.

    Returns (eigenvalue, eigenvector); pass the eigenvector back as `v0` to
    warm-start the next call along a trajectory.
    """
    flat = jnp.asarray(flat)

    def mv(v):
        return np.asarray(prob.hvp(flat, jnp.asarray(v, dtype=jnp.float64)))

    return _top_eig(mv, prob.n_params, v0=v0, tol=tol)


def adaptive_sharpness(prob: Problem, flat, v0=None, tol=1e-6):
    """Largest eigenvalue of diag(|theta|) H diag(|theta|)."""
    flat = jnp.asarray(flat)
    scale = jnp.abs(flat)

    def mv(v):
        v = jnp.asarray(v, dtype=jnp.float64)
        return np.asarray(scale * prob.hvp(flat, scale * v))

    return _top_eig(mv, prob.n_params, v0=v0, tol=tol)
