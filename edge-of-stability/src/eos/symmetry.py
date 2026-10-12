"""Layer rescaling symmetry of ReLU networks and the imbalance it conserves.

For a ReLU network, multiplying (W_l, b_l) by alpha > 0 and W_{l+1} by 1/alpha
leaves the function unchanged (Dinh et al., 2017). Differentiating
L(alpha * theta_l, W_{l+1} / alpha) at alpha = 1 gives the Euler identity

    <theta_l, grad_{theta_l} L> = <W_{l+1}, grad_{W_{l+1}} L>,

where theta_l = (W_l, b_l). Define the layer imbalance

    Q_l = ||W_l||^2 + ||b_l||^2 - ||W_{l+1}||^2.

Gradient flow conserves Q_l (Du, Hu and Lee, 2018). One gradient-descent step
theta <- theta - eta * g changes it by exactly

    Q_l' - Q_l = eta^2 * ( ||g_{W_l}||^2 + ||g_{b_l}||^2 - ||g_{W_{l+1}}||^2 ),

because the first-order terms cancel by the Euler identity. For the scalar model
f = u v this is the identity D' = D (1 - eta^2 r^2).
"""

from __future__ import annotations

import jax.numpy as jnp


def rescale(params, layer: int, alpha: float):
    """Function-preserving rescaling between `layer` and `layer + 1` (ReLU only)."""
    out = [(w, b) for w, b in params]
    w, b = out[layer]
    out[layer] = (w * alpha, b * alpha)
    w2, b2 = out[layer + 1]
    out[layer + 1] = (w2 / alpha, b2)
    return out


def layer_imbalance(params, layer: int) -> float:
    w, b = params[layer]
    w2, _ = params[layer + 1]
    return float(jnp.sum(w * w) + jnp.sum(b * b) - jnp.sum(w2 * w2))


def imbalance_increment(grads, layer: int, eta: float) -> float:
    """Exact change of Q_layer produced by one gradient-descent step."""
    gw, gb = grads[layer]
    gw2, _ = grads[layer + 1]
    return float(eta**2 * (jnp.sum(gw * gw) + jnp.sum(gb * gb) - jnp.sum(gw2 * gw2)))
