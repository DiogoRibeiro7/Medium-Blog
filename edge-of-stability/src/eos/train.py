"""Full-batch gradient descent (optionally with heavy-ball momentum) that records
the training loss at every step and the sharpness at regular intervals."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .mlp import Problem
from .sharpness import adaptive_sharpness, sharpness


@dataclass
class Trace:
    eta: float
    momentum: float
    loss: np.ndarray  # loss before each step, length = steps + 1 (last entry: final loss)
    eval_steps: np.ndarray
    sharpness: np.ndarray
    adaptive_sharpness: np.ndarray
    params: np.ndarray  # final flat parameters
    diverged: bool = False
    extra: dict = field(default_factory=dict)


def _make_chunk(prob: Problem, eta: float, beta: float, n: int):
    vg = prob.value_and_grad

    def body(carry, _):
        theta, mom = carry
        loss, g = vg(theta)
        mom = beta * mom + g
        theta = theta - eta * mom
        return (theta, mom), loss

    @jax.jit
    def chunk(theta, mom):
        (theta, mom), losses = jax.lax.scan(body, (theta, mom), None, length=n)
        return theta, mom, losses

    return chunk


def run_gd(
    prob: Problem,
    theta0,
    eta: float,
    steps: int,
    eval_every: int,
    momentum: float = 0.0,
    with_adaptive: bool = False,
    sharpness_tol: float = 1e-6,
    blowup: float = 1e6,
) -> Trace:
    """Heavy-ball update m <- beta m + g, theta <- theta - eta m (beta = 0 is plain GD)."""
    if steps % eval_every:
        raise ValueError("steps must be a multiple of eval_every")
    chunk = _make_chunk(prob, eta, momentum, eval_every)
    theta = jnp.asarray(theta0)
    mom = jnp.zeros_like(theta)
    losses, ev_steps, sharp, asharp = [], [], [], []
    v0 = a0 = None
    diverged = False
    for k in range(steps // eval_every):
        s, v0 = sharpness(prob, theta, v0=v0, tol=sharpness_tol)
        ev_steps.append(k * eval_every)
        sharp.append(s)
        if with_adaptive:
            a, a0 = adaptive_sharpness(prob, theta, v0=a0, tol=sharpness_tol)
            asharp.append(a)
        theta, mom, ls = chunk(theta, mom)
        ls = np.asarray(ls)
        losses.append(ls)
        if not np.all(np.isfinite(ls)) or ls.max() > blowup:
            diverged = True
            break
    final_loss = float(prob.loss(theta))
    if not diverged and np.isfinite(final_loss):
        s, _ = sharpness(prob, theta, v0=v0, tol=sharpness_tol)
        ev_steps.append(steps)
        sharp.append(s)
        if with_adaptive:
            a, _ = adaptive_sharpness(prob, theta, v0=a0, tol=sharpness_tol)
            asharp.append(a)
    return Trace(
        eta=eta,
        momentum=momentum,
        loss=np.concatenate(losses + [np.array([final_loss])]),
        eval_steps=np.array(ev_steps),
        sharpness=np.array(sharp),
        adaptive_sharpness=np.array(asharp),
        params=np.asarray(theta),
        diverged=diverged,
    )
