"""Fully connected networks with a flat parameter vector, exact gradients and
exact Hessian-vector products (forward-over-reverse automatic differentiation).

The loss is the mean squared error used by Cohen et al. (2021):
    L(theta) = (1 / 2n) * sum_i || f(x_i; theta) - y_i ||^2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

ACTIVATIONS: dict[str, Callable] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
}


def init_params(seed: int, sizes: list[int], gain: float = 1.0):
    """Gaussian weights with variance gain / fan_in, zero biases."""
    key = jax.random.PRNGKey(seed)
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (fan_in, fan_out), dtype=jnp.float64) * jnp.sqrt(gain / fan_in)
        params.append((w, jnp.zeros(fan_out, dtype=jnp.float64)))
    return params


def forward(params, x, activation: str):
    act = ACTIVATIONS[activation]
    h = x
    for w, b in params[:-1]:
        h = act(h @ w + b)
    w, b = params[-1]
    return h @ w + b


@dataclass
class Problem:
    """A network + dataset pair exposing everything the experiments need as
    functions of a flat float64 parameter vector."""

    sizes: list[int]
    activation: str
    x: jnp.ndarray
    y: jnp.ndarray
    unravel: Callable
    n_params: int

    @classmethod
    def build(cls, sizes, activation, x, y, template_params):
        flat, unravel = ravel_pytree(template_params)
        prob = cls(
            sizes=list(sizes),
            activation=activation,
            x=jnp.asarray(x),
            y=jnp.asarray(y),
            unravel=unravel,
            n_params=int(flat.size),
        )
        prob._compile()
        return prob

    # ------------------------------------------------------------------ #
    def _compile(self):
        act, x, y, unravel = self.activation, self.x, self.y, self.unravel

        def loss(flat):
            out = forward(unravel(flat), x, act)
            return 0.5 * jnp.mean(jnp.sum((out - y) ** 2, axis=1))

        self.loss = jax.jit(loss)
        self.value_and_grad = jax.jit(jax.value_and_grad(loss))
        self.grad = jax.jit(jax.grad(loss))
        self.hvp = jax.jit(lambda flat, vec: jax.jvp(jax.grad(loss), (flat,), (vec,))[1])
        self.predict = jax.jit(lambda flat, xs: forward(unravel(flat), xs, act))

    def flatten(self, params) -> jnp.ndarray:
        return ravel_pytree(params)[0]


def accuracy(prob: Problem, flat, x, labels) -> float:
    pred = np.asarray(prob.predict(flat, jnp.asarray(x))).argmax(axis=1)
    return float(np.mean(pred == labels))


def mse(prob: Problem, flat, x, y) -> float:
    out = np.asarray(prob.predict(flat, jnp.asarray(x)))
    return float(0.5 * np.mean(np.sum((out - y) ** 2, axis=1)))
