"""Two ground-state problems in d dimensions with exact answers (units hbar = m = 1).

Both are written as H = -1/2 Laplacian + V(x) with a known positive ground state
psi0 = exp(-phi(x)) and energy E0 = d/2.

* ``oscillator``: phi = |x|^2 / 2, V = |x|^2 / 2. Separable.
* ``coupled``: phi = |x|^2 / 2 + kappa * sum_{i<j} x_i^2 x_j^2 with kappa = g / (d - 1).
  The potential is built backwards from psi0,

      V = E0 + (|grad phi|^2 - Laplacian phi) / 2,

  so that H psi0 = E0 psi0 exactly. psi0 is positive and normalisable, so it is the
  ground state. The quartic coupling term is not a sum of one-coordinate functions and
  is not radial, so the problem does not separate in Cartesian or hyperspherical
  coordinates. Scaling kappa by 1/(d - 1) keeps the coupling felt by each coordinate
  comparable across dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str          # "oscillator" or "coupled"
    d: int
    g: float = 1.0     # coupling strength for "coupled"

    def __post_init__(self):
        if self.name not in ("oscillator", "coupled"):
            raise ValueError(f"unknown problem {self.name!r}")
        if self.d < 1:
            raise ValueError("d must be >= 1")
        if self.name == "coupled" and self.d < 2:
            raise ValueError("the coupled problem needs d >= 2")

    @property
    def kappa(self) -> float:
        return 0.0 if self.name == "oscillator" else self.g / (self.d - 1)

    @property
    def E0(self) -> float:
        return self.d / 2.0

    @property
    def label(self) -> str:
        return f"{self.name}-d{self.d}"

    # --- formulas on points x of shape (..., d); xp is numpy or jax.numpy -----------

    def phi(self, x, xp=np):
        x2 = x * x
        s = xp.sum(x2, axis=-1)
        if self.kappa == 0.0:
            return 0.5 * s
        pair = 0.5 * (s * s - xp.sum(x2 * x2, axis=-1))
        return 0.5 * s + self.kappa * pair

    def potential(self, x, xp=np):
        x2 = x * x
        s = xp.sum(x2, axis=-1)
        if self.kappa == 0.0:
            return 0.5 * s
        k = self.kappa
        others = s[..., None] - x2                     # sum over j != k
        grad2 = xp.sum(x2 * (1.0 + 2.0 * k * others) ** 2, axis=-1)
        lap = self.d + 2.0 * k * (self.d - 1) * s
        return self.E0 + 0.5 * (grad2 - lap)

    # --- the same potential on a tensor-product grid, without forming an (N^d, d) array

    def potential_on_grid(self, nodes: np.ndarray) -> np.ndarray:
        """V at every node of the grid nodes x ... x nodes (d times), shape (N,)*d."""
        d, n = self.d, nodes.size
        x2 = nodes * nodes

        def along(k, values):
            shape = [1] * d
            shape[k] = n
            return values.reshape(shape)

        s = np.zeros((n,) * d)
        for k in range(d):
            s = s + along(k, x2)
        if self.kappa == 0.0:
            return 0.5 * s
        kap = self.kappa
        grad2 = np.zeros_like(s)
        for k in range(d):
            xk2 = along(k, x2)
            grad2 += xk2 * (1.0 + 2.0 * kap * (s - xk2)) ** 2
        lap = d + 2.0 * kap * (d - 1) * s
        return self.E0 + 0.5 * (grad2 - lap)

    def log_psi0_on_grid(self, nodes: np.ndarray) -> np.ndarray:
        d, n = self.d, nodes.size
        x2 = nodes * nodes
        s = np.zeros((n,) * d)
        q = np.zeros((n,) * d)
        for k in range(d):
            shape = [1] * d
            shape[k] = n
            s = s + x2.reshape(shape)
            q = q + (x2 * x2).reshape(shape)
        phi = 0.5 * s + self.kappa * 0.5 * (s * s - q)
        return -phi

    # --- exact samples from |psi0|^2 -----------------------------------------------

    def sample_psi0_squared(self, m: int, rng: np.random.Generator) -> np.ndarray:
        """Exact draws from |psi0|^2.

        |psi0|^2 = exp(-|x|^2) * exp(-2 kappa pair) with pair >= 0, so rejection
        sampling from N(0, I/2) with acceptance exp(-2 kappa pair) is exact.
        """
        sd = np.sqrt(0.5)
        if self.kappa == 0.0:
            return rng.normal(0.0, sd, size=(m, self.d))
        out, have = [], 0
        while have < m:
            x = rng.normal(0.0, sd, size=(max(4 * (m - have), 1024), self.d))
            x2 = x * x
            s = x2.sum(-1)
            pair = 0.5 * (s * s - (x2 * x2).sum(-1))
            keep = rng.random(x.shape[0]) < np.exp(-2.0 * self.kappa * pair)
            out.append(x[keep])
            have += int(keep.sum())
        return np.concatenate(out)[:m]
