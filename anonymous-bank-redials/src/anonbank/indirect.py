"""Indirect inference for the redial probability.

For a chosen statistic, the binding function maps a redial probability p to
the statistic's value on a long simulated record. It is computed on a grid
of p with the same random numbers at every grid point, smoothed by a
quadratic, made monotone and inverted at the observed value. Sampling error
is carried through by inverting bootstrap replicates of the observed
statistic through the same binding function.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .sim import Simulator
from .stats import Statistics

GRID = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)


@dataclass
class Binding:
    statistic: str
    grid: np.ndarray
    values: np.ndarray

    def curve(self, n: int = 801) -> tuple[np.ndarray, np.ndarray]:
        fine = np.linspace(self.grid[0], self.grid[-1], n)
        fit = np.polyval(np.polyfit(self.grid, self.values, 2), fine)
        return fine, np.maximum.accumulate(fit)

    def invert(self, value) -> np.ndarray:
        fine, fit = self.curve()
        return np.interp(value, fit, fine)


def binding_functions(sim: Simulator, statistics: tuple[str, ...],
                      seed: int, n_days: int,
                      grid: tuple[float, ...] = GRID) -> dict[str, Binding]:
    vals = {s: [] for s in statistics}
    for p in grid:
        off, ab, dow = replace(sim, p=p).days(np.random.default_rng(seed),
                                              n_days)
        st = Statistics(off, ab, dow)
        for s in statistics:
            vals[s].append(st.get(s))
    return {s: Binding(s, np.array(grid), np.array(v))
            for s, v in vals.items()}


def estimate(stats: Statistics, binding: Binding,
             rng: np.random.Generator, reps: int = 1000) -> dict[str, float]:
    obs = stats.get(binding.statistic)
    boot = stats.bootstrap(binding.statistic, rng, reps)
    p_boot = binding.invert(boot)
    return {"statistic": obs,
            "statistic_sd": float(boot.std()),
            "p_hat": float(binding.invert(obs)),
            "p_lo": float(np.quantile(p_boot, 0.05)),
            "p_hi": float(np.quantile(p_boot, 0.95))}


# --------------------------------------------------------------------------
# Joint estimation of redials and clustering


JOINT_STATISTICS = ("lag", "lead", "controlled", "acf1", "dispersion")


def _poly_terms(theta: np.ndarray) -> np.ndarray:
    """Full quadratic in (p, rho, sigma), with a constant."""
    p, r, s = theta[..., 0], theta[..., 1], theta[..., 2]
    return np.stack([np.ones_like(p), p, r, s, p * p, r * r, s * s, p * r,
                     p * s, r * s], axis=-1)


@dataclass
class JointEstimator:
    """Indirect inference for (p, rho, sigma) from several statistics.

    Each statistic is simulated on a grid of the three parameters and a
    quadratic response surface is fitted to it; the estimate minimises the
    weighted squared distance between the surfaces and the observed
    statistics, with weights equal to the inverse bootstrap variances.
    """

    statistics: tuple[str, ...]
    grid: np.ndarray               # (n, 3)
    values: np.ndarray             # (n, k)
    coef: np.ndarray               # (10, k)
    r2: np.ndarray                 # (k,)

    @classmethod
    def fit(cls, sim: Simulator, seed: int, n_days: int,
            ps=(0.0, 0.1, 0.2, 0.3, 0.45, 0.6),
            rhos=(0.0, 0.3, 0.5, 0.7, 0.85),
            sigmas=(0.03, 0.08, 0.13, 0.2, 0.3),
            statistics: tuple[str, ...] = JOINT_STATISTICS
            ) -> "JointEstimator":
        from .sim import Clustering
        grid, vals = [], []
        for p in ps:
            for rho in rhos:
                for sg in sigmas:
                    s = replace(sim, p=p, clustering=Clustering(rho, sg))
                    off, ab, dow = s.days(np.random.default_rng(seed),
                                          n_days)
                    grid.append((p, rho, sg))
                    vals.append(Statistics(off, ab, dow).vector(statistics))
        grid, vals = np.array(grid), np.array(vals)
        X = _poly_terms(grid)
        coef, *_ = np.linalg.lstsq(X, vals, rcond=None)
        resid = vals - X @ coef
        r2 = 1 - resid.var(axis=0) / vals.var(axis=0)
        return cls(statistics, grid, vals, coef, r2)

    def predict(self, theta: np.ndarray) -> np.ndarray:
        return _poly_terms(np.asarray(theta, float)) @ self.coef

    def solve(self, observed: np.ndarray, weights: np.ndarray) -> np.ndarray:
        from scipy.optimize import minimize
        lo = self.grid.min(axis=0)
        hi = self.grid.max(axis=0)

        def loss(th):
            return float(np.sum(weights * (self.predict(th) - observed) ** 2))
        starts = [lo + f * (hi - lo) for f in (0.2, 0.5, 0.8)]
        best = min((minimize(loss, x0, method="L-BFGS-B",
                             bounds=list(zip(lo, hi))) for x0 in starts),
                   key=lambda res: res.fun)
        return best.x

    def point(self, stats: Statistics, weights: np.ndarray) -> np.ndarray:
        """Point estimate with given weights (no bootstrap)."""
        return self.solve(stats.vector(self.statistics), weights)

    def estimate(self, stats: Statistics, rng: np.random.Generator,
                 reps: int = 300) -> dict:
        obs = stats.vector(self.statistics)
        idx = rng.integers(0, stats.n_days, size=(reps, stats.n_days))
        boot = np.array([stats.vector(self.statistics, i) for i in idx])
        w = 1.0 / boot.var(axis=0)
        theta = self.solve(obs, w)
        thetas = np.array([self.solve(b, w) for b in boot])
        return {"p_hat": float(theta[0]), "rho_hat": float(theta[1]),
                "sigma_hat": float(theta[2]),
                "p_lo": float(np.quantile(thetas[:, 0], 0.05)),
                "p_hi": float(np.quantile(thetas[:, 0], 0.95)),
                "rho_lo": float(np.quantile(thetas[:, 1], 0.05)),
                "rho_hi": float(np.quantile(thetas[:, 1], 0.95)),
                "observed": dict(zip(self.statistics, obs)),
                "fitted": dict(zip(self.statistics, self.predict(theta))),
                "weights": dict(zip(self.statistics, w))}
