"""Diffusion Monte Carlo for a nodeless ground-state energy, with no trial function.

Walkers diffuse with the exact free-particle propagator, x' = x + sqrt(tau) * N(0, I),
and are reweighted with the symmetric split exp(-tau (V(x) + V(x')) / 2), so the
time-step error in the energy is O(tau^2). The population is kept fixed by systematic
resampling at every step.

Without a trial function the walker density converges to psi0 itself (not psi0^2),
and because H applied to the constant function 1 is just V,

    E0 = integral psi0 V / integral psi0,

so the average potential over the walkers is an estimator of E0. The growth estimator
-log(mean weight) / tau is recorded as well. Standard errors come from block averages
over blocks of at least 5 units of imaginary time.

Nothing about the answer is used: walkers start from N(0, I) and no trial wavefunction
guides them. This is the plainest DMC there is; importance sampling with a good trial
function would reduce both the variance and the time-step error.
"""

from __future__ import annotations

import time

import numpy as np

from .problems import Problem


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = w.size
    c = np.cumsum(w)
    c /= c[-1]
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(c, u), n - 1)


def _block_mean(series: np.ndarray, min_block: int, max_blocks: int = 50) -> tuple[float, float]:
    """Mean and standard error from block averages, with blocks of at least ``min_block``
    steps so that consecutive blocks are close to independent."""
    n_blocks = int(min(max_blocks, max(8, series.size // max(min_block, 1))))
    m = series.size // n_blocks
    blocks = series[: m * n_blocks].reshape(n_blocks, m).mean(axis=1)
    return float(blocks.mean()), float(blocks.std(ddof=1) / np.sqrt(n_blocks))


def run_dmc(problem: Problem, n_walkers: int, tau: float, n_steps: int,
            n_equil: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    t_cpu, t_wall = time.process_time(), time.perf_counter()
    d = problem.d
    X = rng.normal(size=(n_walkers, d))
    V = problem.potential(X)
    sq = np.sqrt(tau)
    e_pot = np.empty(n_steps)
    e_growth = np.empty(n_steps)
    for step in range(n_equil + n_steps):
        Xn = X + sq * rng.normal(size=X.shape)
        Vn = problem.potential(Xn)
        logw = -0.5 * tau * (V + Vn)
        top = logw.max()
        w = np.exp(logw - top)
        if step >= n_equil:
            i = step - n_equil
            e_growth[i] = -(top + np.log(w.mean())) / tau
        idx = _systematic_resample(w, rng)
        X, V = Xn[idx], Vn[idx]
        if step >= n_equil:
            e_pot[i] = V.mean()
    cpu = time.process_time() - t_cpu
    wall = time.perf_counter() - t_wall
    # walker correlations decay on an imaginary-time scale of order 1 / gap ~ 1, so the
    # blocks span at least 5 units of imaginary time
    block = int(np.ceil(5.0 / tau))
    Ep, sep = _block_mean(e_pot, block)
    Eg, seg = _block_mean(e_growth, block)
    E0 = problem.E0
    return {
        "problem": problem.name, "d": d, "g": problem.g, "method": "dmc",
        "n_walkers": n_walkers, "tau": tau, "n_steps": n_steps, "n_equil": n_equil,
        "seed": seed,
        "E_potential": Ep, "se_potential": sep,
        "E_growth": Eg, "se_growth": seg,
        "rel_energy_error": abs(Ep - E0) / E0, "rel_se": sep / E0,
        "E0": E0, "cpu_s": cpu, "wall_s": wall,
        "peak_bytes": int(8 * n_walkers * d * 4),
    }

