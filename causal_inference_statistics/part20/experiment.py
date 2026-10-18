"""Causal decision-making under limited capacity and heterogeneous treatment effects.

Reproduces all reported estimates with independent train/test trials.
Python >=3.12; NumPy only. No external data required.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

SEED: int = 20261018
REPLICATIONS: int = 500
N_TRAIN: int = 1200
N_TEST: int = 3000
CAPACITY: float = 0.2

def sample(rng: np.random.Generator, n: int) -> tuple[NDArray[np.float64], ...]:
    """Generate randomized data with risk and treatment benefit negatively aligned."""
    if not isinstance(n, int) or n < 20:
        raise ValueError('n must be integer >=20')
    x = rng.normal(size=n)
    a = rng.binomial(1, .5, size=n).astype(float)
    risk = 3.0 - 1.2*x
    benefit = .5 + 1.5*x
    y = risk - a*benefit + rng.normal(size=n)
    return x, a, y, risk, benefit

def fit_benefit(x: NDArray[np.float64], a: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate linear conditional reduction by randomized treatment."""
    design = np.column_stack((np.ones(x.size), x, a, a*x))
    return np.linalg.lstsq(design, y, rcond=None)[0][2:4] * -1

def mean_benefit(benefit: NDArray[np.float64], rank: NDArray[np.float64]) -> float:
    """Evaluate true per-treated benefit of a top-capacity policy."""
    if benefit.shape != rank.shape:
        raise ValueError('benefit and ranking must have equal shape')
    k = int(round(CAPACITY * len(benefit)))
    chosen = np.argsort(rank, kind='stable')[-k:]
    return float(benefit[chosen].mean())

def main() -> None:
    """Write mean policy effects, variability, and relative policy regret."""
    rng = np.random.default_rng(SEED)
    values: dict[str, list[float]] = {k: [] for k in ('random', 'high_risk', 'estimated_benefit', 'oracle')}
    ate_est: list[float] = []
    for _ in range(REPLICATIONS):
        x, a, y, _, _ = sample(rng, N_TRAIN)
        beta = fit_benefit(x, a, y)
        ate_est.append(float((-y[a==1].mean() + y[a==0].mean())))
        xt, _, _, risk, benefit = sample(rng, N_TEST)
        scores = {'random': rng.random(N_TEST), 'high_risk': risk,
                  'estimated_benefit': beta[0] + beta[1]*xt, 'oracle': benefit}
        for key, rank in scores.items():
            values[key].append(mean_benefit(benefit, rank))
    rows=[]
    oracle = np.asarray(values['oracle'])
    for name, observations in values.items():
        arr=np.asarray(observations)
        rows.append(dict(method=name,replications=REPLICATIONS,n_train=N_TRAIN,n_test=N_TEST,
                         capacity=CAPACITY,true_ate=.5,mean_benefit=float(arr.mean()),
                         sd=float(arr.std(ddof=1)),mean_regret=float(np.mean(oracle-arr)),
                         mean_population_gain=float(CAPACITY*arr.mean())))
    with Path(__file__).with_name('results.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:print(row)
    print('naive_trial_ate_mean',np.mean(ate_est))

if __name__ == '__main__':main()
