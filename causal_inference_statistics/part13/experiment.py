"""Cross-fitted partially linear DML benchmark; Python 3.12+.

Requires numpy and scikit-learn. All features and results are simulated.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

SEED: int = 20261018
REPS: int = 20
N: int = 800
TAU: float = 1.0


def data(rng: np.random.Generator, nonlinear: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one partially linear outcome and endogenous treatment."""
    x = rng.normal(size=(N, 3))
    if nonlinear:
        m = 1.4 * np.sin(x[:, 0]) + 0.9 * (x[:, 1] ** 2 - 1) + 0.5 * x[:, 0] * x[:, 2]
        g = 1.2 * np.sin(x[:, 0]) + 1.3 * (x[:, 1] ** 2 - 1) + 0.8 * x[:, 0] * x[:, 2]
    else:
        m = 1.4 * x[:, 0] + 0.9 * x[:, 1] + 0.5 * x[:, 2]
        g = 1.2 * x[:, 0] + 1.3 * x[:, 1] + 0.8 * x[:, 2]
    d = m + rng.normal(size=N)
    y = TAU * d + g + rng.normal(size=N)
    return x, d, y


def dml(x: np.ndarray, d: np.ndarray, y: np.ndarray, *, flexible: bool, crossfit: bool, seed: int) -> float:
    """Partial-out DML: regress outcome residual on treatment residual."""
    folds = list(KFold(n_splits=3, shuffle=True, random_state=seed).split(x)) if crossfit else [(np.arange(N), np.arange(N))]
    yhat = np.zeros(N)
    dhat = np.zeros(N)
    for train, test in folds:
        def model() -> LinearRegression | HistGradientBoostingRegressor:
            return (HistGradientBoostingRegressor(max_iter=45, max_leaf_nodes=15, min_samples_leaf=30,
                                                   learning_rate=0.08, l2_regularization=2.0,
                                                   random_state=seed) if flexible else LinearRegression())
        gy = model().fit(x[train], y[train])
        gd = model().fit(x[train], d[train])
        yhat[test] = gy.predict(x[test])
        dhat[test] = gd.predict(x[test])
    residual_d = d - dhat
    return float(np.dot(residual_d, y-yhat) / np.dot(residual_d, residual_d))


def main() -> None:
    """Print and persist Monte Carlo point-estimation comparisons."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, str | int | float]] = []
    for nonlinear in (False, True):
        records: dict[str, list[float]] = {k: [] for k in ('naive', 'linear_cf', 'boost_cf', 'boost_insample')}
        for iteration in range(REPS):
            x, d, y = data(rng, nonlinear)
            records['naive'].append(float(np.cov(d,y)[0,1]/np.var(d,ddof=1)))
            records['linear_cf'].append(dml(x,d,y,flexible=False,crossfit=True,seed=iteration))
            records['boost_cf'].append(dml(x,d,y,flexible=True,crossfit=True,seed=iteration))
            records['boost_insample'].append(dml(x,d,y,flexible=True,crossfit=False,seed=iteration))
        for method, observations in records.items():
            v = np.asarray(observations)
            rows.append({'design':'nonlinear' if nonlinear else 'linear','method':method,
                         'n':N,'replications':REPS,'truth':TAU,'mean':float(v.mean()),
                         'bias':float(v.mean()-TAU),'sd':float(v.std(ddof=1)),
                         'rmse':float(np.sqrt(np.mean((v-TAU)**2)))})
    with Path(__file__).with_name('results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for r in rows:print(r)

if __name__ == '__main__':main()
