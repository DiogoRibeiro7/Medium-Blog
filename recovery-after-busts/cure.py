"""Mixture cure model for recovery after a bust, with country-cluster bootstrap.

Each episode either never recovers (probability pi, the "cured" or immune fraction) or
recovers after a lognormal time T. With covariates z for the non-recovery part and x
for the timing part,

    pi_i      = 1 / (1 + exp(-z_i' gamma))
    log T_i   ~ Normal(x_i' beta, s^2)

and the likelihood of an episode observed for t years is

    recovered at t:      (1 - pi_i) f(t | x_i)
    not yet recovered:   pi_i + (1 - pi_i) S(t | x_i).

Identification of pi rests on long follow-up: it is the height of the plateau a
Kaplan-Meier curve would reach if observed forever. With short follow-up, the two parts
trade off and the bootstrap intervals widen accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr


@dataclass
class CureFit:
    gamma: pd.Series  # non-recovery (logit)
    beta: pd.Series  # log time to recovery
    log_s: float
    loglik: float
    n: int
    events: int


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in cols])
    return X


def fit(df: pd.DataFrame, t: str, event: str, z_cols: list[str], x_cols: list[str], start: np.ndarray | None = None, ridge: float = 1e-3) -> CureFit:
    T = np.maximum(df[t].to_numpy(dtype=float), 0.5)
    E = df[event].to_numpy(dtype=float)
    Z = _design(df, z_cols)
    X = _design(df, x_cols)
    kz, kx = Z.shape[1], X.shape[1]
    logT = np.log(T)

    def nll(theta):
        g, b, ls = theta[:kz], theta[kz : kz + kx], theta[-1]
        s = np.exp(ls)
        pi = np.clip(expit(Z @ g), 1e-10, 1 - 1e-10)
        u = (logT - X @ b) / s
        logf = -0.5 * u * u - np.log(s * np.sqrt(2 * np.pi)) - logT
        logS = log_ndtr(-u)
        ll_event = np.log1p(-pi) + logf
        ll_cens = np.log(pi + (1 - pi) * np.exp(logS))
        ll = np.where(E == 1, ll_event, ll_cens)
        return -ll.sum() + ridge * np.sum(theta[:-1] ** 2)

    if start is None:
        start = np.zeros(kz + kx + 1)
        start[kz] = np.log(np.median(T[E == 1])) if E.sum() else 1.0
    res = minimize(nll, start, method="BFGS", options={"maxiter": 2000, "gtol": 1e-6})
    th = res.x
    return CureFit(
        gamma=pd.Series(th[:kz], index=["const"] + z_cols),
        beta=pd.Series(th[kz : kz + kx], index=["const"] + x_cols),
        log_s=float(th[-1]),
        loglik=float(-res.fun),
        n=len(df),
        events=int(E.sum()),
    )


def fit_lognormal(df: pd.DataFrame, t: str, event: str, x_cols: list[str], ridge: float = 1e-3) -> CureFit:
    """Lognormal accelerated failure time model with no never-recover fraction (pi = 0)."""
    T = np.maximum(df[t].to_numpy(dtype=float), 0.5)
    E = df[event].to_numpy(dtype=float)
    X = _design(df, x_cols)
    logT = np.log(T)

    def nll(theta):
        b, ls = theta[:-1], theta[-1]
        s = np.exp(ls)
        u = (logT - X @ b) / s
        logf = -0.5 * u * u - np.log(s * np.sqrt(2 * np.pi)) - logT
        ll = np.where(E == 1, logf, log_ndtr(-u))
        return -ll.sum() + ridge * np.sum(b**2)

    start = np.zeros(X.shape[1] + 1)
    start[0] = np.log(np.median(T))
    res = minimize(nll, start, method="BFGS", options={"maxiter": 2000, "gtol": 1e-6})
    th = res.x
    return CureFit(gamma=pd.Series(dtype=float), beta=pd.Series(th[:-1], index=["const"] + x_cols), log_s=float(th[-1]), loglik=float(-res.fun), n=len(df), events=int(E.sum()))


def not_recovered(f: CureFit, df: pd.DataFrame, t: float) -> np.ndarray:
    """Model probability that each episode is still below its peak t years after the trough."""
    X = _design(df, [c for c in f.beta.index if c != "const"])
    u = (np.log(t) - X @ f.beta.to_numpy()) / np.exp(f.log_s)
    S = ndtr(-u)
    if f.gamma.empty:
        return S
    Z = _design(df, [c for c in f.gamma.index if c != "const"])
    pi = expit(Z @ f.gamma.to_numpy())
    return pi + (1 - pi) * S


def bootstrap(df: pd.DataFrame, t: str, event: str, z_cols: list[str], x_cols: list[str], cluster: str = "geo", B: int = 400, seed: int = 11) -> pd.DataFrame:
    """Cluster bootstrap over countries; returns one row per replicate."""
    rng = np.random.default_rng(seed)
    base = fit(df, t, event, z_cols, x_cols)
    start = np.concatenate([base.gamma.values, base.beta.values, [base.log_s]])
    groups = df[cluster].unique()
    idx = {g: np.flatnonzero(df[cluster].to_numpy() == g) for g in groups}
    out = []
    for _ in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        b = df.iloc[rows]
        if b[event].sum() < 3:
            continue
        try:
            f = fit(b, t, event, z_cols, x_cols, start=start)
        except Exception:
            continue
        out.append({**{f"gamma_{k}": v for k, v in f.gamma.items()}, **{f"beta_{k}": v for k, v in f.beta.items()}})
    return pd.DataFrame(out)


def predicted_never(f: CureFit, z: dict) -> float:
    v = f.gamma["const"] + sum(f.gamma[k] * z.get(k, 0.0) for k in f.gamma.index if k != "const")
    return float(expit(v))


def predicted_median_years(f: CureFit, x: dict) -> float:
    v = f.beta["const"] + sum(f.beta[k] * x.get(k, 0.0) for k in f.beta.index if k != "const")
    return float(np.exp(v))


def km(t: np.ndarray, e: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival (share not yet recovered) on a grid of times."""
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    times = np.unique(t[e == 1])
    S = 1.0
    surv = []
    for u in times:
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & (e == 1))
        S *= 1 - d / at_risk
        surv.append(S)
    surv = np.array(surv)
    out = np.ones_like(grid, dtype=float)
    for i, g in enumerate(grid):
        k = np.searchsorted(times, g, side="right") - 1
        out[i] = surv[k] if k >= 0 else 1.0
    return out


def _check() -> None:
    """Recover known parameters from simulated data."""
    rng = np.random.default_rng(0)
    n = 4000
    z1 = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n)
    pi = expit(-1.0 + 1.2 * z1)
    cured = rng.random(n) < pi
    T = np.exp(1.0 + 0.5 * x1 + 0.6 * rng.standard_normal(n))
    C = rng.uniform(2, 30, n)
    obs = np.where(cured, C, np.minimum(T, C))
    ev = (~cured) & (T <= C)
    df = pd.DataFrame({"t": obs, "e": ev.astype(int), "z1": z1, "x1": x1})
    f = fit(df, "t", "e", ["z1"], ["x1"])
    assert abs(f.gamma["const"] + 1.0) < 0.15 and abs(f.gamma["z1"] - 1.2) < 0.2
    assert abs(f.beta["const"] - 1.0) < 0.1 and abs(f.beta["x1"] - 0.5) < 0.1


if __name__ == "__main__":
    _check()
    print("cure model check passed")
