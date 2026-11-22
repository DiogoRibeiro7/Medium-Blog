"""Scores: the average and the tail.

- WAPE: sum of absolute errors over sum of actual load, over all scored hours (the measure
  the grid-operator comparison in the LinkedIn post used);
- daily WAPE: the same within one local day;
- P95 and CVaR95 of daily WAPE: the 95th percentile, and the mean over the worst 5% of days;
- WAPE within strata of days (holidays, bridge days, the year end, extreme temperature, ...).

Intervals come from a circular moving-block bootstrap over calendar dates. The same dates
are drawn for every country and every model in a replication, so contrasts are paired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_errors(forecast: pd.DataFrame, actual: pd.Series, quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Per target day: absolute error, actual load, squared error, bias, interval coverage."""
    frame = forecast.copy()
    frame["actual"] = actual.reindex(pd.DatetimeIndex(frame["timestamp_utc"])).to_numpy()
    if frame["actual"].isna().any():
        raise ValueError("forecast hours without an actual value")
    point = frame["q0.5"]
    frame["abs"] = (frame["actual"] - point).abs()
    frame["sq"] = (frame["actual"] - point) ** 2
    frame["err"] = point - frame["actual"]
    lo, hi = f"q{min(quantiles):g}", f"q{max(quantiles):g}"
    frame["inside"] = (frame["actual"] >= frame[lo]) & (frame["actual"] <= frame[hi])
    g = frame.groupby("date")
    out = pd.DataFrame(
        {
            "abs": g["abs"].sum(),
            "actual": g["actual"].sum(),
            "sq": g["sq"].sum(),
            "err": g["err"].sum(),
            "inside": g["inside"].sum(),
            "hours": g.size(),
        }
    )
    out["wape"] = out["abs"] / out["actual"]
    return out


def wape(abs_err, actual) -> float:
    total = float(np.nansum(actual))
    return float(np.nansum(abs_err)) / total if total > 0 else float("nan")


def cvar(values, q: float) -> float:
    """Mean of the values at or above the q-quantile (the worst 1 - q share)."""
    v = np.sort(np.asarray(values, dtype=float)[~np.isnan(values)])
    if v.size == 0:
        return float("nan")
    return float(v[-tail_count(v.size, q) :].mean())


def tail_count(n: int, q: float) -> int:
    """How many of n values lie in the tail above the q-quantile (at least one)."""
    return max(1, int(np.ceil(round((1 - q) * n, 9))))


def summarise(days: pd.DataFrame, tail_q: float) -> dict:
    """Scores over a set of days (rows: one country-day each, columns from daily_errors)."""
    if days.empty:
        return {"days": 0}
    return {
        "days": int(len(days)),
        "wape": wape(days["abs"], days["actual"]),
        "rmse_mw": float(np.sqrt(days["sq"].sum() / days["hours"].sum())),
        "bias_pct": float(days["err"].sum() / days["actual"].sum()),
        "coverage_80": float(days["inside"].sum() / days["hours"].sum()) if "inside" in days else float("nan"),
        "mean_daily_wape": float(days["wape"].mean()),
        "p95_daily_wape": float(np.quantile(days["wape"], tail_q)),
        "cvar95_daily_wape": cvar(days["wape"].to_numpy(), tail_q),
        "max_daily_wape": float(days["wape"].max()),
    }


def block_indices(n: int, block: int, replications: int, seed: int) -> np.ndarray:
    """Circular moving-block bootstrap: (replications, n) indices into 0..n-1."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(replications, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(replications, -1)[:, :n]


def bootstrap(
    abs_err: np.ndarray,
    actual: np.ndarray,
    valid: np.ndarray,
    tail_q: float,
    block: int,
    replications: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Replicated pooled WAPE and CVaR of daily WAPE for each run.

    abs_err, actual: (runs, countries, dates); valid: (countries, dates), the days every run
    is scored on. Returns arrays of shape (replications, runs).
    """
    runs, _, n = abs_err.shape
    a = np.where(valid[None], abs_err, 0.0)
    y = np.where(valid[None], actual, 0.0)
    a_day, y_day = a.sum(axis=1), y.sum(axis=1)  # (runs, dates)
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = np.where(valid[None], abs_err / actual, np.nan)  # (runs, countries, dates)
    idx = block_indices(n, block, replications, seed)
    w = np.empty((replications, runs))
    c = np.empty((replications, runs))
    for r in range(replications):
        j = idx[r]
        w[r] = a_day[:, j].sum(axis=1) / y_day[:, j].sum(axis=1)
        sample = daily[:, :, j].reshape(runs, -1)
        keep = ~np.isnan(sample[0])
        sample = sample[:, keep]
        k = tail_count(sample.shape[1], tail_q)
        c[r] = np.partition(sample, sample.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
    return {"wape": w, "cvar": c}


def interval(values: np.ndarray, level: float = 0.95) -> list[float]:
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def rank(values: dict[str, float]) -> dict[str, int]:
    order = sorted(values, key=lambda k: (np.inf if np.isnan(values[k]) else values[k], k))
    return {k: i + 1 for i, k in enumerate(order)}


def kendall_tau(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) & set(b))
    n, s = len(keys), 0
    for i in range(n):
        for j in range(i + 1, n):
            x = np.sign(a[keys[i]] - a[keys[j]])
            y = np.sign(b[keys[i]] - b[keys[j]])
            s += x * y
    pairs = n * (n - 1) / 2
    return float(s / pairs) if pairs else float("nan")


def accuracy(pred: list, truth: list) -> float:
    pairs = [(p, t) for p, t in zip(pred, truth, strict=True) if isinstance(t, str)]
    if not pairs:
        return float("nan")
    return float(np.mean([p == t for p, t in pairs]))


def cluster_bootstrap_accuracy(correct: np.ndarray, clusters: np.ndarray, replications: int, seed: int) -> list[float]:
    """95% interval for an accuracy, resampling whole clusters (country-weeks)."""
    if correct.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=correct.astype(float))
    counts = np.bincount(inverse)
    draws = rng.integers(0, labels.size, size=(replications, labels.size))
    acc = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return interval(acc)
