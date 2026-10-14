"""Scoring forecasts.

Scores are kept as sums per origin and lead day, so that every reported metric (for any
window, horizon or subset of days) is a ratio of sums computed later, never an average
of averages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_scale(contexts: np.ndarray, period: int) -> np.ndarray:
    """In-sample mean absolute error of the seasonal naive forecast on each context."""
    contexts = np.asarray(contexts, dtype=float)
    if contexts.shape[1] <= period:
        raise ValueError("context must be longer than the seasonal period")
    return np.mean(np.abs(contexts[:, period:] - contexts[:, :-period]), axis=1)


def crossing_rate(f: np.ndarray) -> float:
    """Share of (origin, hour) forecasts whose quantiles are not non-decreasing."""
    return float(np.mean(np.any(np.diff(f, axis=2) < 0, axis=2)))


def pinball(y: np.ndarray, f: np.ndarray, quantiles) -> np.ndarray:
    """Pinball loss for each quantile; shape (origins, horizon, quantiles)."""
    q = np.asarray(quantiles, dtype=float)[None, None, :]
    diff = y[:, :, None] - f
    return np.maximum(q * diff, (q - 1.0) * diff)


def score(
    y: np.ndarray,
    f: np.ndarray,
    quantiles,
    scale: np.ndarray,
    holiday: np.ndarray,
    hours_per_block: int = 24,
) -> pd.DataFrame:
    """Sums of errors for each origin and lead day.

    ``y`` has shape (origins, horizon), ``f`` (origins, horizon, quantiles). Quantiles are
    sorted at each hour before scoring (the rearrangement of Chernozhukov, Fernandez-Val
    and Galichon, 2010), which can only lower the pinball loss.
    """
    y = np.asarray(y, dtype=float)
    f = np.sort(np.asarray(f, dtype=float), axis=2)
    n, horizon = y.shape
    if f.shape[:2] != (n, horizon) or f.shape[2] != len(quantiles):
        raise ValueError(f"forecast shape {f.shape} does not match targets {y.shape}")
    if horizon % hours_per_block:
        raise ValueError("horizon must be a whole number of days")
    if not np.isfinite(f).all():
        raise ValueError("forecasts contain non-finite values")
    qs = list(quantiles)
    med = f[:, :, qs.index(0.5)]
    lo, hi = f[:, :, qs.index(0.1)], f[:, :, qs.index(0.9)]

    err = y - med
    parts = {
        "sum_abs_err": np.abs(err),
        "sum_sq_err": err**2,
        "sum_err": err,
        "sum_ape": np.abs(err) / np.abs(y),
        "sum_abs_y": np.abs(y),
        "sum_wql_num": 2.0 * pinball(y, f, qs).sum(axis=2),
        "n_inside80": ((lo <= y) & (y <= hi)).astype(float),
        "n_holiday_hours": np.asarray(holiday, dtype=float),
    }
    days = horizon // hours_per_block
    out = {"origin_idx": np.repeat(np.arange(n), days), "lead_day": np.tile(np.arange(1, days + 1), n)}
    for name, values in parts.items():
        out[name] = values.reshape(n, days, hours_per_block).sum(axis=2).ravel()
    out["n_hours"] = np.full(n * days, hours_per_block)
    out["n_quantiles"] = np.full(n * days, len(qs))
    out["scale"] = np.repeat(np.asarray(scale, dtype=float), days)
    return pd.DataFrame(out)


SUM_COLUMNS = [
    "sum_abs_err",
    "sum_sq_err",
    "sum_err",
    "sum_ape",
    "sum_abs_y",
    "sum_wql_num",
    "n_inside80",
    "n_hours",
    "n_holiday_hours",
]


def summarise(frame: pd.DataFrame) -> dict:
    """Metrics from a block of score rows (any subset of origins and lead days)."""
    s = frame[SUM_COLUMNS].sum()
    n_hours = s["n_hours"]
    q = frame["n_quantiles"].iloc[0]
    # MASE: each origin's MAE over the selected lead days divided by that origin's scale,
    # then averaged over origins (Hyndman and Koehler, 2006, with a seasonal naive scale).
    per_origin = frame.groupby("origin")[["sum_abs_err", "n_hours", "scale"]].agg(
        {"sum_abs_err": "sum", "n_hours": "sum", "scale": "first"}
    )
    mase = float(np.mean(per_origin["sum_abs_err"] / per_origin["n_hours"] / per_origin["scale"]))
    return {
        "origins": int(frame["origin"].nunique()),
        "hours": int(n_hours),
        "mae": float(s["sum_abs_err"] / n_hours),
        "rmse": float(np.sqrt(s["sum_sq_err"] / n_hours)),
        "bias": float(s["sum_err"] / n_hours),
        "mape": float(100.0 * s["sum_ape"] / n_hours),
        "mase": mase,
        "wql": float(s["sum_wql_num"] / (q * s["sum_abs_y"])),
        "coverage80": float(s["n_inside80"] / n_hours),
    }
