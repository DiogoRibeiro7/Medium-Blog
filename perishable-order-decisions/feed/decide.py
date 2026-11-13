"""From forecasts to orders, and the cost of an order (PLAN.md, "Decision layers" and "Cost")."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGETS = ("sales", "recovered")
RULES = ("point", "pooled", "series")
TAUS = (0.5, 0.7, 0.9)
DEFAULT = ("sales", "point")
FULL = ("recovered", "series")


@dataclass
class Calibration:
    level: np.ndarray  # (n,) mean target over the fit window
    scale: np.ndarray  # (n,) root mean squared calibration error
    by_level: np.ndarray  # pooled errors / level, series with level > 0
    by_scale: np.ndarray  # pooled errors / scale, series with scale > 0


def calibrate(y: np.ndarray, f: np.ndarray, fit: np.ndarray, cal: np.ndarray) -> Calibration:
    level = y[:, fit].mean(axis=1)
    err = y[:, cal] - f[:, cal]
    scale = np.sqrt((err**2).mean(axis=1))
    by_level = (err[level > 0] / level[level > 0, None]).ravel()
    by_scale = (err[scale > 0] / scale[scale > 0, None]).ravel()
    return Calibration(level, scale, by_level, by_scale)


def order(f: np.ndarray, rule: str, tau: float, cal: Calibration) -> np.ndarray:
    """Orders for the columns of f (series x days)."""
    if rule == "point":
        q = f
    elif rule == "pooled":
        q = f + cal.level[:, None] * np.quantile(cal.by_level, tau)
    elif rule == "series":
        q = f + cal.scale[:, None] * np.quantile(cal.by_scale, tau)
    else:
        raise ValueError(rule)
    return np.maximum(q, 0.0)


def cost(q: np.ndarray, d: np.ndarray, tau: float) -> np.ndarray:
    return tau * np.maximum(d - q, 0.0) + (1 - tau) * np.maximum(q - d, 0.0)
