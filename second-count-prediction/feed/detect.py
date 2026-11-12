"""The detector of Part 4, the prediction rule under test, and the loss grid.

Detector (unchanged from Part 4, ``data-tests-limits/feed/counts.py``): r_t = log(delivered_t / outside_t);
robust z of r_t against the 12 previous months that have both counts (median, 1.4826 x MAD, scale floored at
0.005); a random loss of a fraction p of the rows adds log(1 - p) to r_t. The threshold is matched to a budget
of 5% alarms on the clean months, and a faulted month counts as caught only if the detector fires on it and
not on the same month without the loss.

Prediction under test (conjecture 1 of Part 4): with D = the 95th percentile, over the evaluated months, of
|r_t - median of the 12 previous r|, the smallest loss caught at least 90% of the time is about 2D (in log
units). The comparison rule replaces D by its Gaussian counterpart, 1.96 x the median of the trailing robust
scales, so it uses the typical spread and ignores the tails.
"""

from __future__ import annotations

import math

import numpy as np

WINDOW, FLOOR, BUDGET, TARGET = 12, 0.005, 0.05, 0.90
GRID = tuple(round(10 ** (k / 10) / 1000, 6) for k in range(0, 28))  # 0.1% to 50.1%, ratio 10^(1/10)


def ratio_series(delivered: dict[str, int], outside: dict[str, float]) -> dict[str, float]:
    return {
        m: math.log(delivered[m] / outside[m]) for m in sorted(delivered) if outside.get(m, 0) > 0 and delivered[m] > 0
    }


def history(r: dict[str, float], first: str | None = None) -> list[tuple[str, float, list[float]]]:
    """(month, value, the 12 previous values) for every month with a full window, from ``first`` on."""
    ms = sorted(r)
    return [
        (m, r[m], [r[ms[j]] for j in range(i - WINDOW, i)])
        for i, m in enumerate(ms)
        if i >= WINDOW and (first is None or m >= first)
    ]


def robust_z(x: float, hist: list[float], floor: float = FLOOR) -> float:
    h = np.asarray(hist, float)
    med = float(np.median(h))
    return (x - med) / max(1.4826 * float(np.median(np.abs(h - med))), floor)


def scores(r: dict[str, float], loss: float = 0.0, first: str | None = None) -> dict[str, float]:
    d = math.log(1 - loss)
    return {m: abs(robust_z(x + d, h)) for m, x, h in history(r, first)}


def matched_threshold(clean: dict[str, float], budget: float = BUDGET) -> float:
    v = np.sort(np.asarray(list(clean.values()), float))
    allowed = int(math.floor(budget * len(v)))
    return float(v[len(v) - allowed - 1]) if allowed < len(v) else 0.0


def detection(r: dict[str, float], loss: float, first: str | None = None) -> float:
    clean = scores(r, 0.0, first)
    t = matched_threshold(clean)
    faulted = scores(r, loss, first)
    hits = sum(1 for m in faulted if faulted[m] > t and clean[m] <= t)
    return hits / len(faulted) if faulted else math.nan


def curve(r: dict[str, float], first: str | None = None) -> dict[float, float]:
    return {p: detection(r, p, first) for p in GRID}


def observed_l90(cur: dict[float, float], target: float = TARGET) -> float | None:
    """Smallest grid loss from which detection stays at or above the target; None if it never gets there."""
    ps = sorted(cur)
    for i, p in enumerate(ps):
        if all(cur[q] >= target for q in ps[i:]):
            return p
    return None


def deviation_p95(r: dict[str, float], first: str | None = None) -> float:
    return float(np.percentile([abs(x - float(np.median(h))) for _, x, h in history(r, first)], 95))


def median_scale(r: dict[str, float], first: str | None = None) -> float:
    s = []
    for _, _, h in history(r, first):
        a = np.asarray(h, float)
        s.append(1.4826 * float(np.median(np.abs(a - np.median(a)))))
    return float(np.median(s))


def as_loss(d: float) -> float:
    """A shift of d in log units as a fraction of rows lost."""
    return 1 - math.exp(-d)


def predicted(r: dict[str, float], first: str | None = None) -> dict[str, float]:
    return {
        "p95_rule": as_loss(2 * deviation_p95(r, first)),
        "gaussian_rule": as_loss(2 * 1.96 * median_scale(r, first)),
    }


def within_factor_two(pred: float, obs: float | None, top: float = GRID[-1]) -> bool:
    """PLAN.md: obs within [pred / 2, 2 pred]; if the target is never reached, it holds when 2 pred > top."""
    if obs is None:
        return 2 * pred > top
    return pred / 2 <= obs <= 2 * pred


def log_error(pred: float, obs: float | None, top: float = GRID[-1]) -> float:
    """|log(obs / pred)|, with a target never reached taken as the top of the grid (a lower bound)."""
    return abs(math.log((obs if obs is not None else top) / pred))
