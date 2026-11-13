"""The split at the largest level shift, and the forward-in-time windows (PLAN.md).

r is a pair's monthly log(delivered / outside count). Everything else (the detector, the matched 5% budget, the
loss grid, the p95 rule) is the second-count-prediction code, copied unchanged into ``feed/detect.py``.
"""

from __future__ import annotations

import numpy as np

from feed import detect as D

MIN_LEFT, MIN_RIGHT = 24, 36  # months on each side of a split
REGIME = 0.5  # a split that removes at least half of the absolute deviation is a regime change
TRAIL, AHEAD = 36, 24  # forward test: predict from 36 months, observe over the next 24
MIN_TRAIL, MIN_AHEAD = 30, 20  # months that must be present in each


def l1(a: np.ndarray) -> float:
    return float(np.abs(a - np.median(a)).sum()) if len(a) else 0.0


def split(r: dict[str, float], min_left: int = MIN_LEFT, min_right: int = MIN_RIGHT) -> dict:
    """The month that starts the later segment, chosen to minimise the summed absolute deviation of r from the
    median of each segment; ``reduction`` is the share of the unsplit absolute deviation it removes."""
    ms = sorted(r)
    x = np.asarray([r[m] for m in ms], float)
    full = l1(x)
    best: tuple[int, float] | None = None
    for k in range(min_left, len(x) - min_right + 1):
        c = l1(x[:k]) + l1(x[k:])
        if best is None or c < best[1] - 1e-15:
            best = (k, c)
    if best is None:
        return {"month": None, "reduction": 0.0, "left": len(x), "right": 0}
    k, c = best
    return {
        "month": ms[k],
        "reduction": 1 - c / full if full > 0 else 0.0,
        "left": k,
        "right": len(x) - k,
        "median_left": float(np.median(x[:k])),
        "median_right": float(np.median(x[k:])),
    }


def segment(r: dict[str, float], start: str) -> dict[str, float]:
    return {m: v for m, v in r.items() if m >= start}


def add_months(m: str, k: int) -> str:
    y, mo = map(int, m.split("-"))
    i = y * 12 + mo - 1 + k
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def deviations(r: dict[str, float], lo: str | None = None, hi: str | None = None) -> list[float]:
    """|r - median of the 12 previous r| for the months in [lo, hi) that have 12 earlier months in the pair."""
    return [
        abs(x - float(np.median(h))) for m, x, h in D.history(r) if (lo is None or m >= lo) and (hi is None or m < hi)
    ]


def rule(devs: list[float]) -> float:
    return D.as_loss(2 * float(np.percentile(devs, 95)))


def window_curve(r: dict[str, float], lo: str, hi: str) -> tuple[dict[float, float], int]:
    """Detection at every grid loss over the months in [lo, hi), with the z-score history taken from the whole
    pair and the threshold matched to 5% of the window's clean months."""
    rows = [(m, x, h) for m, x, h in D.history(r) if lo <= m < hi]
    clean = {m: abs(D.robust_z(x, h)) for m, x, h in rows}
    t = D.matched_threshold(clean)
    cur = {}
    for p in D.GRID:
        d = np.log(1 - p)
        hits = sum(1 for m, x, h in rows if abs(D.robust_z(x + d, h)) > t and clean[m] <= t)
        cur[p] = hits / len(rows) if rows else float("nan")
    return cur, len(rows)


def forward_windows(r: dict[str, float]) -> list[dict]:
    """One window per January t: predict from [t - 36, t), observe over [t, t + 24)."""
    ms = sorted(r)
    if not ms:
        return []
    out = []
    t = f"{int(ms[0][:4]) + 1:04d}-01"
    while add_months(t, AHEAD) <= add_months(ms[-1], 1):
        trail = deviations(r, add_months(t, -TRAIL), t)
        expanding = deviations(r, None, t)
        cur, n_ahead = window_curve(r, t, add_months(t, AHEAD))
        if len(trail) >= MIN_TRAIL and n_ahead >= MIN_AHEAD:
            out.append(
                {
                    "start": t,
                    "n_trail": len(trail),
                    "n_expanding": len(expanding),
                    "n_ahead": n_ahead,
                    "pred_trailing": rule(trail),
                    "pred_expanding": rule(expanding),
                    "observed": D.observed_l90(cur),
                }
            )
        t = add_months(t, 12)
    return out
