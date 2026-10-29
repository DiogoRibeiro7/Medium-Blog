"""Shared helpers: the transcribed BLS tables and piecewise-exponential survival through their knots."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

KNOT_YEARS = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 15.0])
LT_COLUMNS = ["lt1", "lt2", "lt5", "lt10", "lt15"]


# ---------------------------------------------------------------------------
# Reading the tables
# ---------------------------------------------------------------------------

def read_csv(name: str) -> list[dict[str, str]]:
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def value(cell: str) -> float:
    return float(cell) if cell.strip() else float("nan")


def nlsy79_rows() -> list[dict]:
    rows = []
    for r in read_csv("nlsy79_table2_vintages.csv"):
        rows.append({
            "release": r["release_date"],
            "survey": r["survey"],
            "phrasing": r["phrasing"],
            "group": r["age_group"],
            "lo": int(r["age_lo"]),
            "hi": int(r["age_hi"]),
            "F": np.array([value(r[c]) for c in LT_COLUMNS]) / 100.0,
            "ongoing": value(r["ongoing"]) / 100.0,
        })
    return rows


def pick(rows: list[dict], release: str, group: str) -> dict:
    for r in rows:
        if r["release"] == release and r["group"] == group:
            return r
    raise KeyError((release, group))


# ---------------------------------------------------------------------------
# Piecewise-exponential survival through published knots
# ---------------------------------------------------------------------------

def knots_from_F(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Survival at the knots 0, 1, 2, 5, 10, 15 years, truncated at the last published cell."""
    S = np.concatenate([[1.0], 1.0 - F])
    keep = ~np.isnan(S)
    last = np.max(np.nonzero(keep)[0])
    return KNOT_YEARS[: last + 1], S[: last + 1]


def interval_hazards(t: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.log(S[:-1] / S[1:]) / np.diff(t)


def survival_at(x: np.ndarray | float, t: np.ndarray, S: np.ndarray, tail_hazard: float | None = None) -> np.ndarray:
    """Piecewise-exponential survival; beyond the last knot, a constant hazard (or nan)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    h = interval_hazards(t, S)
    out = np.full_like(x, np.nan)
    for i, xi in enumerate(x):
        if xi <= t[-1]:
            k = min(np.searchsorted(t, xi, side="right") - 1, len(h) - 1)
            out[i] = S[k] * np.exp(-h[k] * (xi - t[k]))
        elif tail_hazard is not None:
            out[i] = S[-1] * np.exp(-tail_hazard * (xi - t[-1]))
    return out


def rmst(H: float, t: np.ndarray, S: np.ndarray, tail_hazard: float | None = None) -> float:
    """Restricted mean time with the employer up to H years: the integral of S from 0 to H."""
    h = interval_hazards(t, S)
    total = 0.0
    for k in range(len(h)):
        a, b = t[k], min(t[k + 1], H)
        if b <= a:
            break
        total += S[k] * (1.0 - np.exp(-h[k] * (b - a))) / h[k] if h[k] > 0 else S[k] * (b - a)
    if H > t[-1]:
        if tail_hazard is None:
            return float("nan")
        d = H - t[-1]
        total += S[-1] * ((1.0 - np.exp(-tail_hazard * d)) / tail_hazard if tail_hazard > 0 else d)
    return float(total)


def median_from_knots(t: np.ndarray, S: np.ndarray) -> float:
    h = interval_hazards(t, S)
    for k in range(len(h)):
        if S[k + 1] <= 0.5 < S[k]:
            return float(t[k] + np.log(S[k] / 0.5) / h[k])
    return float("nan")


def renewal_hires(H: float, t: np.ndarray, S: np.ndarray, tail_hazard: float | None = None, step: float = 1 / 120) -> float:
    """Expected number of hires needed to keep one seat filled for H years, 1 + M(H).

    Each hire's time with the employer is drawn from the same survival curve; vacancies
    are filled at once. M solves the renewal equation M(x) = F(x) + int_0^x M(x - u) dF(u)."""
    grid = np.arange(0.0, H + step / 2, step)
    if H > t[-1] and tail_hazard is None:
        return float("nan")
    F = 1.0 - survival_at(grid, t, S, tail_hazard)
    f = np.diff(F)  # probability mass of an exit in each step
    M = np.zeros_like(grid)
    for i in range(1, len(grid)):
        M[i] = F[i] + np.dot(M[i - 1 :: -1][: i], f[:i])
    return float(1.0 + M[-1])
