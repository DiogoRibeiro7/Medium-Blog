"""Hourly workload on the local-clock grid of the test weeks, and the requirement per hour (PLAN.md)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_MONDAY = pd.Timestamp("2018-12-31")
WEEKS = 52
WEEK = 168
LENGTHS = (6, 8, 9, 10, 12)
DIVISORS = (6, 8, 12)
BOROUGHS = ("BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "RICHMOND / STATEN ISLAND")
DATASETS = ("ems_city", "ems_bronx", "ems_brooklyn", "ems_manhattan", "ems_queens", "ems_staten_island", "nyc311")
LABELS = {
    "ems_city": "EMS, citywide",
    "ems_bronx": "EMS, Bronx",
    "ems_brooklyn": "EMS, Brooklyn",
    "ems_manhattan": "EMS, Manhattan",
    "ems_queens": "EMS, Queens",
    "ems_staten_island": "EMS, Staten Island",
    "nyc311": "NYC 311 calls",
}
BOROUGH_OF = dict(zip(DATASETS[1:6], BOROUGHS, strict=True))
EMS_ALPHA = 0.10
AHT_311 = 300.0  # seconds
SL_TARGET_311 = 0.80
WAIT_311 = 30.0


def grid() -> pd.DatetimeIndex:
    """Every local clock hour of the 52 test weeks."""
    return pd.date_range(FIRST_MONDAY, periods=WEEKS * WEEK, freq="h")


def load_ems(path: Path) -> dict[str, np.ndarray]:
    """Offered load (mean busy ambulances) per hour of the grid: citywide and per borough."""
    g = grid()
    pos = {t: i for i, t in enumerate(g)}
    minutes = {name: np.zeros(len(g)) for name in DATASETS if name.startswith("ems")}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = pos.get(pd.Timestamp(r["hour"] + ":00"))
            if i is None:
                continue
            m = float(r["busy_minutes"])
            minutes["ems_city"][i] += m
            for name, b in BOROUGH_OF.items():
                if r["borough"] == b:
                    minutes[name][i] += m
    return {k: v / 60.0 for k, v in minutes.items()}


def load_311(path: Path) -> np.ndarray:
    """Calls per hour of the grid; hours absent from the counts had no calls."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    s = pd.Series({pd.Timestamp(r["hour"] + ":00"): float(r["calls"]) for r in rows})
    s = s.groupby(level=0).sum()
    return s.reindex(grid(), fill_value=0.0).to_numpy(dtype=np.float64)


def _erlang_b(a: np.ndarray, smax: int) -> np.ndarray:
    B = np.empty((smax + 1, a.size))
    B[0] = 1.0
    for k in range(1, smax + 1):
        B[k] = a * B[k - 1] / (k + a * B[k - 1])
    return B


def erlang_c(s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Probability of waiting in M/M/s with offered load a; 1 where s <= a."""
    s = np.asarray(s, dtype=np.int64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    out = np.ones(a.size)
    ok = s > a
    if ok.any():
        B = _erlang_b(a[ok], int(s[ok].max()))
        b = B[s[ok], np.arange(ok.sum())]
        out[ok] = s[ok] * b / (s[ok] - a[ok] * (1 - b))
    return out


def _smallest(a: np.ndarray, ok) -> np.ndarray:
    """Smallest s > a with ok(s, a) True, per element; 0 where a == 0."""
    a = np.asarray(a, dtype=np.float64)
    shape = a.shape
    a = a.ravel()
    need = np.zeros(a.size, dtype=np.int64)
    pos = a > 0
    if not pos.any():
        return need.reshape(shape)
    ap = a[pos]
    smax = int(np.ceil(ap.max() + 12 * np.sqrt(ap.max()) + 20))
    B = _erlang_b(ap, smax)
    found = np.zeros(ap.size, dtype=np.int64)
    for k in range(1, smax + 1):
        todo = (found == 0) & (k > ap)
        if not todo.any():
            if (found > 0).all():
                break
            continue
        b = B[k, todo]
        C = k * b / (k - ap[todo] * (1 - b))
        idx = np.flatnonzero(todo)[ok(k, ap[todo], C)]
        found[idx] = k
    if (found == 0).any():
        raise ValueError("requirement search did not converge")
    need[pos] = found
    return need.reshape(shape)


def ems_need(load: np.ndarray, alpha: float | None = EMS_ALPHA) -> np.ndarray:
    """Ambulances needed: Erlang C probability of finding none free at most alpha; alpha None: load rounded up."""
    load = np.asarray(load, dtype=np.float64)
    if alpha is None:
        return np.ceil(load - 1e-9).astype(np.int64)
    return _smallest(load, lambda k, a, C: alpha >= C)


def nyc311_need(calls: np.ndarray) -> np.ndarray:
    """Agents needed to answer 80% of calls within 30 seconds, 5-minute handle time (as in #194)."""
    calls = np.asarray(calls, dtype=np.float64)
    a = calls * AHT_311 / 3600.0
    return _smallest(a, lambda k, a, C: 1 - C * np.exp(-(k - a) * WAIT_311 / AHT_311) >= SL_TARGET_311)


def weeks(hourly: np.ndarray) -> np.ndarray:
    """(52, 168) array of a series on the grid."""
    return np.asarray(hourly).reshape(WEEKS, WEEK)


def requirements(sources: Path, alpha: float | None = EMS_ALPHA) -> dict[str, np.ndarray]:
    """Requirement per dataset as (52, 168) integer arrays."""
    ems = load_ems(sources / "ems_hourly.csv")
    out = {name: weeks(ems_need(ems[name], alpha)) for name in DATASETS if name.startswith("ems")}
    out["nyc311"] = weeks(nyc311_need(load_311(sources / "nyc311_hourly_calls.csv")))
    return out
