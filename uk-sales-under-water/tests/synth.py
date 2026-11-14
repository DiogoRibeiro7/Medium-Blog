"""Synthetic Price Paid Data with a known index and a planted lock-in below the purchase price.

Each month, an owner sells with probability BASE_HAZARD, multiplied by LOCK_IN when the
home's value is below the price they paid. Prices follow the postcode area's index plus a
small home-specific drift and transaction noise.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

AREAS = {"LS": ("WEST YORKSHIRE", 0.0), "SW": ("GREATER LONDON", 0.7), "CF": ("CARDIFF", -0.1),
         "OX": ("OXFORDSHIRE", 0.4)}
N_MONTHS = 372          # 1995-01 .. 2025-12
BASE_HAZARD = 1 / 110
LOCK_IN = 0.4


def area_paths(rng: np.random.Generator) -> dict[str, np.ndarray]:
    t = np.arange(N_MONTHS)
    g = np.where(t < 153, 0.007, 0.0)
    g = np.where((t >= 153) & (t < 171), -0.012, g)
    g = np.where((t >= 171) & (t < 216), -0.001, g)
    g = np.where((t >= 216) & (t < 324), 0.004, g)
    g = np.where((t >= 324) & (t < 348), -0.003, g)
    g = np.where(t >= 348, 0.002, g)
    out = {}
    for k, a in enumerate(AREAS):
        gg = g + 0.001 * (k - 1.5) * (t < 216) + rng.normal(0, 0.002, N_MONTHS)
        out[a] = np.concatenate([[0.0], np.cumsum(gg[1:])])
    return out


def make_ppd(path: Path, homes_per_area: int = 4000, seed: int = 11, lock_in: float = LOCK_IN):
    rng = np.random.default_rng(seed)
    paths = area_paths(rng)
    rows, tid = [], 0
    for a, (county, prem) in AREAS.items():
        idx = paths[a]
        for h in range(homes_per_area):
            outward = f"{a}{1 + h % 6}"
            pc = f"{outward} {1 + (h // 6) % 4}AB"
            ptype = rng.choice(["D", "S", "T", "F"], p=[0.2, 0.3, 0.3, 0.2])
            v0 = np.log(120000) + prem + rng.normal(0, 0.35)
            drift = np.concatenate([[0.0], np.cumsum(rng.normal(0, 0.006, N_MONTHS - 1))])
            t = int(rng.integers(0, 120))
            lp_prev, first = None, True
            while t < N_MONTHS:
                value = v0 + idx[t] + drift[t]
                sell = True
                if lp_prev is not None:
                    hz = BASE_HAZARD * (lock_in if value < lp_prev else 1.0)
                    sell = rng.random() < hz
                if sell:
                    price = int(round(np.exp(value + rng.normal(0, 0.06)), -2))
                    tid += 1
                    rows.append([f"{{S{tid:07d}}}", str(price),
                                 f"{1995 + t // 12}-{t % 12 + 1:02d}-{1 + tid % 27:02d} 00:00", pc, ptype,
                                 "Y" if first and rng.random() < 0.1 else "N",
                                 "L" if ptype == "F" else "F", str(h + 1), "", "STREET", "", "TOWN",
                                 "DIST", county, "A", "A"])
                    lp_prev, first = np.log(price), False
                    t += 6
                else:
                    t += 1
    with path.open("w", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerows(rows)
    return paths
