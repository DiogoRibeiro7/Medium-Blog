"""Synthetic Price Paid Data with a known market index and a planted purchase-price effect.

Each home has a value that follows its postcode area's index plus a small home-specific drift.
A sale price is that value plus transaction noise. At a resale the seller knows the home's
value; when it is below the purchase price, the price is raised by MARKUP times the expected
loss (the Genesove-Mayer pattern), and a resale that would land just below the purchase price
either moves to exactly the old price or does not happen that month (the floor).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

AREAS = {"LS": ("WEST YORKSHIRE", 0.0), "SW": ("GREATER LONDON", 0.7), "CF": ("CARDIFF", -0.1),
         "OX": ("OXFORDSHIRE", 0.4)}
N_MONTHS = 372  # 1995-01 .. 2025-12
MARKUP = 0.3
FLOOR_BAND = 0.04


def area_paths(rng: np.random.Generator) -> dict[str, np.ndarray]:
    t = np.arange(N_MONTHS)
    g = np.where(t < 153, 0.007, 0.0)                       # boom to 2007-09
    g = np.where((t >= 153) & (t < 171), -0.012, g)          # bust to 2009-03
    g = np.where((t >= 171) & (t < 216), -0.001, g)          # flat to 2012-12
    g = np.where((t >= 216) & (t < 324), 0.004, g)           # recovery to 2021-12
    g = np.where((t >= 324) & (t < 348), -0.003, g)          # 2022-2023 fall
    g = np.where(t >= 348, 0.002, g)
    out = {}
    for k, a in enumerate(AREAS):
        tilt = 0.001 * (k - 1.5)
        gg = g + tilt * (t < 216) + rng.normal(0, 0.002, N_MONTHS)
        out[a] = np.concatenate([[0.0], np.cumsum(gg[1:])])
    return out


def month_str(t: int, day: int) -> str:
    return f"{1995 + t // 12}-{t % 12 + 1:02d}-{day:02d} 00:00"


def make_ppd(path: Path, homes_per_area: int = 5000, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    paths = area_paths(rng)
    rows = []
    tid = 0
    for a, (county, prem) in AREAS.items():
        idx = paths[a]
        for h in range(homes_per_area):
            outward = f"{a}{1 + h % 3}"
            pc = f"{outward} {1 + (h // 3) % 4}AB"
            ptype = rng.choice(["D", "S", "T", "F"], p=[0.2, 0.3, 0.3, 0.2])
            tenure = "L" if ptype == "F" else "F"
            v0 = np.log(120000) + prem + rng.normal(0, 0.35)
            drift = np.concatenate([[0.0], np.cumsum(rng.normal(0, 0.006, N_MONTHS - 1))])
            new_first = rng.random() < 0.08
            t = int(rng.integers(0, 190))
            lp_prev = None
            prev_price = 0
            first = True
            while t < N_MONTHS:
                value = v0 + idx[t] + drift[t]
                lp = value + rng.normal(0, 0.08) + (0.10 if (first and new_first) else 0.0)
                sold, exact = True, False
                if lp_prev is not None:
                    lp += MARKUP * max(0.0, lp_prev - value)
                    if -FLOOR_BAND < lp - lp_prev < 0:
                        u = rng.random()
                        exact = u < 0.35
                        sold = exact or u >= 0.70
                if sold:
                    price = prev_price if exact else int(round(np.exp(lp), -2))
                    tid += 1
                    rows.append([f"{{S{tid:07d}}}", str(price), month_str(t, 1 + tid % 27), pc, ptype,
                                 "Y" if (first and new_first) else "N", tenure, str(h + 1),
                                 "FLAT 1" if ptype == "F" else "", "STREET", "", "TOWN", "DIST",
                                 county, "A", "A"])
                    lp_prev = np.log(price)
                    prev_price = price
                    first = False
                    t += int(rng.integers(7, 160))
                else:
                    t += int(rng.integers(1, 12))
    with path.open("w", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerows(rows)
    return paths


def true_index_frame(paths: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({"area": a, "t": np.arange(N_MONTHS), "true": p})
                      for a, p in paths.items()], ignore_index=True)
