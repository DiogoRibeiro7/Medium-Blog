"""Drift between price bases for every pair of economies, and the groups of the PPP programme.

For economies i and j, relative to the reference year 2020,

    drift_ij,t = ln(P_i,t / P_j,t) - ln(PPP_i,t / PPP_j,t)       (P deflator, PPP per US dollar; 2020 = 1)

the change in relative prices recorded by the national deflators minus the change recorded by the PPPs. Because
deflator ratios and PPPs are both transitive, drift_ij = drift_iUS - drift_jUS exactly: every pair is the
difference of two country scores measured against any common reference.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
BASE, LAST = 2020, 2024

EU27 = ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA",
        "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]
# economies whose PPPs Eurostat compiles (the United Kingdom until it moved to the OECD side after leaving the EU)
EUROSTAT_GROUP = EU27 + ["CHE", "NOR", "ISL", "TUR", "SRB", "MKD", "ALB", "MNE", "BIH", "GBR"]
# economies whose PPPs the OECD compiles: its non-European members
OECD_GROUP = ["USA", "JPN", "KOR", "CAN", "AUS", "NZL", "ISR", "MEX", "CHL", "COL", "CRI"]
AGGREGATES = {"HFCE": ("S14", "P3", "PPP_P31S14"), "AIC": ("S1", "P41", "PPP_P41"), "GDP": ("S1", "B1GQ", "PPP_B1GQ")}
# Eurostat-OECD benchmark years in the period (the OECD's own calculation was triennial until 2016)
BENCHMARKS = [1999, 2002, 2005, 2008, 2011, 2014, 2017]


def _pivot(df: pd.DataFrame, **match) -> pd.DataFrame:
    x = df[df.TIME_PERIOD <= LAST]
    for k, v in match.items():
        x = x[x[k] == v]
    return x.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE", aggfunc="first")


def load() -> dict:
    t1 = pd.read_csv(DATA / "oecd_t1.csv", low_memory=False)
    t4 = pd.read_csv(DATA / "oecd_t4.csv", low_memory=False)
    out: dict = {"V": {}, "LR": {}, "PPP": {}}
    for name, (sector, tr, ppp) in AGGREGATES.items():
        x = t1[(t1.SECTOR == sector) & (t1.TRANSACTION == tr)]
        out["V"][name] = _pivot(x, PRICE_BASE="V")
        out["LR"][name] = _pivot(x, PRICE_BASE="LR")
        out["PPP"][name] = _pivot(t4, TRANSACTION=ppp)
    out["EXC"] = _pivot(t4, TRANSACTION="EXC_A")
    return out


def group_of(code: str) -> str | None:
    if code in EUROSTAT_GROUP:
        return "Eurostat"
    if code in OECD_GROUP:
        return "OECD"
    return None


def deflator(v: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    p = v / lr
    return p / p.loc[BASE]


def drift_vs(d: dict, agg: str, ref: str = "USA") -> pd.DataFrame:
    """ln drift x 100 of every economy against `ref`, 2020 = 0."""
    p = deflator(d["V"][agg], d["LR"][agg])
    ppp = d["PPP"][agg] / d["PPP"][agg].loc[BASE]
    return 100 * np.log(p.div(p[ref], axis=0) / ppp.div(ppp[ref], axis=0))


def pair_drift(d: dict, agg: str, i: str, j: str) -> pd.Series:
    """ln drift x 100 of i against j, computed directly from the pair (not through the US)."""
    return drift_vs(d, agg, ref=j)[i]


def window(cd: pd.DataFrame, a: int, b: int) -> pd.Series:
    """Change in drift between years a and b for every economy with both years."""
    return (cd.loc[b] - cd.loc[a]).dropna()


def programme_scores(cd: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    s = window(cd, a, b)
    df = pd.DataFrame({"drift": s})
    df["group"] = [group_of(c) for c in df.index]
    return df.dropna(subset=["group"])


def pairs_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, j in combinations(sorted(scores.index), 2):
        gi, gj = scores.loc[i, "group"], scores.loc[j, "group"]
        kind = "within Eurostat" if gi == gj == "Eurostat" else "within OECD" if gi == gj == "OECD" else "across"
        rows.append({"i": i, "j": j, "kind": kind, "drift": scores.loc[i, "drift"] - scores.loc[j, "drift"]})
    return pd.DataFrame(rows)


def block_model(scores: pd.DataFrame, control: pd.Series | None = None) -> dict:
    """OLS of country drift on a Eurostat-group indicator (and optionally a control)."""
    y = scores["drift"].astype(float)
    e = (scores["group"] == "Eurostat").astype(float)
    cols = [np.ones(len(y)), e.values]
    if control is not None:
        cols.append(control.reindex(y.index).values.astype(float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    fit = X @ beta
    r2 = 1 - ((y.values - fit) ** 2).sum() / ((y.values - y.values.mean()) ** 2).sum()
    return {"beta": beta, "r2": float(r2), "n": len(y)}


def permutation_p(scores: pd.DataFrame, reps: int = 10000, seed: int = 20261113) -> float:
    """Share of random relabellings (same group sizes) whose gap is at least as large as the observed one."""
    rng = np.random.default_rng(seed)
    y = scores["drift"].values.astype(float)
    e = (scores["group"] == "Eurostat").values
    obs = y[e].mean() - y[~e].mean()
    hits = 0
    for _ in range(reps):
        perm = rng.permutation(e)
        if abs(y[perm].mean() - y[~perm].mean()) >= abs(obs):
            hits += 1
    return (hits + 1) / (reps + 1)
