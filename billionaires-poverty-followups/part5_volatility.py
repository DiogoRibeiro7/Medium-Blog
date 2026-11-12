"""Part 5: are fortunes near $1 billion more volatile, and is that why they leave the list?

Part 4's crossing model applied each year's changes of fortunes above $3 billion
to everyone and predicted 74% of the departures observed below $1.5 billion. It
conjectured that fortunes near the line are more volatile, and that this, not
deaths or reassessments, explains most of the gap. This script tests that on the
linked Forbes lists 2010-2025:

* how fortunes are rounded, which fixes where people sit relative to the line;
* the upper tail of annual changes by band, which is never cut off by the line,
  compared with fortunes above $3 billion in the same year;
* a scale factor per band estimated only from that upper tail, and the
  departures it implies once applied to the lower tail;
* a floor of departures unrelated to market value (deaths, reassessments),
  standardised for the age of each band;
* departures by tenure (new entrants, people returning, incumbents) and by the
  previous year's change.

Every number quoted in the Part 5 article is computed here and written to
``results/part5/numbers.json``; ``--check`` recomputes everything and fails if
any number differs. Figures go to ``figures/part5/``.

Usage::

    python part5_volatility.py           # print, write results/part5 and figures/part5
    python part5_volatility.py --check   # fail if any number differs from numbers.json

All results are descriptive, not causal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from part1_poverty_lines import (AQUA, BASELINE, BLUE, INK, MUTED, ORANGE, SECONDARY,  # noqa: E402
                                 _compare, _frame, _style, fmt)
from part4_threshold_crossings import ChangeDist  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results" / "part5"
FIGURES = HERE / "figures" / "part5"

FIRST, LAST = 2010, 2024          # pairs of lists t -> t+1, t = 2010..2024
NEAR = 1500
FAR = 3000
UPPER_X = [0.02, 0.06, 0.1, 0.2, 0.3]
BANDS = [1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000]
EDGE_HI = {1000: 1100, 1100: 1200, 1200: 1300, 1300: 1400, 1400: 1500, 1500: 1750, 1750: 2000, 2000: 2500,
           2500: 3000, 3000: 4000}
N_BOOT = 400
SOURCE = ("Data: Forbes annual billionaire lists 2010-2025, linked year to year; only aggregate counts are stored. "
          "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/billionaires-poverty-followups")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for man, files in (("forbes_flows_manifest.json", {"forbes_global.csv.gz": "global_sha256",
                                                       "forbes_flows.csv.gz": "flows_sha256"}),
                       ("forbes_tenure_manifest.json", {"forbes_tenure.csv.gz": "tenure_sha256"})):
        m = json.loads((DATA / man).read_text(encoding="utf-8"))
        for name, key in files.items():
            if _sha(DATA / name) != m[key]:
                raise ValueError(f"{name} does not match {man}")
            out[name.split(".")[0]] = pd.read_csv(DATA / name, keep_default_na=False)
        out[man.split(".")[0]] = m
    return out


def check_against_part4(tenure: pd.DataFrame, glob: pd.DataFrame) -> dict[str, Any]:
    """The tenure tables must reproduce Part 4's stock, departures and annual changes exactly."""
    t = tenure[tenure.year.between(FIRST, LAST)]
    g = glob[glob.year.between(FIRST, LAST)]
    out = {}
    pairs = {
        "stock": (t[t.table == "stock"].groupby(["year", "band_lo_musd"]).n.sum(),
                  g[g.table.isin(["stay_age", "exit_age"])].groupby(["year", "band_lo_musd"]).n.sum()),
        "exit": (t[t.table == "exit"].groupby(["year", "band_lo_musd"]).n.sum(),
                 g[g.table == "exit_age"].groupby(["year", "band_lo_musd"]).n.sum()),
        "logchange": (t[t.table == "logchange"].groupby(["year", "band_lo_musd", "key"]).n.sum(),
                      g[g.table == "logchange"].groupby(["year", "band_lo_musd", "key"]).n.sum()),
    }
    for k, (a, b) in pairs.items():
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0)
        bad = int((j.a != j.b).sum())
        if bad:
            raise ValueError(f"tenure table {k} disagrees with Part 4 in {bad} cells")
        out[k] = {"cells": int(len(j)), "mismatches": bad}
    return out


# ---------------------------------------------------------------------------
# 1. Rounding
# ---------------------------------------------------------------------------

def rounding(tenure: pd.DataFrame) -> dict[str, Any]:
    r = tenure[(tenure.table == "rounding") & (tenure.band_lo_musd >= 1000)].copy()
    r["grp"] = np.where(r.band_lo_musd < 2000, "below_2000", "from_2000")
    t = r.groupby(["grp", "key"]).n.sum().unstack(fill_value=0)
    out = {g: {"m100": float(t.loc[g, "m100"] / t.loc[g].sum()), "n": int(t.loc[g].sum())} for g in t.index}
    b = r[r.band_lo_musd == 1000].groupby("key").n.sum()
    out["band_1000_m100"] = float(b.get("m100", 0) / b.sum())
    return out


# ---------------------------------------------------------------------------
# 2. Change distributions and the upper tail
# ---------------------------------------------------------------------------

def _hist(lc: pd.DataFrame) -> pd.Series:
    return lc.assign(lo=lc.key.astype(int)).groupby("lo").n.sum()


def far_dists(tenure: pd.DataFrame, width: float, years: Sequence[int], incumbents_only: bool = False) -> dict[int, ChangeDist]:
    """Each year's distribution of changes for fortunes from $3bn (the benchmark)."""
    out = {}
    for y in years:
        lc = tenure[(tenure.table == "logchange") & (tenure.year == y) & (tenure.band_lo_musd >= FAR)]
        if incumbents_only:
            lc = lc[lc.tenure == "incumbent"]
        h = _hist(lc)
        out[y] = ChangeDist(h.index.to_numpy(float) / 1000.0, h.to_numpy(float), width)
    return out


def cell_counts(tenure: pd.DataFrame, years: Sequence[int], band_sel: Any, tenure_sel: Any = None) -> dict[int, dict[str, Any]]:
    """Per year: stock, departures and the histogram of stayers' changes for the selected bands."""
    out = {}
    for y in years:
        d = tenure[(tenure.year == y) & band_sel(tenure.band_lo_musd)]
        if tenure_sel is not None:
            d = d[tenure_sel(d)]
        st = int(d[d.table == "stock"].n.sum())
        ex = int(d[d.table == "exit"].n.sum())
        h = _hist(d[d.table == "logchange"])
        out[y] = {"stock": st, "exit": ex, "hist": h}
    return out


def upper_tail(cells: dict[int, dict[str, Any]], dists: dict[int, ChangeDist], xs: Sequence[float]) -> dict[str, Any]:
    """Observed against expected rises of at least x, expected from the same year's large fortunes."""
    out = {}
    for x in xs:
        o = e = 0.0
        for y, c in cells.items():
            o += float(c["hist"][c["hist"].index >= round(x * 1000) - 1e-9].sum())
            e += c["stock"] * (1 - float(dists[y].cdf(x)[0]))
        out[str(x)] = {"observed": o, "expected": e, "ratio": o / e,
                       "ci95": [o / e * np.exp(-1.96 / np.sqrt(o)), o / e * np.exp(1.96 / np.sqrt(o))]}
    return out


TAIL_X = (0.1, 0.14, 0.2, 0.3)


def fit_scale(cells: dict[int, dict[str, Any]], dists: dict[int, ChangeDist], floor: float,
              xs: Sequence[float] = TAIL_X) -> float:
    """Scale of changes relative to large fortunes, estimated only from rises of at least 0.1 in log10.

    Rises that large are observed for everyone who stays on the list, whatever their distance from the line, and
    rounding to $100 million barely moves them. The model is change = m_t + s * (benchmark change - m_t), with m_t
    the median change of fortunes above $3bn in year t, and a share ``floor`` of people leaving for reasons unrelated
    to market value. The likelihood is multinomial over the intervals between the thresholds ``xs`` and a remainder
    (smaller rises, falls and departures), so it does not depend on how the lower part is cut off or rounded."""
    edges = [round(x * 1000) for x in xs]

    def nll(log_s: float) -> float:
        s = np.exp(log_s)
        ll = 0.0
        for y, c in cells.items():
            d, h = dists[y], c["hist"]
            m = d.quantile(0.5)
            surv = [(1 - floor) * (1 - float(d.cdf(m + (x - m) / s)[0])) for x in xs] + [0.0]
            counts = [float(h[(h.index >= a) & (h.index < b)].sum()) for a, b in zip(edges, edges[1:] + [10**9])]
            probs = [max(a - b, 1e-12) for a, b in zip(surv[:-1], surv[1:])]
            rest = c["stock"] - sum(counts)
            ll += sum(k * np.log(q) for k, q in zip(counts, probs)) + rest * np.log(max(1 - surv[0], 1e-12))
        return -ll
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.4), np.log(3.0)), method="bounded").x))


# ---------------------------------------------------------------------------
# 3. Departures implied by position, floor and scale
# ---------------------------------------------------------------------------

def _grid(lo: float, hi: float, k: int = 25) -> np.ndarray:
    return np.exp(np.linspace(np.log(lo), np.log(hi), k + 1)[:-1] + np.log(hi / lo) / (2 * k))


RULES = ("down", "nearest_950", "nearest_1000")


def exit_prob(lo: int, d: ChangeDist, s: float, rule: str, m100: float) -> float:
    """Probability of falling off the list for a fortune reported in band ``lo``.

    The share ``m100`` of fortunes reported in exact multiples of $100 million sits at the band's lower edge; the
    true value behind the report depends on Forbes's rounding and inclusion rule, which is not published:

    * ``down``: true value in [lo, lo + width), line at $1bn. This is Part 4's assumption for every band.
    * ``nearest_950``: rounded to the nearest $100 million, true value in [lo - 50, lo + 50), and the next report
      falls below $1bn when the true value falls below $950 million.
    * ``nearest_1000``: rounded to the nearest $100 million but listed only from a true $1bn, so the true value is
      in [max(lo - 50, 1000), lo + 50) and the line is at $1bn.

    Fortunes not reported in multiples of $100 million are spread evenly in log within the band, line at $1bn."""
    m = d.quantile(0.5)
    scaled = lambda c: d.cdf(m + (c - m) / s)  # noqa: E731
    p_band = float(np.mean(scaled(np.log10(1000.0 / _grid(float(lo), float(EDGE_HI[lo]))))))
    if rule == "down":
        return p_band
    if rule == "nearest_950":
        p_edge = float(np.mean(scaled(np.log10(950.0 / _grid(lo - 50.0, lo + 50.0)))))
    else:
        p_edge = float(np.mean(scaled(np.log10(1000.0 / _grid(max(lo - 50.0, 1000.0), lo + 50.0)))))
    return m100 * p_edge + (1 - m100) * p_band


def age_floor(glob: pd.DataFrame) -> dict[str, Any]:
    """Departures unrelated to market value: the exit rate by age above $3bn, applied to each band's age mix."""
    a = glob[glob.table.isin(["exit_age", "stay_age"]) & glob.year.between(FIRST, LAST)].copy()
    a["age"] = a.key.astype(int)
    far = a[a.band_lo_musd >= FAR].groupby(["age", "table"]).n.sum().unstack(fill_value=0)
    rate = far.exit_age / (far.exit_age + far.stay_age)
    out = {"far_rate_by_age": {str(k): float(v) for k, v in rate.items()}}
    near = a[(a.band_lo_musd >= 1000) & (a.band_lo_musd < NEAR)].groupby("age").n.sum()
    out["near_floor"] = float((near * rate.reindex(near.index)).sum() / near.sum())
    out["far_floor"] = float((far.exit_age.sum()) / (far.exit_age + far.stay_age).sum())
    out["near_age_mix"] = {str(k): float(v / near.sum()) for k, v in near.items()}
    by_band = {}
    for b in BANDS:
        nb = a[a.band_lo_musd == b].groupby("age").n.sum()
        by_band[str(b)] = float((nb * rate.reindex(nb.index)).sum() / nb.sum())
    out["by_band"] = by_band
    return out


def predicted(tenure: pd.DataFrame, dists: dict[int, ChangeDist], bands: Sequence[int], years: Sequence[int],
              scales: dict[int, float], floors: dict[int, float], rule: str, m100: float,
              tenure_sel: Any = None) -> dict[str, float]:
    exp = obs = stock = 0.0
    for y in years:
        d = tenure[(tenure.year == y)]
        if tenure_sel is not None:
            d = d[tenure_sel(d)]
        for b in bands:
            st = float(d[(d.table == "stock") & (d.band_lo_musd == b)].n.sum())
            if not st:
                continue
            p = exit_prob(b, dists[y], scales[b], rule, m100)
            h = floors[b]
            exp += st * (h + (1 - h) * p)
            obs += float(d[(d.table == "exit") & (d.band_lo_musd == b)].n.sum())
            stock += st
    return {"stock": stock, "observed": obs, "predicted": exp, "obs_rate": obs / stock, "pred_rate": exp / stock}


def decomposition(tenure: pd.DataFrame, dists: dict[int, ChangeDist], m100: float, floor: dict[str, Any],
                  scale_near: float, years: Sequence[int]) -> dict[str, Any]:
    near = [b for b in BANDS if b < NEAR]
    one, zero = {b: 1.0 for b in near}, {b: 0.0 for b in near}
    age = {b: floor["by_band"][str(b)] for b in near}
    sc = {b: scale_near for b in near}
    out = {"part4": predicted(tenure, dists, near, years, one, zero, "down", m100),
           "floor": predicted(tenure, dists, near, years, one, age, "down", m100),
           "volatility": predicted(tenure, dists, near, years, sc, age, "down", m100)}
    out["rounding_rules"] = {r: predicted(tenure, dists, near, years, sc, age, r, m100)["pred_rate"] for r in RULES}
    out["rounding_rules_scale1"] = {r: predicted(tenure, dists, near, years, one, age, r, m100)["pred_rate"]
                                    for r in RULES}
    out["observed_rate"] = out["part4"]["obs_rate"]
    return out


# ---------------------------------------------------------------------------
# 4. Tenure and the previous year's change
# ---------------------------------------------------------------------------

def newcomer_excess(tenure: pd.DataFrame, bands: Sequence[int], years: Sequence[int]) -> float:
    """Departures of new and returning people beyond the incumbents' rate in the same band and year."""
    excess = 0.0
    for b in bands:
        for y in years:
            d = tenure[(tenure.year == y) & (tenure.band_lo_musd == b)]
            inc = d[d.tenure == "incumbent"]
            si, ei = inc[inc.table == "stock"].n.sum(), inc[inc.table == "exit"].n.sum()
            for tt in ("new", "returning"):
                x = d[d.tenure == tt]
                sn, en = x[x.table == "stock"].n.sum(), x[x.table == "exit"].n.sum()
                if si and sn:
                    excess += float(en - sn * ei / si)
    return excess


def by_tenure(tenure: pd.DataFrame, dists: dict[int, ChangeDist], m100: float, floor: dict[str, Any]) -> dict[str, Any]:
    yrs = list(range(FIRST + 1, LAST + 1))
    t = tenure[tenure.year.isin(yrs)]
    near = [b for b in BANDS if b < NEAR]
    out: dict[str, Any] = {}
    for grp, sel in (("near", lambda b: (b >= 1000) & (b < NEAR)), ("middle", lambda b: (b >= NEAR) & (b < FAR)),
                     ("far", lambda b: b >= FAR)):
        g = {}
        for ten in ("incumbent", "new", "returning"):
            d = t[sel(t.band_lo_musd) & (t.tenure == ten)]
            st, ex = int(d[d.table == "stock"].n.sum()), int(d[d.table == "exit"].n.sum())
            lc = d[d.table == "logchange"].assign(lo=lambda z: z.key.astype(int))
            g[ten] = {"stock": st, "exit": ex, "exit_rate": ex / st,
                      "up_0.02": float(lc[lc.lo >= 20].n.sum() / st), "up_0.1": float(lc[lc.lo >= 100].n.sum() / st),
                      "up_0.2": float(lc[lc.lo >= 200].n.sum() / st)}
        out[grp] = g
    for ten in ("incumbent", "new"):
        cells = cell_counts(t, yrs, lambda b: (b >= 1000) & (b < NEAR), lambda d, ten=ten: d.tenure == ten)
        sc = fit_scale(cells, dists, floor["near_floor"])
        pr = predicted(t, dists, near, yrs, {b: sc for b in near}, {b: floor["by_band"][str(b)] for b in near},
                       "down", m100, tenure_sel=lambda d, ten=ten: d.tenure == ten)
        out["near"][ten].update(scale=sc, pred_rate=pr["pred_rate"])
    # Incumbents near the line: which rounding rule fits, band by band, with their own scale and the age floor.
    inc_sel = lambda d: d.tenure == "incumbent"  # noqa: E731
    sc_inc = out["near"]["incumbent"]["scale"]
    rules = {}
    for r in RULES:
        bands = {}
        for b in near:
            pr = predicted(t, dists, [b], yrs, {b: sc_inc}, {b: floor["by_band"][str(b)]}, r, m100, tenure_sel=inc_sel)
            bands[str(b)] = {"obs": pr["obs_rate"], "pred": pr["pred_rate"]}
        tot = predicted(t, dists, near, yrs, {b: sc_inc for b in near}, {b: floor["by_band"][str(b)] for b in near},
                        r, m100, tenure_sel=inc_sel)
        rules[r] = {"pred_rate": tot["pred_rate"], "obs_rate": tot["obs_rate"], "bands": bands,
                    "mean_abs_error_points": float(100 * np.mean([abs(v["obs"] - v["pred"]) for v in bands.values()]))}
    out["incumbent_rules"] = rules
    st = out["near"]
    out["near_new_share"] = st["new"]["stock"] / sum(st[k]["stock"] for k in ("incumbent", "new", "returning"))
    within = {}
    for b in near:
        d = t[t.band_lo_musd == b]
        r = {}
        for ten in ("incumbent", "new"):
            x = d[d.tenure == ten]
            stck, ex = int(x[x.table == "stock"].n.sum()), int(x[x.table == "exit"].n.sum())
            r[ten] = ex / stck
        within[str(b)] = r
    out["within_band"] = within
    return out


def by_previous(tenure: pd.DataFrame) -> dict[str, Any]:
    t = tenure[tenure.year.between(FIRST + 1, LAST) & (tenure.tenure == "incumbent")]
    out: dict[str, Any] = {}
    for grp, sel in (("near", lambda b: (b >= 1000) & (b < NEAR)), ("far", lambda b: b >= FAR)):
        g = {}
        for prev in ("fell >0.1", "fell 0.03-0.1", "flat", "rose 0.03-0.1", "rose >0.1"):
            d = t[sel(t.band_lo_musd) & (t.prev == prev)]
            st, ex = int(d[d.table == "stock"].n.sum()), int(d[d.table == "exit"].n.sum())
            lc = d[d.table == "logchange"].assign(lo=lambda z: z.key.astype(int))
            g[prev] = {"stock": st, "exit_rate": ex / st, "up_0.1": float(lc[lc.lo >= 100].n.sum() / st),
                       "down_0.1": float(lc[lc.lo < -100].n.sum() / st), "up_0.02": float(lc[lc.lo >= 20].n.sum() / st)}
        out[grp] = g
    f = out["far"]
    big = [f["fell >0.1"], f["rose >0.1"]]
    out["far_move_0.1_after_big"] = float(sum((v["up_0.1"] + v["down_0.1"]) * v["stock"] for v in big)
                                          / sum(v["stock"] for v in big))
    out["far_move_0.1_after_flat"] = float(f["flat"]["up_0.1"] + f["flat"]["down_0.1"])
    return out


# ---------------------------------------------------------------------------
# 5. Bootstrap over years
# ---------------------------------------------------------------------------

def bootstrap(tenure: pd.DataFrame, dists: dict[int, ChangeDist], years: Sequence[int], m100: float,
              floor: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(20261112)
    near = [b for b in BANDS if b < NEAR]
    age = {b: floor["by_band"][str(b)] for b in near}
    cells_all = cell_counts(tenure, years, lambda b: (b >= 1000) & (b < NEAR))
    per_year = {}
    for y in years:
        per_year[y] = {"p4": predicted(tenure, dists, near, [y], {b: 1.0 for b in near}, {b: 0.0 for b in near},
                                       "down", m100),
                       "floor": predicted(tenure, dists, near, [y], {b: 1.0 for b in near}, age, "down", m100),
                       "new": newcomer_excess(tenure, near, [y]) if y > FIRST else 0.0}
    scales, vol_pts, vol_share, new_share, rest_share = [], [], [], [], []
    for _ in range(N_BOOT):
        ys = list(rng.choice(years, size=len(years), replace=True))
        s = fit_scale({i: cells_all[y] for i, y in enumerate(ys)}, {i: dists[y] for i, y in enumerate(ys)},
                      floor["near_floor"])
        scales.append(s)
        stock = sum(per_year[y]["p4"]["stock"] for y in ys)
        obs = sum(per_year[y]["p4"]["observed"] for y in ys)
        p4 = sum(per_year[y]["p4"]["predicted"] for y in ys)
        fl = sum(per_year[y]["floor"]["predicted"] for y in ys)
        vol = sum(predicted(tenure, dists, near, [y], {b: s for b in near}, age, "down", m100)["predicted"] for y in ys)
        new = sum(per_year[y]["new"] for y in ys)
        gap = obs - p4
        vol_pts.append(100 * (vol - fl) / stock)
        vol_share.append((vol - fl) / gap)
        new_share.append(new / gap)
        rest_share.append((obs - vol - new) / gap)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"scale_near_ci95": q(scales), "volatility_points_ci95": q(vol_pts), "volatility_share_ci95": q(vol_share),
            "newcomer_share_ci95": q(new_share), "residual_share_ci95": q(rest_share), "n_boot": N_BOOT}


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------

def compute(inp: dict[str, Any]) -> dict[str, Any]:
    tenure, glob = inp["forbes_tenure"], inp["forbes_global"]
    width = float(inp["forbes_flows_manifest"]["logchange_bin"])
    years = list(range(FIRST, LAST + 1))
    check = check_against_part4(tenure, glob)
    rnd = rounding(tenure)
    m100 = rnd["below_2000"]["m100"]
    dists = far_dists(tenure, width, years)
    floor = age_floor(glob)
    scales, tails = {}, {}
    for b in BANDS:
        cells = cell_counts(tenure, years, lambda x, b=b: x == b)
        scales[str(b)] = fit_scale(cells, dists, floor["by_band"][str(b)])
        tails[str(b)] = upper_tail(cells, dists, UPPER_X)
    near_cells = cell_counts(tenure, years, lambda x: (x >= 1000) & (x < NEAR))
    scale_near = fit_scale(near_cells, dists, floor["near_floor"])
    scale_middle = fit_scale(cell_counts(tenure, years, lambda x: (x >= NEAR) & (x < FAR)), dists,
                             float(np.mean([floor["by_band"][str(b)] for b in BANDS if NEAR <= b < FAR])))
    scale_far_self = fit_scale(cell_counts(tenure, years, lambda x: x >= FAR), dists, floor["far_floor"])
    dec = decomposition(tenure, dists, m100, floor, scale_near, years)
    near_b = [b for b in BANDS if b < NEAR]
    stock, obs = dec["part4"]["stock"], dec["part4"]["observed"]
    gap = obs - dec["part4"]["predicted"]
    new = newcomer_excess(tenure, near_b, range(FIRST + 1, LAST + 1))
    pieces = {"floor": dec["floor"]["predicted"] - dec["part4"]["predicted"],
              "volatility": dec["volatility"]["predicted"] - dec["floor"]["predicted"],
              "newcomers": new}
    pieces["residual"] = obs - dec["volatility"]["predicted"] - new
    gap_out = {"points": 100 * gap / stock,
               "pieces": {k: {"points": 100 * v / stock, "share_of_gap": v / gap} for k, v in pieces.items()}}
    yearly = {}
    for y in years:
        a = predicted(tenure, dists, near_b, [y], {b: 1.0 for b in near_b}, {b: 0.0 for b in near_b}, "down", m100)
        z = predicted(tenure, dists, near_b, [y], {b: scale_near for b in near_b},
                      {b: floor["by_band"][str(b)] for b in near_b}, "down", m100)
        yearly[str(y)] = {"obs": a["obs_rate"], "part4": a["pred_rate"], "model": z["pred_rate"]}
    ob = np.array([v["obs"] for v in yearly.values()])
    fit = {"by_year": yearly,
           "mae_part4_points": float(100 * np.mean(np.abs(ob - np.array([v["part4"] for v in yearly.values()])))),
           "mae_model_points": float(100 * np.mean(np.abs(ob - np.array([v["model"] for v in yearly.values()])))),
           "corr_part4": float(np.corrcoef(ob, [v["part4"] for v in yearly.values()])[0, 1]),
           "corr_model": float(np.corrcoef(ob, [v["model"] for v in yearly.values()])[0, 1])}
    return {
        "yearly_fit": fit,
        "check_part4": check,
        "fetch": {"years": {y: e.get("n_persons") for y, e in inp["forbes_tenure_manifest"]["years"].items()},
                  "same_as_part4": sum(bool(e.get("same_as_part4")) for e in inp["forbes_tenure_manifest"]["years"].values())},
        "rounding": rnd,
        "far_median_change": {str(y): dists[y].quantile(0.5) for y in years},
        "scales_by_band": scales, "scale_near": scale_near, "scale_middle": scale_middle,
        "scale_far_self": scale_far_self,
        "upper_tail_by_band": tails, "upper_tail_near": upper_tail(near_cells, dists, UPPER_X),
        "floor": floor, "decomposition": dec, "gap": gap_out,
        "tenure": by_tenure(tenure, dists, m100, floor),
        "previous": by_previous(tenure),
        "bootstrap": bootstrap(tenure, dists, years, m100, floor),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def frame(fig: plt.Figure, title: str, subtitle: str, **kw: Any) -> None:
    _frame(fig, title, subtitle, source=SOURCE, **kw)


def _bn(v: float, _: Any = None) -> str:
    return f"${v / 1000:g}bn"


def fig_upper_tail(n: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.array([np.sqrt(b * EDGE_HI[b]) for b in BANDS])
    for k, col, lab in (("0.1", BLUE, "Rise of 26% or more"), ("0.2", ORANGE, "Rise of 58% or more")):
        r = np.array([n["upper_tail_by_band"][str(b)][k]["ratio"] for b in BANDS])
        lo = np.array([n["upper_tail_by_band"][str(b)][k]["ci95"][0] for b in BANDS])
        hi = np.array([n["upper_tail_by_band"][str(b)][k]["ci95"][1] for b in BANDS])
        off = 0.985 if k == "0.1" else 1.015
        ax.vlines(x * off, lo, hi, color=col, lw=1.8, alpha=0.8)
        ax.plot(x * off, r, "o", color=col, ms=7, label=lab)
    ax.axhline(1, color=BASELINE, lw=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1000, 1500, 2000, 3000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(_bn))
    ax.set_xlim(950, 3800)
    ax.set_ylim(0.6, 2.6)
    ax.yaxis.set_major_locator(FixedLocator([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}×"))
    ax.set_xlabel("Net worth on this year's list (log scale)")
    ax.set_ylabel("Observed ÷ expected from fortunes above $3bn")
    ax.legend(loc="upper right", fontsize=9.5)
    frame(fig, "Near the line, large rises are somewhat more common",
          "People in each band whose fortune rose by at least the stated amount by the next list, relative to the share\n"
          "expected from the same year's fortunes above $3bn. Lists 2010-2024 pooled, 95% intervals.",
          top=0.83, bottom=0.14, left=0.1)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_gap(n: dict[str, Any], out: Path) -> None:
    g = n["gap"]["pieces"]
    d = n["decomposition"]
    start = 100 * d["part4"]["pred_rate"]
    steps = [("Part 4 crossing model", start, None),
             ("Deaths and reassessments (age-adjusted)", g["floor"]["points"], ORANGE),
             ("Higher volatility near the line", g["volatility"]["points"], BLUE),
             ("Newcomers leaving more than incumbents", g["newcomers"]["points"], AQUA),
             ("Not explained", g["residual"]["points"], MUTED)]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    level = 0.0
    for i, (lab, v, col) in enumerate(steps):
        if col is None:
            ax.barh(i, v, color=SECONDARY, height=0.6)
            level = v
            ax.text(v + 0.2, i, f"{fmt(v, 1)}%", va="center", fontsize=10, color=INK)
            continue
        left = level if v >= 0 else level + v
        ax.barh(i, abs(v), left=left, color=col, height=0.6)
        ax.text(max(level, level + v) + 0.2, i, f"{fmt(v, 1, sign=True)} points", va="center", fontsize=10, color=INK)
        level += v
    obs = 100 * d["observed_rate"]
    ax.barh(len(steps), obs, color=INK, height=0.6)
    ax.text(obs + 0.2, len(steps), f"{fmt(obs, 1)}%", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(steps) + 1))
    ax.set_yticklabels([s[0] for s in steps] + ["Observed"])
    ax.invert_yaxis()
    ax.set_xlim(0, 28)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Share of people below $1.5bn not on the next list, a year")
    ax.grid(axis="y", visible=False)
    frame(fig, "From Part 4's prediction to what happened, one ingredient at a time",
          "Departure rate below $1.5bn, lists 2010-2024. Each step adds one ingredient to the crossing model; the\n"
          "newcomer step adds the departures of new and returning people above the incumbents' rate in the same band.",
          top=0.83, bottom=0.17, left=0.34)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_tenure(n: dict[str, Any], out: Path) -> None:
    t = n["tenure"]
    groups = [("near", "Below $1.5bn"), ("middle", "$1.5bn to $3bn"), ("far", "$3bn and above")]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(groups))
    for off, ten, col, lab in ((-0.2, "incumbent", BLUE, "On the previous list"), (0.2, "new", ORANGE, "First time on any list")):
        v = [100 * t[g][ten]["exit_rate"] for g, _ in groups]
        ax.bar(x + off, v, width=0.38, color=col, label=lab)
        pr = 100 * t["near"][ten]["pred_rate"]
        for i, (xi, vi) in enumerate(zip(x + off, v)):
            top = max(vi, pr) if i == 0 else vi
            ax.text(xi, top + 0.7, f"{fmt(vi, 1)}%", ha="center", fontsize=9.5, color=INK)
        ax.plot([off - 0.17, off + 0.17], [pr, pr], color=INK, lw=2.4,
                label="Crossing model with the group's own volatility" if ten == "new" else None)
    ax.set_xticks(x)
    ax.set_xticklabels([g[1] for g in groups])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Share not on the next list")
    ax.set_ylim(0, 38)
    ax.legend(loc="upper right", fontsize=9.5)
    frame(fig, "Newcomers leave more often at every level of wealth",
          "Departure rates by tenure and net worth, lists 2011-2024 (tenure needs the previous list). Black lines: the\n"
          "crossing model with each group's volatility, estimated from its rises, and the age-adjusted floor.",
          top=0.83, bottom=0.12, left=0.08)
    fig.savefig(out, dpi=250)
    plt.close(fig)


FIGURE_FILES = ("fig1_upper_tail.png", "fig2_gap.png", "fig3_tenure.png")


def render(n: dict[str, Any], outdir: Path) -> None:
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    fig_upper_tail(n, outdir / FIGURE_FILES[0])
    fig_gap(n, outdir / FIGURE_FILES[1])
    fig_tenure(n, outdir / FIGURE_FILES[2])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary_lines(n: dict[str, Any]) -> list[str]:
    r3 = lambda d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    out = [f"Check against Part 4: {n['check_part4']}; fetch {n['fetch']}",
           f"Rounding: {({k: r3(v) if isinstance(v, dict) else round(v, 3) for k, v in n['rounding'].items()})}",
           f"Far median change: {r3(n['far_median_change'])}",
           f"Scale near {n['scale_near']:.3f}; middle {n['scale_middle']:.3f}; far self-check {n['scale_far_self']:.3f}; "
           f"by band {r3(n['scales_by_band'])}"]
    for b, v in n["upper_tail_by_band"].items():
        out.append(f"  upper tail {b}: " + "; ".join(f"x{x} {r['ratio']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] "
                                                   f"(o {r['observed']:.0f}, e {r['expected']:.1f})" for x, r in v.items()))
    out.append("  upper tail near: " + "; ".join(f"x{x} {r['ratio']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
                                                 for x, r in n["upper_tail_near"].items()))
    f = n["floor"]
    out.append(f"Floor: near (age-standardised) {f['near_floor']:.4f}; far {f['far_floor']:.4f}; by band {r3(f['by_band'])}; "
               f"far rate by age {r3(f['far_rate_by_age'])}; near age mix {r3(f['near_age_mix'])}")
    for k, v in n["decomposition"].items():
        out.append(f"Decomposition {k}: {v if isinstance(v, float) else r3(v)}")
    out.append(f"  scale_far_self {n['scale_far_self']:.3f}")
    out.append(f"Gap {n['gap']['points']:.2f} points; pieces " + "; ".join(
        f"{k} {v['points']:+.2f} pts ({v['share_of_gap']:.3f})" for k, v in n["gap"]["pieces"].items()))
    t = n["tenure"]
    for g in ("near", "middle", "far"):
        out.append(f"Tenure {g}: " + "; ".join(f"{k} {r3(v)}" for k, v in t[g].items()))
    out.append(f"  near new share {t['near_new_share']:.3f}; within band {t['within_band']}")
    for r, v in t["incumbent_rules"].items():
        out.append(f"  incumbents rule {r}: pred {v['pred_rate']:.3f} obs {v['obs_rate']:.3f} MAE {v['mean_abs_error_points']:.2f} pts; "
                   + "; ".join(f"{b} {x['obs']:.3f}/{x['pred']:.3f}" for b, x in v["bands"].items()))
    yf = n["yearly_fit"]
    out.append(f"Yearly near-line fit: MAE Part 4 {yf['mae_part4_points']:.2f} pts, model {yf['mae_model_points']:.2f} pts; "
               f"corr {yf['corr_part4']:.3f} / {yf['corr_model']:.3f}")
    p = n["previous"]
    for g in ("near", "far"):
        out.append(f"Previous {g}: " + "; ".join(f"{k} {r3(v)}" for k, v in p[g].items()))
    out.append(f"  far |move|>=0.1 after big {p['far_move_0.1_after_big']:.3f}, after flat {p['far_move_0.1_after_flat']:.3f}")
    out.append(f"Bootstrap: {r3({k: v for k, v in n['bootstrap'].items() if not isinstance(v, list)})} "
               f"{ {k: [round(x, 3) for x in v] for k, v in n['bootstrap'].items() if isinstance(v, list)} }")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    inp = load_inputs()
    numbers = json.loads(json.dumps(compute(inp)))
    lines_out = summary_lines(numbers)
    print("\n".join(lines_out))
    target = RESULTS / "numbers.json"
    if args.check:
        problems: list[str] = []
        _compare(numbers, json.loads(target.read_text(encoding="utf-8")), "numbers", problems)
        if problems:
            print("\nREPRODUCTION FAILED:\n" + "\n".join(problems[:50]), file=sys.stderr)
            return 1
        print("\nREPRODUCED: every number matches results/part5/numbers.json")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "summary.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    if not args.no_figures:
        render(numbers, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
