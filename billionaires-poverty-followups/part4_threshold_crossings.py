"""Part 4: how much of a country's year-to-year change in billionaires is fortunes crossing $1 billion?

Part 3 conjectured that most year-to-year changes in a country's billionaire
count come from fortunes near the $1 billion threshold crossing it in either
direction. This script tests that on the Forbes lists of 2010-2025 linked year
to year (aggregates written by ``fetch_forbes_flows.py``):

* where entries and exits sit relative to the threshold, against the stock;
* how much of the variance of country-level changes the near-threshold flows
  account for, as a function of where "near" ends;
* how often people near the line leave and come back;
* whether a simple crossing model (the year's moves of large fortunes applied
  to the stock near the line) reproduces the exits and entries;
* what the answer implies for billionaire counts as an indicator, and whether
  a less noisy count changes Part 3's within-country poverty estimate.

Every number quoted in the Part 4 article is computed here and written to
``results/part4/numbers.json``; ``--check`` recomputes everything and fails if
any number differs. Figures go to ``figures/part4/``.

Usage::

    python part4_threshold_crossings.py           # print, write results/part4 and figures/part4
    python part4_threshold_crossings.py --check   # fail if any number differs from numbers.json

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
import statsmodels.formula.api as smf  # noqa: E402
from scipy.optimize import brentq, minimize_scalar  # noqa: E402

from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import part3_within_country as p3  # noqa: E402
from part1_poverty_lines import (AQUA, BASELINE, BLUE, INK, MUTED, ORANGE, SECONDARY,  # noqa: E402
                                 _compare, _frame, _style, fmt)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results" / "part4"
FIGURES = HERE / "figures" / "part4"

FIRST, LAST = 2010, 2025
CUTS = [1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000]
NEAR = 1500        # main definition of "near the threshold": below $1.5 billion
NEAR_WIDE = 2000   # wider definition: below $2 billion
FAR = 3000         # fortunes from which a year's move rarely reaches $1 billion
FIXED = [1000, 1500, 2000, 3000, 5000]
SOURCE = ("Data: Forbes annual billionaire lists 2010-2025, linked year to year; only aggregate counts are stored. "
          "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/billionaires-poverty-followups")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> dict[str, Any]:
    m = json.loads((DATA / "forbes_flows_manifest.json").read_text(encoding="utf-8"))
    for name, key in (("forbes_bands.csv.gz", "bands_sha256"), ("forbes_flows.csv.gz", "flows_sha256"),
                      ("forbes_global.csv.gz", "global_sha256")):
        if _sha(DATA / name) != m[key]:
            raise ValueError(f"{name} does not match forbes_flows_manifest.json")
    read = lambda name: pd.read_csv(DATA / name, keep_default_na=False)  # noqa: E731
    return {"bands": read("forbes_bands.csv.gz"), "flows": read("forbes_flows.csv.gz"),
            "global": read("forbes_global.csv.gz"), "manifest": m, "p3": p3.load_inputs()}


def check_against_part3(bands: pd.DataFrame, history: pd.DataFrame) -> dict[str, Any]:
    """The linked lists must give exactly Part 3's country totals."""
    h = history[(history.field == "citizenship") & history.year.between(FIRST, LAST)]
    a = bands.groupby(["year", "country_label"])[["n", "worth_musd"]].sum()
    b = h.set_index(["year", "country_label"])[["n", "worth_musd"]]
    j = a.join(b, rsuffix="_p3", how="outer").fillna(0)
    bad_n = int((j.n != j.n_p3).sum())
    bad_w = float((j.worth_musd - j.worth_musd_p3).abs().max())
    if bad_n or bad_w > 1e-6:
        raise ValueError(f"linked lists disagree with Part 3 totals: {bad_n} counts, worth diff {bad_w}")
    return {"country_years": int(len(j)), "count_mismatches": bad_n, "max_worth_diff": bad_w}


# ---------------------------------------------------------------------------
# Helpers: bands, change distributions
# ---------------------------------------------------------------------------

def _hi(edges: list[int]) -> dict[int, float]:
    out = {lo: float(hi) for lo, hi in zip(edges, edges[1:])}
    out[edges[-1]] = float(edges[-1]) * 10.0
    return out


def above(n: np.ndarray, lo: np.ndarray, hi: np.ndarray, t: float) -> float:
    """Count above t, assuming fortunes are spread uniformly in log within each band."""
    frac = np.clip(np.log(hi / t) / np.log(hi / lo), 0.0, 1.0)
    frac = np.where(lo >= t, 1.0, np.where(hi <= t, 0.0, frac))
    return float(np.sum(n * frac))


def pareto_alpha(lo: np.ndarray, hi: np.ndarray, n: np.ndarray, open_top: np.ndarray, x0: float = 1000.0) -> float:
    """Grouped maximum likelihood for a Pareto tail above x0."""
    def nll(a: float) -> float:
        p = np.where(open_top, (lo / x0) ** -a, (lo / x0) ** -a - (hi / x0) ** -a)
        return -float(np.sum(n * np.log(np.maximum(p, 1e-300))))
    return float(minimize_scalar(nll, bounds=(0.2, 4.0), method="bounded").x)


class ChangeDist:
    """Empirical distribution of annual changes in log10 net worth (bins of 0.02, uniform within a bin)."""

    def __init__(self, lows: np.ndarray, counts: np.ndarray, width: float):
        order = np.argsort(lows)
        self.lo, self.p = lows[order], counts[order] / counts.sum()
        self.w = width
        self.cum = np.concatenate([[0.0], np.cumsum(self.p)])

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, float))
        k = np.searchsorted(self.lo, x, side="right") - 1
        inside = np.clip((x - self.lo[np.clip(k, 0, None)]) / self.w, 0, 1)
        out = np.where(k < 0, 0.0, self.cum[np.clip(k, 0, None)] + self.p[np.clip(k, 0, None)] * inside)
        return np.clip(out, 0, 1)

    def quantile(self, q: float) -> float:
        return float(brentq(lambda x: float(self.cdf(x)[0]) - q, -1.0, 1.1))

    def mids(self) -> np.ndarray:
        return self.lo + self.w / 2


def change_dist(glob: pd.DataFrame, year: int, width: float, min_band: int = FAR) -> ChangeDist:
    lc = glob[(glob.table == "logchange") & (glob.year == year) & (glob.band_lo_musd >= min_band)]
    g = lc.assign(key=lc.key.astype(int)).groupby("key").n.sum()
    return ChangeDist(g.index.to_numpy(float) / 1000.0, g.to_numpy(float), width)


# ---------------------------------------------------------------------------
# 1. The stock and the gross flows
# ---------------------------------------------------------------------------

def stock(bands: pd.DataFrame, edges: list[int]) -> dict[str, Any]:
    hi = _hi(edges)
    out: dict[str, Any] = {"by_year": {}}
    for year, g in bands[bands.band_lo_musd >= 1000].groupby("year"):
        s = g.groupby("band_lo_musd").n.sum()
        lo = s.index.to_numpy(float)
        his = np.array([hi[int(x)] for x in lo])
        alpha = pareto_alpha(lo, his, s.to_numpy(float), lo == edges[-1])
        tot = float(s.sum())
        out["by_year"][str(year)] = {
            "n": int(tot), "alpha": alpha,
            "share_below": {str(c): float(s[s.index < c].sum() / tot) for c in (1500, 2000, 3000)},
        }
    by = out["by_year"]
    out["alpha_range"] = [min(v["alpha"] for v in by.values()), max(v["alpha"] for v in by.values())]
    out["share_below_1500_range"] = [min(v["share_below"]["1500"] for v in by.values()),
                                     max(v["share_below"]["1500"] for v in by.values())]
    return out


def gross_flows(flows: pd.DataFrame, bands: pd.DataFrame) -> dict[str, Any]:
    f = flows
    tot = f.groupby(["year", "flow"]).n.sum().unstack(fill_value=0)
    n_t = bands.groupby("year").n.sum()
    pairs = {}
    for y, r in tot.iterrows():
        pairs[str(int(y))] = {"entries": int(r.get("entry", 0)), "exits": int(r.get("exit", 0)),
                              "switches": int(r.get("switch_out", 0)), "n_t": int(n_t[y]), "n_t1": int(n_t[y + 1])}
    stock_t = f[f.flow.isin(["stay", "exit", "switch_out"])].groupby("band_lo_musd").n.sum()
    exits = f[f.flow == "exit"].groupby("band_lo_musd").n.sum()
    entries = f[f.flow == "entry"].groupby("band_lo_musd").n.sum()
    rate = (exits.reindex(stock_t.index, fill_value=0) / stock_t)
    near = {}
    for c in CUTS:
        near[str(c)] = {"entries": float(entries[entries.index < c].sum() / entries.sum()),
                        "exits": float(exits[exits.index < c].sum() / exits.sum()),
                        "stock_t": float(stock_t[stock_t.index < c].sum() / stock_t.sum())}
    sum_e, sum_x = int(entries.sum()), int(exits.sum())
    return {
        "pairs": pairs, "total_entries": sum_e, "total_exits": sum_x,
        "total_switches": int(f[f.flow == "switch_out"].n.sum()),
        "person_years_at_risk": int(stock_t.sum()),
        "exit_rate_by_band": {str(int(k)): float(v) for k, v in rate.items() if k >= 1000},
        "exit_rate_far": float(exits[exits.index >= FAR].sum() / stock_t[stock_t.index >= FAR].sum()),
        "exit_rate_near": float(exits[(exits.index >= 1000) & (exits.index < NEAR)].sum()
                                / stock_t[(stock_t.index >= 1000) & (stock_t.index < NEAR)].sum()),
        "near_shares": near,
        "entries_by_band": {str(int(k)): int(v) for k, v in entries.items()},
        "exits_by_band": {str(int(k)): int(v) for k, v in exits.items()},
    }


def global_decomposition(flows: pd.DataFrame, cut: int = NEAR) -> dict[str, Any]:
    f = flows.copy()
    out = {}
    for y, g in f.groupby("year"):
        e, x = g[g.flow == "entry"], g[g.flow == "exit"]
        out[str(int(y))] = {
            "entries_near": int(e[e.band_lo_musd < cut].n.sum()), "entries_far": int(e[e.band_lo_musd >= cut].n.sum()),
            "exits_near": int(x[x.band_lo_musd < cut].n.sum()), "exits_far": int(x[x.band_lo_musd >= cut].n.sum()),
            "net_switch": int(g[g.flow == "switch_in"].n.sum() - g[g.flow == "switch_out"].n.sum()),
        }
        r = out[str(int(y))]
        r["net"] = r["entries_near"] + r["entries_far"] - r["exits_near"] - r["exits_far"] + r["net_switch"]
        r["net_near"] = r["entries_near"] - r["exits_near"]
        r["net_far"] = r["entries_far"] - r["exits_far"]
    nets = np.array([v["net"] for v in out.values()], float)
    near = np.array([v["net_near"] for v in out.values()], float)
    far = np.array([v["net_far"] for v in out.values()], float)
    return {"by_year": out,
            "var_share_near": float(np.cov(near, nets)[0, 1] / nets.var(ddof=1)),
            "var_share_far": float(np.cov(far, nets)[0, 1] / nets.var(ddof=1)),
            "sum_net_near": int(near.sum()), "sum_net_far": int(far.sum()), "sum_net": int(nets.sum())}


# ---------------------------------------------------------------------------
# 2. Country-level changes: how much is near the line?
# ---------------------------------------------------------------------------

def country_components(flows: pd.DataFrame, bands: pd.DataFrame, cut: int) -> pd.DataFrame:
    f = flows.copy()
    f["part"] = np.where(f.flow.isin(["switch_in", "switch_out"]), "switch",
                         np.where(f.band_lo_musd < cut, "near", "far"))
    sign = {"entry": 1, "switch_in": 1, "exit": -1, "switch_out": -1, "stay": 0}
    f["v"] = f.n * f.flow.map(sign)
    c = f.groupby(["year", "country_label", "part"]).v.sum().unstack(fill_value=0)
    for k in ("near", "far", "switch"):
        if k not in c:
            c[k] = 0
    c = c[["near", "far", "switch"]].copy()
    n = bands.groupby(["year", "country_label"]).n.sum()
    c["n0"] = [n.get((y, k), 0) for y, k in c.index]
    c["n1"] = [n.get((y + 1, k), 0) for y, k in c.index]
    c["d"] = c.near + c.far + c.switch
    if not (c.d == c.n1 - c.n0).all():
        raise ValueError("flows do not add up to the change in counts")
    c = c[(c.n0 > 0) | (c.n1 > 0)]
    return c


def _shares(c: pd.DataFrame, relative: bool, demean: bool) -> dict[str, float]:
    x = c.copy()
    if relative:
        den = (x.n0 + x.n1) / 2 + 0.5
        for k in ("near", "far", "switch", "d"):
            x[k] = x[k] / den
    if demean:
        for k in ("near", "far", "switch", "d"):
            x[k] = x[k] - x.groupby(level="year")[k].transform("mean")
    v = float(x.d.var(ddof=1))
    return {k: float(np.cov(x[k], x.d)[0, 1] / v) for k in ("near", "far", "switch")}


def country_decomposition(flows: pd.DataFrame, bands: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"curve": {}}
    for cut in CUTS:
        c = country_components(flows, bands, cut)
        out["curve"][str(cut)] = {"abs": _shares(c, False, False), "rel": _shares(c, True, False),
                                  "rel_demeaned": _shares(c, True, True)}
    for cut in (NEAR, NEAR_WIDE):
        c = country_components(flows, bands, cut)
        nz = c[c.d != 0]
        out[str(cut)] = {
            "n_country_years": int(len(c)), "n_countries": int(c.index.get_level_values(1).nunique()),
            "n_nonzero": int(len(nz)),
            "share_near_dominates": float((nz.near.abs() > nz.far.abs()).mean()),
            "share_far_zero": float((nz.far == 0).mean()),
            "abs": _shares(c, False, False), "abs_demeaned": _shares(c, False, True),
            "rel": _shares(c, True, False), "rel_demeaned": _shares(c, True, True),
            "persistence": persistence(c),
            "big_countries": big_countries(c),
            "horizons": horizons(flows, bands, cut),
            "showcase": {k: {str(int(y)): {c: int(r[c]) for c in ("near", "far", "switch", "n0", "n1")}
                             for y, r in c.xs(k, level="country_label").iterrows()}
                         for k in ("China", "United States", "India", "Russia", "Germany")},
        }
    return out


def persistence(c: pd.DataFrame) -> dict[str, Any]:
    """Correlation of each relative component with its own value a year later, within country (year means removed)."""
    x = c.copy()
    den = (x.n0 + x.n1) / 2 + 0.5
    for k in ("near", "far", "d"):
        x[k] = x[k] / den
        x[k] = x[k] - x.groupby(level="year")[k].transform("mean")
    x = x.reset_index().sort_values(["country_label", "year"])
    out = {}
    for k in ("near", "far", "d"):
        x[k + "_next"] = x.groupby("country_label")[k].shift(-1)
        nxt = x.groupby("country_label").year.shift(-1)
        ok = x[(nxt == x.year + 1)].dropna(subset=[k + "_next"])
        fit = smf.ols(f"{k}_next ~ {k}", ok).fit(cov_type="cluster",
                                                 cov_kwds={"groups": pd.factorize(ok.country_label)[0]})
        out[k] = p3.coef(fit, k)
    return out


def big_countries(c: pd.DataFrame, top: int = 10) -> dict[str, Any]:
    tot = c.groupby(level="country_label").n1.max().sort_values(ascending=False)
    out = {}
    for k in tot.index[:top]:
        g = c.xs(k, level="country_label")
        v = float(g.d.var(ddof=1))
        out[k] = {"near_share": float(np.cov(g.near, g.d)[0, 1] / v), "max_n": int(tot[k]),
                  "sd_change": float(g.d.std(ddof=1))}
    return out


def horizons(flows: pd.DataFrame, bands: pd.DataFrame, cut: int, ks: Sequence[int] = (1, 2, 3, 5, 10)) -> dict[str, Any]:
    """Share of the variance of k-year changes carried by the near-threshold flows, by horizon k."""
    c = country_components(flows, bands, cut).reset_index()
    n = bands.groupby(["year", "country_label"]).n.sum()
    labels = sorted(c.country_label.unique())
    full = pd.MultiIndex.from_product([range(FIRST, LAST), labels], names=["year", "country_label"])
    c = c.set_index(["year", "country_label"])[["near", "far", "switch"]].reindex(full, fill_value=0).reset_index()
    out = {}
    for k in ks:
        rows = []
        for lab, g in c.groupby("country_label"):
            g = g.set_index("year")
            for t in range(FIRST, LAST - k + 1):
                seg = g.loc[t:t + k - 1]
                n0, n1 = n.get((t, lab), 0), n.get((t + k, lab), 0)
                if n0 == 0 and n1 == 0:
                    continue
                rows.append({"near": seg.near.sum(), "far": seg.far.sum(), "switch": seg.switch.sum(),
                             "d": n1 - n0, "den": (n0 + n1) / 2 + 0.5, "t": t})
        x = pd.DataFrame(rows)
        if not np.allclose(x.near + x.far + x.switch, x.d):
            raise ValueError("k-year components do not add up")
        v = float(x.d.var(ddof=1))
        rel = x[["near", "far", "switch", "d"]].div(x.den, axis=0)
        vr = float(rel.d.var(ddof=1))
        out[str(k)] = {"n": int(len(x)), "abs_near": float(np.cov(x.near, x.d)[0, 1] / v),
                       "rel_near": float(np.cov(rel.near, rel.d)[0, 1] / vr),
                       "abs_far": float(np.cov(x.far, x.d)[0, 1] / v),
                       "rel_far": float(np.cov(rel.far, rel.d)[0, 1] / vr)}
    return out


# ---------------------------------------------------------------------------
# 3. Churn
# ---------------------------------------------------------------------------

def churn(flows: pd.DataFrame) -> dict[str, Any]:
    f = flows
    e = f[(f.flow == "entry") & (f.year >= FIRST + 1)]           # a list before year t exists
    x = f[(f.flow == "exit") & (f.year <= LAST - 4)]            # at least three later lists to come back on
    def split(d: pd.DataFrame) -> dict[str, Any]:
        near, far = d[d.band_lo_musd < NEAR], d[d.band_lo_musd >= NEAR]
        return {"n": int(d.n.sum()), "share": float(d.n_linked_elsewhere.sum() / d.n.sum()),
                "near_n": int(near.n.sum()), "near_share": float(near.n_linked_elsewhere.sum() / near.n.sum()),
                "far_n": int(far.n.sum()), "far_share": float(far.n_linked_elsewhere.sum() / far.n.sum())}
    return {"entries_returning": split(e), "exits_returning": split(x),
            "entry_years": [FIRST + 1, LAST - 1], "exit_years": [FIRST, LAST - 4]}


# ---------------------------------------------------------------------------
# 4. A crossing model
# ---------------------------------------------------------------------------

def _grid(lo: float, hi: float, k: int = 25) -> np.ndarray:
    return np.exp(np.linspace(np.log(lo), np.log(hi), k + 1)[:-1] + np.log(hi / lo) / (2 * k))


def crossing_model(flows: pd.DataFrame, bands: pd.DataFrame, glob: pd.DataFrame,
                   manifest: dict[str, Any], stk: dict[str, Any]) -> dict[str, Any]:
    """Apply each year's changes of fortunes above $3bn to the stock near the line."""
    width = float(manifest["logchange_bin"])
    fedges = manifest["flow_edges_musd"]
    fhi = _hi(fedges)
    f = flows
    years = sorted(f.year.unique())
    by_year, by_band_obs, by_band_pred, by_band_stock = {}, {}, {}, {}
    medians, q10, q90 = {}, {}, {}
    for y in years:
        dist = change_dist(glob, y, width)
        medians[str(y)] = dist.quantile(0.5)
        q10[str(y)], q90[str(y)] = dist.quantile(0.1), dist.quantile(0.9)
        g = f[f.year == y]
        st = g[g.flow.isin(["stay", "exit", "switch_out"])].groupby("band_lo_musd").n.sum()
        ex = g[g.flow == "exit"].groupby("band_lo_musd").n.sum()
        pred_exit = {}
        for lo, n in st.items():
            if lo < 1000:
                continue
            w = _grid(float(lo), fhi[int(lo)])
            p = float(np.mean(dist.cdf(np.log10(1000.0 / w))))
            pred_exit[int(lo)] = n * p
            by_band_pred[int(lo)] = by_band_pred.get(int(lo), 0.0) + n * p
            by_band_obs[int(lo)] = by_band_obs.get(int(lo), 0) + int(ex.get(lo, 0))
            by_band_stock[int(lo)] = by_band_stock.get(int(lo), 0) + int(n)
        # Entries: Pareto stock extrapolated below $1bn, moved by the same changes.
        alpha = stk["by_year"][str(y)]["alpha"]
        n_t = stk["by_year"][str(y)]["n"]
        w_edges = 1000.0 * 10 ** np.linspace(-1.0, 0.0, 201)
        mass = n_t * ((w_edges[:-1] / 1000.0) ** -alpha - (w_edges[1:] / 1000.0) ** -alpha)
        w_mid = np.sqrt(w_edges[:-1] * w_edges[1:])
        d_mid, d_p = dist.mids(), dist.p
        new_w = w_mid[:, None] * 10 ** d_mid[None, :]
        flow_mass = mass[:, None] * d_p[None, :]
        crosses = new_w >= 1000.0
        pred_entries = float(flow_mass[crosses].sum())
        pred_entries_near = float(flow_mass[crosses & (new_w < NEAR)].sum())
        en = g[g.flow == "entry"]
        by_year[str(y)] = {
            "exits_obs": int(ex[ex.index >= 1000].sum()), "exits_pred": float(sum(pred_exit.values())),
            "exits_near_obs": int(ex[(ex.index >= 1000) & (ex.index < NEAR)].sum()),
            "exits_near_pred": float(sum(v for k, v in pred_exit.items() if k < NEAR)),
            "entries_obs": int(en.n.sum()), "entries_near_obs": int(en[en.band_lo_musd < NEAR].n.sum()),
            "entries_pred": pred_entries, "entries_near_pred": pred_entries_near,
            "median_change": medians[str(y)],
        }
    rows = pd.DataFrame(by_year).T.astype(float)
    corr = lambda a, b: float(np.corrcoef(rows[a], rows[b])[0, 1])  # noqa: E731
    bands_out = {str(k): {"stock": by_band_stock[k], "obs": by_band_obs[k], "pred": by_band_pred[k],
                          "obs_rate": by_band_obs[k] / by_band_stock[k], "pred_rate": by_band_pred[k] / by_band_stock[k]}
                 for k in sorted(by_band_stock)}
    near_keys = [k for k in by_band_stock if k < NEAR]
    far_keys = [k for k in by_band_stock if k >= FAR]
    return {
        "by_year": by_year, "by_band": bands_out,
        "change_median": medians, "change_q10": q10, "change_q90": q90,
        "exits_total_obs": int(rows.exits_obs.sum()), "exits_total_pred": float(rows.exits_pred.sum()),
        "exits_near_total_obs": int(rows.exits_near_obs.sum()), "exits_near_total_pred": float(rows.exits_near_pred.sum()),
        "entries_total_obs": int(rows.entries_obs.sum()), "entries_total_pred": float(rows.entries_pred.sum()),
        "entries_near_total_obs": int(rows.entries_near_obs.sum()),
        "entries_near_total_pred": float(rows.entries_near_pred.sum()),
        "corr_exits": corr("exits_obs", "exits_pred"), "corr_exits_near": corr("exits_near_obs", "exits_near_pred"),
        "corr_entries": corr("entries_obs", "entries_pred"),
        "corr_entries_near": corr("entries_near_obs", "entries_near_pred"),
        "near_obs_rate": sum(by_band_obs[k] for k in near_keys) / sum(by_band_stock[k] for k in near_keys),
        "near_pred_rate": sum(by_band_pred[k] for k in near_keys) / sum(by_band_stock[k] for k in near_keys),
        "far_obs_rate": sum(by_band_obs[k] for k in far_keys) / sum(by_band_stock[k] for k in far_keys),
        "far_pred_rate": sum(by_band_pred[k] for k in far_keys) / sum(by_band_stock[k] for k in far_keys),
    }


def exits_by_age(glob: pd.DataFrame) -> dict[str, Any]:
    a = glob[glob.table.isin(["exit_age", "stay_age"])].copy()
    a["key"] = a.key.astype(int)
    a["far"] = np.where(a.band_lo_musd >= FAR, "far", np.where(a.band_lo_musd < NEAR, "near", "middle"))
    t = a.groupby(["far", "key", "table"]).n.sum().unstack(fill_value=0)
    t["rate"] = t.exit_age / (t.exit_age + t.stay_age)
    out = {}
    for (grp, key), r in t.iterrows():
        out.setdefault(grp, {})[str(key)] = {"exits": int(r.exit_age), "at_risk": int(r.exit_age + r.stay_age),
                                             "rate": float(r.rate)}
    far = t.loc["far"]
    young = far.loc[[k for k in far.index if 0 <= k < 70]]
    old = far.loc[[k for k in far.index if k >= 85]]
    out["far_rate_under70"] = float(young.exit_age.sum() / (young.exit_age + young.stay_age).sum())
    out["far_rate_85plus"] = float(old.exit_age.sum() / (old.exit_age + old.stay_age).sum())
    out["far_exits_85plus_share"] = float(old.exit_age.sum() / far.exit_age.sum())
    return out


def entries_self_made(glob: pd.DataFrame) -> dict[str, Any]:
    s = glob[glob.table == "entry_selfmade"].copy()
    s["near"] = s.band_lo_musd < NEAR
    t = s.groupby(["near", "key"]).n.sum().unstack(fill_value=0)
    out = {}
    for near, r in t.iterrows():
        known = r.get("yes", 0) + r.get("no", 0)
        out["near" if near else "far"] = {"self_made_share": float(r.get("yes", 0) / known), "n": int(r.sum())}
    return out


# ---------------------------------------------------------------------------
# 5. The count as an indicator
# ---------------------------------------------------------------------------

def thresholds(bands: pd.DataFrame, edges: list[int], model: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Threshold paths: fixed nominal, indexed to the median change of large fortunes, and a fixed global rank."""
    hi = _hi(edges)
    years = list(range(FIRST, LAST + 1))
    out: dict[str, dict[int, float]] = {f"fixed_{t}": {y: float(t) for y in years} for t in FIXED}
    level, path = 0.0, {}
    for y in years:
        path[y] = 1000.0 * 10 ** level
        if str(y) in model["change_median"]:
            level += model["change_median"][str(y)]
    out["indexed"] = path
    g = bands[bands.band_lo_musd >= 1000].groupby(["year", "band_lo_musd"]).n.sum()
    target = float(g.loc[FIRST].sum())
    rank = {}
    for y in years:
        s = g.loc[y]
        lo = s.index.to_numpy(float)
        his = np.array([hi[int(x)] for x in lo])
        n = s.to_numpy(float)
        rank[y] = float(brentq(lambda t: above(n, lo, his, t) - target, 1000.0, 50000.0))
    out["rank"] = rank
    return out


def country_counts(bands: pd.DataFrame, edges: list[int], path: dict[int, float]) -> pd.DataFrame:
    """Counts above each year's threshold; at $1bn, everyone on the list (as in Part 3, including the one
    2017 entry Forbes listed below $1bn)."""
    hi = _hi(edges)
    b = bands.copy()
    b["hi"] = b.band_lo_musd.map(lambda x: hi[int(x)])
    b["iso3"] = b.country_label.map(p3.FORBES_ISO3)
    rows = []
    for (iso, y), g in b.groupby(["iso3", "year"]):
        t = path[int(y)]
        if t <= 1000:
            n = float(g.n.sum())
        else:
            g = g[g.band_lo_musd >= 1000]
            n = above(g.n.to_numpy(float), g.band_lo_musd.to_numpy(float), g.hi.to_numpy(float), t)
        rows.append({"iso3": iso, "year": int(y), "n": n})
    return pd.DataFrame(rows)


def indicator_dynamics(counts: pd.DataFrame, w: pd.DataFrame, iso: list[str]) -> dict[str, Any]:
    full = pd.MultiIndex.from_product([iso, range(FIRST, LAST + 1)], names=["iso3", "year"])
    a = counts.set_index(["iso3", "year"])[["n"]].groupby(level=[0, 1]).sum().reindex(full, fill_value=0.0)
    a = a.reset_index().join(w, on=["iso3", "year"])
    a["B"], a["G"] = p3.density(a.n, a["pop"]), np.log10(a.gdp)
    a = a.sort_values(["iso3", "year"])
    for v in ("B", "G"):
        a["d" + v] = a.groupby("iso3")[v].diff()
        a["d" + v + "_l1"] = a.groupby("iso3")["d" + v].shift(1)
    x = a.dropna(subset=["B", "G"])
    lev = p3.cl("B ~ G + C(iso3) + C(year)", x, "iso3")
    g = x.dropna(subset=["dB", "dB_l1", "dG_l1"])
    ar = p3.cl("dB ~ dB_l1 + dG_l1 + C(year)", g, "iso3")
    v = x.dropna(subset=["dB", "dG"])
    r_year = smf.ols("dB ~ C(year)", v).fit().rsquared
    r_all = smf.ols("dB ~ C(year) + dG", v).fit().rsquared
    rho = float(ar.params["dB_l1"])
    return {"n": int(len(x)), "n_countries": int(x.iso3.nunique()), "levels_G": p3.coef(lev, "G"),
            "dB_on_dB_lag": p3.coef(ar, "dB_l1"), "noise_share": float(min(max(-2 * rho, 0.0), 1.0)),
            "sd_dB": float(x.dB.std()), "r2_year": float(r_year), "r2_year_growth": float(r_all),
            "share_zero": float((x.n < 0.5).mean()), "median_n_positive": float(x.n[x.n >= 0.5].median()),
            "global_2010": float(x[x.year == FIRST].n.sum()), "global_2025": float(x[x.year == LAST].n.sum())}


def indicators(inp: dict[str, Any], model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    edges = inp["manifest"]["band_edges_musd"]
    bands = inp["bands"]
    paths = thresholds(bands, edges, model)
    w = p3.wdi_wide(inp["p3"]["wdi"])
    cit = p3.forbes_counts(inp["p3"]["forbes"]).reset_index()
    ever = cit.groupby("iso3").n.max()
    iso = sorted(ever[ever > 0].index)
    out: dict[str, Any] = {"paths": {k: {str(y): v for y, v in p.items()} for k, p in paths.items()}}
    counts = {}
    for k, path in paths.items():
        counts[k] = country_counts(bands, edges, path)
        out[k] = indicator_dynamics(counts[k], w, iso)
    world = bands.groupby("year").n.sum()
    hi = _hi(edges)
    gb = bands[bands.band_lo_musd >= 1000].groupby(["year", "band_lo_musd"]).n.sum()
    glob_n = {}
    for k, path in paths.items():
        glob_n[k] = {}
        for y in range(FIRST, LAST + 1):
            sy = gb.loc[y]
            lo = sy.index.to_numpy(float)
            glob_n[k][str(y)] = (float(world[y]) if path[y] <= 1000 else
                                 above(sy.to_numpy(float), lo, np.array([hi[int(x)] for x in lo]), path[y]))
    out["global_counts"] = glob_n
    out["global_growth"] = {k: {"2010": v[str(FIRST)], "2025": v[str(LAST)], "factor": v[str(LAST)] / v[str(FIRST)]}
                            for k, v in glob_n.items()}
    out["log_share_of_growth_from_fixed_line"] = float(
        1 - np.log(out["global_growth"]["indexed"]["factor"]) / np.log(out["global_growth"]["fixed_1000"]["factor"]))
    out["global_n"] = {str(int(y)): int(v) for y, v in world.items()}
    return out, counts


def poverty_rerun(inp: dict[str, Any], counts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    panel, _, _ = p3.build_panel(inp["p3"])
    out = {}
    for k, c in counts.items():
        m = c.groupby(["iso3", "year"]).n.sum()
        d = panel.copy()
        d["n_alt"] = [float(m.get((i, y), 0.0)) for i, y in zip(d.country_code, d.year)]
        d["Bx"] = p3.density(d.n_alt, d["pop"])
        within = p3.coef(p3.cl("p_30 ~ Bx + G + C(spell) + C(year)", d), "Bx")
        pooled = p3.coef(p3.cl("p_30 ~ Bx + G + C(year)", d), "Bx")
        rows = []
        for sp, g in d.groupby("spell"):
            if len(g) >= 2 and g.year.iloc[-1] - g.year.iloc[0] >= 5:
                a, b = g.iloc[0], g.iloc[-1]
                rows.append({"iso3": a.country_code, "dp": b.p_30 - a.p_30, "dB": b.Bx - a.Bx, "dG": b.G - a.G})
        ld = pd.DataFrame(rows)
        long = p3.coef(p3.cl("dp ~ dB + dG", ld, "iso3"), "dB")
        dm = d.groupby("spell")[["Bx", "B"]].transform("mean")
        out[k] = {"within": within, "pooled": pooled, "long": long,
                  "corr_within_with_1bn": float(np.corrcoef(d.Bx - dm.Bx, d.B - dm.B)[0, 1])}
    return out


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------

def compute(inp: dict[str, Any]) -> dict[str, Any]:
    bands, flows, glob, man = inp["bands"], inp["flows"], inp["global"], inp["manifest"]
    flows = flows[flows.year.between(FIRST, LAST - 1)]
    check = check_against_part3(bands, inp["p3"]["forbes"])
    stk = stock(bands, man["band_edges_musd"])
    model = crossing_model(flows, bands, glob, man, stk)
    ind, counts = indicators(inp, model)
    p3n = json.loads((HERE / "results" / "part3" / "numbers.json").read_text(encoding="utf-8"))
    base = ind["fixed_1000"]["dB_on_dB_lag"]["coef"]
    if abs(base - p3n["annual"]["dB_on_dB_lag"]["coef"]) > 1e-9:
        raise ValueError("the $1bn indicator does not reproduce Part 3's annual autocorrelation")
    pairs_links = {p: e["links_by_rule"] for p, e in man["pairs"].items()}
    return {
        "check_part3": check,
        "fetch": {"years": {y: e.get("n_persons") for y, e in man["years"].items()},
                  "failed": sorted(y for y, e in man["years"].items() if "error" in e),
                  "links_by_rule": {r: int(sum(v.get(r, 0) for v in pairs_links.values()))
                                    for r in ("uri", "natural", "name")},
                  "missing_age": {y: e.get("missing_age") for y, e in man["years"].items() if "error" not in e}},
        "stock": stk,
        "gross": gross_flows(flows, bands),
        "global": global_decomposition(flows),
        "global_wide": global_decomposition(flows, NEAR_WIDE),
        "country": country_decomposition(flows, bands),
        "churn": churn(flows),
        "model": model,
        "age": exits_by_age(glob[glob.year.between(FIRST, LAST - 1)]),
        "self_made": entries_self_made(glob[glob.year.between(FIRST, LAST - 1)]),
        "indicators": ind,
        "poverty": poverty_rerun(inp, {k: counts[k] for k in ("fixed_1000", "fixed_2000", "indexed", "rank")}),
        "part3_annual": {"dB_on_dB_lag": p3n["annual"]["dB_on_dB_lag"]["coef"], "sd_dB": p3n["annual"]["sd_dB"]},
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def frame(fig: plt.Figure, title: str, subtitle: str, **kw: Any) -> None:
    _frame(fig, title, subtitle, source=SOURCE, **kw)


def _bn(v: float, _: Any = None) -> str:
    return f"${v / 1000:g}bn"


def fig_exit_rates(n: dict[str, Any], out: Path) -> None:
    m, edges = n["model"]["by_band"], n["flow_edges"]
    hi = {lo: h for lo, h in zip(edges, edges[1:])} | {edges[-1]: edges[-1] * 2.5}
    los = sorted(int(k) for k in m)
    x = np.array([np.sqrt(lo * hi[lo]) for lo in los])
    obs = np.array([m[str(k)]["obs_rate"] for k in los]) * 100
    pred = np.array([m[str(k)]["pred_rate"] for k in los]) * 100
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, pred, color=ORANGE, lw=2.2, marker="o", ms=5, label="Crossing model: this year's moves of fortunes above $3bn")
    ax.plot(x, obs, color=BLUE, lw=0, marker="o", ms=8, label="Observed share leaving the list")
    far = n["model"]["far_obs_rate"] * 100
    ax.axhline(far, color=BASELINE, lw=1, ls="--")
    ax.text(26000, far + 1.0, f"Observed above $3bn: {fmt(far, 1)}% a year", color=SECONDARY, fontsize=9.5, ha="right")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1000, 1500, 2000, 3000, 5000, 10000, 20000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(_bn))
    ax.set_xlim(950, 40000)
    ax.set_ylim(0, 42)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Net worth on this year's list (log scale)")
    ax.set_ylabel("Share not on next year's list")
    ax.legend(loc="upper right", fontsize=9.5)
    frame(fig, "Who leaves the list is decided mostly by distance from $1 billion",
          "Share of people on each list who are not on the next one, by net-worth band, 2010-2024 lists pooled.\n"
          "The model applies each year's changes in large fortunes to everyone; it has no deaths or reassessments.",
          top=0.83, bottom=0.15, left=0.09)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_global_years(n: dict[str, Any], out: Path) -> None:
    g = n["global"]["by_year"]
    years = sorted(int(y) for y in g)
    near = np.array([g[str(y)]["net_near"] for y in years])
    far = np.array([g[str(y)]["net_far"] for y in years])
    net = np.array([g[str(y)]["net"] for y in years])
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - 0.2, near, width=0.38, color=BLUE, label="Net entries below $1.5bn")
    ax.bar(x + 0.2, far, width=0.38, color=ORANGE, label="Net entries at $1.5bn or more")
    ax.plot(x, net, lw=0, marker="D", ms=6, color=INK, label="Change in the global count")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}–{str(y + 1)[2:]}" for y in years], rotation=0, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v, 0, sign=v != 0)))
    ax.set_ylabel("People")
    ax.legend(loc="upper left", fontsize=9.5)
    frame(fig, "Booms and busts move the list through both groups",
          "Net entries to the Forbes list from one year to the next, split by net worth at entry or exit.",
          top=0.86, bottom=0.12, left=0.08)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_variance_curve(n: dict[str, Any], out: Path) -> None:
    c, gr = n["country"]["curve"], n["gross"]["near_shares"]
    cuts = [int(k) for k in c]
    one = np.array([c[str(k)]["abs"]["near"] for k in cuts]) * 100
    stock_share = np.array([gr[str(k)]["stock_t"] for k in cuts]) * 100
    ten = {int(k): v for k, v in (("1500", n["country"]["1500"]["horizons"]["10"]),
                                   ("2000", n["country"]["2000"]["horizons"]["10"]))}
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(cuts, one, color=BLUE, lw=2.4, marker="o", ms=5)
    ax.plot(cuts, stock_share, color=MUTED, lw=2, ls="--")
    ax.plot(list(ten), [v["abs_near"] * 100 for v in ten.values()], lw=0, marker="s", ms=8, color=AQUA)
    ax.text(cuts[-1] + 40, one[-1], "Share of the variance of\none-year country changes", color=BLUE, fontsize=9.5,
            va="center")
    ax.text(cuts[-1] + 40, stock_share[-1], "Share of the people\non the list", color=SECONDARY, fontsize=9.5, va="center")
    ax.text(2040, ten[2000]["abs_near"] * 100 - 7, "Ten-year changes", color=AQUA, fontsize=9.5)
    ax.set_xlim(1050, 3650)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(FixedLocator([1100, 1500, 2000, 2500, 3000]))
    ax.xaxis.set_major_formatter(FuncFormatter(_bn))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Where 'near the threshold' ends: fortunes below this value")
    ax.set_ylabel("Share")
    frame(fig, "Where 'near the line' ends decides how much of the change is near it",
          "Share of the variance of country-level changes in the number of billionaires carried by entries and exits\n"
          "below each cutoff, against the share of all billionaires below it. 91 countries, 2010-2025.",
          top=0.83, bottom=0.14, left=0.08, right=0.82)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_model_years(n: dict[str, Any], out: Path) -> None:
    by = n["model"]["by_year"]
    years = sorted(int(y) for y in by)
    x = np.arange(len(years))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=False)
    for ax, (key, title) in zip(axes, (("exits", "Leaving the list"), ("entries", "Joining the list"))):
        obs = np.array([by[str(y)][f"{key}_obs"] for y in years])
        pred = np.array([by[str(y)][f"{key}_pred"] for y in years])
        ax.bar(x, obs, width=0.7, color=BLUE, label="Observed")
        ax.plot(x, pred, color=ORANGE, lw=2, marker="o", ms=5, label="Crossing model")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"{str(y)[2:]}–{str(y + 1)[2:]}" for y in years][::2], fontsize=9.5)
        ax.set_xlabel("From one list to the next (2010–11 to 2024–25)")
        ax.set_ylabel("People")
    axes[0].legend(loc="upper left", fontsize=9.5)
    m = n["model"]
    frame(fig, "A crossing model gets the timing right and the levels partly wrong",
          f"Observed entries and exits by year against the model. Correlation over the years: exits "
          f"{m['corr_exits']:.2f}, entries {m['corr_entries']:.2f}.\nEntries are modelled from a Pareto stock "
          "extrapolated below $1bn, so their level rests on that assumption.",
          top=0.81, bottom=0.12, left=0.07)
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(out, dpi=250)
    plt.close(fig)


INDICATOR_LABELS = {"fixed_1000": "$1bn, as listed", "fixed_1500": "$1.5bn", "fixed_2000": "$2bn",
                    "fixed_3000": "$3bn", "fixed_5000": "$5bn",
                    "indexed": "Indexed to large fortunes", "rank": "Top 1,011 in the world"}


def fig_indicators(n: dict[str, Any], out: Path) -> None:
    ind = n["indicators"]
    keys = list(INDICATOR_LABELS)
    y = np.arange(len(keys))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    for yi, k in zip(y, keys):
        r = ind[k]["dB_on_dB_lag"]
        col = ORANGE if k in ("indexed", "rank") else BLUE
        axes[0].plot([-r["ci95"][1], -r["ci95"][0]], [yi, yi], color=col, lw=2.2)
        axes[0].plot(-r["coef"], yi, "o", color=col, ms=8)
        f = ind["global_growth"][k]["factor"]
        axes[1].plot(f, yi, "o", color=col, ms=8)
        axes[1].text(f + 0.12, yi, f"×{f:.2f}", va="center", fontsize=9.5, color=SECONDARY)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([INDICATOR_LABELS[k] for k in keys])
    axes[0].set_xlim(0, 0.45)
    axes[0].xaxis.set_major_locator(FixedLocator([0, 0.1, 0.2, 0.3, 0.4]))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    axes[0].set_xlabel("Share of a year's change reversed the next year")
    axes[1].set_xlim(0.7, 5.2)
    axes[1].axvline(1, color=BASELINE, lw=1)
    axes[1].set_xlabel("Global count in 2025 relative to 2010")
    axes[0].set_title("Within-country reversal, with 95% interval", loc="left", fontsize=11, color=INK)
    axes[1].set_title("Growth of the global count", loc="left", fontsize=11, color=INK)
    frame(fig, "Raising the line does not quiet the count; moving it with large fortunes does",
          "Counts of people above different thresholds in 83 countries, 2010-2025. The indexed threshold grows with the\n"
          "median change of fortunes above $3bn and reaches $1.87bn in 2025; the rank threshold keeps 1,011 people.",
          top=0.80, bottom=0.12, left=0.19)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out, dpi=250)
    plt.close(fig)


FIGURE_FILES = ("fig1_exit_rates.png", "fig2_global_years.png", "fig3_variance_curve.png",
                "fig4_model_years.png", "fig5_indicators.png")


def render(inp: dict[str, Any], n: dict[str, Any], outdir: Path) -> None:
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    n = dict(n, flow_edges=[e for e in inp["manifest"]["flow_edges_musd"] if e >= 1000])
    fig_exit_rates(n, outdir / FIGURE_FILES[0])
    fig_global_years(n, outdir / FIGURE_FILES[1])
    fig_variance_curve(n, outdir / FIGURE_FILES[2])
    fig_model_years(n, outdir / FIGURE_FILES[3])
    fig_indicators(n, outdir / FIGURE_FILES[4])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _c(r: dict[str, Any], k: int = 2) -> str:
    return f"{fmt(r['coef'], k, sign=True)} [{fmt(r['ci95'][0], k)}, {fmt(r['ci95'][1], k)}] p {r.get('p', float('nan')):.3f}"


def summary_lines(n: dict[str, Any]) -> list[str]:
    s, g, gl, c, ch, m, ind = (n[k] for k in ("stock", "gross", "global", "country", "churn", "model", "indicators"))
    r3 = lambda d: {k: round(v, 3) for k, v in d.items()}  # noqa: E731
    out = [f"Check against Part 3: {n['check_part3']}; fetch failed {n['fetch']['failed']}; links {n['fetch']['links_by_rule']}",
           f"Stock alpha range {r3(dict(zip(('min', 'max'), s['alpha_range'])))}; share below 1.5bn range "
           f"{[round(x, 3) for x in s['share_below_1500_range']]}; 2010 {r3(s['by_year']['2010']['share_below'])} "
           f"alpha {s['by_year']['2010']['alpha']:.3f}; 2025 {r3(s['by_year']['2025']['share_below'])} alpha {s['by_year']['2025']['alpha']:.3f}",
           f"Gross: entries {g['total_entries']} exits {g['total_exits']} switches {g['total_switches']} person-years {g['person_years_at_risk']}; "
           f"exit rate near {g['exit_rate_near']:.3f} far {g['exit_rate_far']:.3f}",
           "  exit rate by band: " + "; ".join(f"{k} {v:.3f}" for k, v in g["exit_rate_by_band"].items()),
           "  near shares: " + "; ".join(f"<{k}: entries {v['entries']:.3f} exits {v['exits']:.3f} stock {v['stock_t']:.3f}"
                                        for k, v in g["near_shares"].items()),
           f"Global by year (near<1.5bn): var share near {gl['var_share_near']:.3f} far {gl['var_share_far']:.3f}; sums near "
           f"{gl['sum_net_near']} far {gl['sum_net_far']} net {gl['sum_net']}; wide var share near {n['global_wide']['var_share_near']:.3f}"]
    for y, v in gl["by_year"].items():
        out.append(f"  {y}->{int(y) + 1}: {v}")
    out.append("Country variance shares by cut (abs / rel / rel demeaned, near): " + "; ".join(
        f"<{k} {v['abs']['near']:.3f}/{v['rel']['near']:.3f}/{v['rel_demeaned']['near']:.3f}" for k, v in c["curve"].items()))
    for cut in (str(NEAR), str(NEAR_WIDE)):
        v = c[cut]
        out.append(f"Cut {cut}: {v['n_country_years']} country-years, {v['n_countries']} countries, nonzero {v['n_nonzero']}, "
                   f"near dominates {v['share_near_dominates']:.3f}, far zero {v['share_far_zero']:.3f}; abs {r3(v['abs'])}; "
                   f"abs dm {r3(v['abs_demeaned'])}; rel {r3(v['rel'])}; rel dm {r3(v['rel_demeaned'])}")
        out.append("  persistence: " + "; ".join(f"{k} {_c(x, 3)}" for k, x in v["persistence"].items()))
        out.append("  horizons: " + "; ".join(f"k{k} n {x['n']} abs {x['abs_near']:.3f} rel {x['rel_near']:.3f} far rel {x['rel_far']:.3f}"
                                            for k, x in v["horizons"].items()))
        out.append("  big countries: " + "; ".join(f"{k} {x['near_share']:.2f} (max {x['max_n']}, sd {x['sd_change']:.1f})"
                                                for k, x in v["big_countries"].items()))
    out.append(f"Churn: entries returning {r3({k: v for k, v in ch['entries_returning'].items() if 'share' in k})} "
               f"(n {ch['entries_returning']['n']}, years {ch['entry_years']}); exits returning "
               f"{r3({k: v for k, v in ch['exits_returning'].items() if 'share' in k})} (n {ch['exits_returning']['n']}, years {ch['exit_years']})")
    out.append(f"Model: exits obs {m['exits_total_obs']} pred {m['exits_total_pred']:.0f}; near obs {m['exits_near_total_obs']} pred "
               f"{m['exits_near_total_pred']:.0f}; entries obs {m['entries_total_obs']} pred {m['entries_total_pred']:.0f}; near obs "
               f"{m['entries_near_total_obs']} pred {m['entries_near_total_pred']:.0f}; corr exits {m['corr_exits']:.3f} near "
               f"{m['corr_exits_near']:.3f} entries {m['corr_entries']:.3f} near {m['corr_entries_near']:.3f}; rates near obs "
               f"{m['near_obs_rate']:.3f} pred {m['near_pred_rate']:.3f}; far obs {m['far_obs_rate']:.4f} pred {m['far_pred_rate']:.4f}")
    out.append("  by band (obs/pred rate): " + "; ".join(f"{k} {v['obs_rate']:.3f}/{v['pred_rate']:.3f}" for k, v in m["by_band"].items()))
    for y, v in m["by_year"].items():
        out.append(f"  {y}: " + ", ".join(f"{k} {x:.3f}" if isinstance(x, float) else f"{k} {x}" for k, x in v.items()))
    out.append(f"  change q10 {r3(m['change_q10'])}")
    out.append(f"  change q90 {r3(m['change_q90'])}")
    a = n["age"]
    out.append(f"Age: far exit rate under 70 {a['far_rate_under70']:.4f}; 85+ {a['far_rate_85plus']:.4f}; far exits aged 85+ "
               f"{a['far_exits_85plus_share']:.3f}; near {({k: round(v['rate'], 3) for k, v in a['near'].items()})}; far "
               f"{({k: round(v['rate'], 3) for k, v in a['far'].items()})}")
    out.append(f"Self-made entrants: {n['self_made']}")
    out.append("Global growth 2010->2025 (all countries): " + "; ".join(f"{k} {v['2010']:.0f}->{v['2025']:.0f} x{v['factor']:.2f}"
                                                                      for k, v in ind["global_growth"].items()))
    out.append(f"Share of log growth of the global count from the fixed line: {ind['log_share_of_growth_from_fixed_line']:.3f}")
    out.append("Thresholds 2025: " + "; ".join(f"{k} {v['2025']:.0f}" for k, v in ind["paths"].items()))
    for k in ind["paths"]:
        v = ind[k]
        out.append(f"Indicator {k}: n {v['n']} countries {v['n_countries']} levels G {_c(v['levels_G'])}; dB lag "
                   f"{_c(v['dB_on_dB_lag'], 3)}; noise share {v['noise_share']:.3f}; sd dB {v['sd_dB']:.3f}; R2 year "
                   f"{v['r2_year']:.3f} +growth {v['r2_year_growth']:.3f}; zero {v['share_zero']:.3f}; median n "
                   f"{v['median_n_positive']:.1f}; global {v['global_2010']:.0f}->{v['global_2025']:.0f}")
    for k, v in n["poverty"].items():
        out.append(f"Poverty {k}: within {_c(v['within'])} pooled {_c(v['pooled'])} long {_c(v['long'])} corr {v['corr_within_with_1bn']:.3f}")
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
        print("\nREPRODUCED: every number matches results/part4/numbers.json")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "summary.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    if not args.no_figures:
        render(inp, numbers, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
